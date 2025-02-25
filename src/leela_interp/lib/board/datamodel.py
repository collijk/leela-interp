import struct
from collections import Counter
from collections.abc import Iterator
from itertools import product
from typing import NamedTuple

import chess
import numpy as np
import numpy.typing as npt
from chess import (
    Bitboard,
    Color,
    Square,
)

# Enumerate constant masks we can reference when creating board states
_FLAT_PLANES: list[npt.NDArray[np.uint8]] = [
    i * np.ones((8, 8), dtype=np.uint8) for i in range(256)
]


class TranspositionKey(NamedTuple):
    """A hashable representation of a chess board state.

    This is a collection of bitboards and other info representing the state of
    the board. Each bitboard represents a different aspect of the board state:

        - Squares occupied by each piece type
        - Squares occupied by each player
        - Which player is to move
        - Castling rights
        - En passant square if one is available

    """

    # Bitboards for each piece type
    pawns: Bitboard
    knights: Bitboard
    bishops: Bitboard
    rooks: Bitboard
    queens: Bitboard
    kings: Bitboard
    # Bitboards for each player
    white: Bitboard
    black: Bitboard
    # Which player is to move
    side_to_move: Color
    # Castling rights
    castling_rights: Bitboard
    # En passant square if one is available
    en_passant: Square | None

    @property
    def color_bitmasks(self) -> list[Bitboard]:
        """Return the bitmasks for each player in canonical order."""
        return [self.white, self.black]

    @property
    def piece_type_bitmasks(self) -> list[Bitboard]:
        """Return the bitmasks for each piece type in canonical order."""
        return [
            self.pawns,
            self.knights,
            self.bishops,
            self.rooks,
            self.queens,
            self.kings,
        ]

    @property
    def piece_bitmasks(self) -> list[Bitboard]:
        """Return the bitmasks for each color and piece in canonical order."""
        return [
            color & piece
            for color, piece in product(self.color_bitmasks, self.piece_type_bitmasks)
        ]


class LeelaBoardState:
    """Data for a Leela Chess Zero board state."""

    _struct = struct.Struct(">Q")

    def __init__(
        self,
        transposition_key: TranspositionKey,
        repetitions: int,
        rule50_count: int,
    ) -> None:
        self.key = transposition_key
        self.repetitions = repetitions
        self.rule50_count = rule50_count

    def position_planes(self, as_side: Color | None = None) -> npt.NDArray[np.uint8]:
        """Convert this board state to the 13 planes used by Leela Chess Zero."""
        as_side = self.key.side_to_move if as_side is None else as_side

        plane_bytes = b"".join(self._struct.pack(bm) for bm in self.key.piece_bitmasks)
        # The lc0 input respresentation of the piece positions is a stack of
        # 12 8x8 planes, each representing a different piece type for both players
        # Here we unpack the byte representation and flip the x and y axes to match
        # the board representation
        planes = np.unpackbits(memoryview(plane_bytes)).reshape(12, 8, 8)[:, ::-1, ::-1]  # type: ignore[arg-type]
        if as_side == chess.BLACK:
            # Flip the planes to represent the board from the black player's perspective
            # We first flip the color axis, and then the horizontal axis. We don't need
            # to flip the vertical axis as the king side is the same for both players
            planes = planes.reshape(2, 6, 8, 8)[::-1, :, ::-1].reshape(12, 8, 8)

        # The last plane is a constant plane that represents whether this board state
        # is a repetition of a prior board state
        is_repitition = np.full((1, 8, 8), self.repetitions > 1, dtype=np.uint8)

        planes = np.concatenate([planes, is_repitition], axis=0)
        return planes

    def supplementary_planes(
        self,
        as_side: Color | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Create the supplementary planes for this board state."""
        as_side = self.key.side_to_move if as_side is None else as_side
        # The supplementary planes are a stack of 8 8x8 planes that represent
        # castling rights, which side is to move, and the rule 50 count
        _c = self.key.castling_rights
        white = (_c >> chess.A1) & 1, (_c >> chess.H1) & 1
        black = (_c >> chess.A8) & 1, (_c >> chess.H8) & 1

        if as_side == chess.WHITE:
            us_queenside, us_kingside = white
            them_queenside, them_kingside = black
        else:
            us_queenside, us_kingside = black
            them_queenside, them_kingside = white

        planes = np.array(
            [
                _FLAT_PLANES[us_queenside],
                _FLAT_PLANES[us_kingside],
                _FLAT_PLANES[them_queenside],
                _FLAT_PLANES[them_kingside],
                # 0 if white, 1 if black, opposite of board repr
                _FLAT_PLANES[1 - as_side],
                _FLAT_PLANES[self.rule50_count],
                _FLAT_PLANES[0],
                _FLAT_PLANES[1],
            ],
            dtype=np.uint8,
        )
        return planes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.key}, repetitions={self.repetitions}, rule50_count={self.rule50_count})"


class LeelaBoardData(NamedTuple):
    """Data for a Leela Chess Zero board state."""

    # A byte string representing the board state
    plane_bytes: bytes
    # Whether this board state is a repetition of a prior board state
    repetition: bool
    # The transposition key representation of the board state
    transposition_key: TranspositionKey
    # Whether queenside castling is available for the current player
    us_ooo: int
    # Whether kingside castling is available for the current player
    us_oo: int
    # Whether queenside castling is available for the opponent
    them_ooo: int
    # Whether kingside castling is available for the opponent
    them_oo: int
    # Who is to move
    side_to_move: int
    rule50_count: int


class LeelaBoardStates:
    def __init__(self) -> None:
        self._stack: list[LeelaBoardState] = []
        self._transposition_counter: Counter[TranspositionKey] = Counter()

    def push(
        self,
        transposition_key: TranspositionKey,
        halfmove_clock: int,
    ) -> None:
        """Push a board state onto the stack."""
        self._transposition_counter.update((transposition_key,))
        repetitions = self._transposition_counter[transposition_key] - 1
        new_state = LeelaBoardState(
            transposition_key=transposition_key,
            repetitions=repetitions,
            rule50_count=halfmove_clock,
        )
        self._stack.append(new_state)

    def pop(self) -> LeelaBoardState:
        """Pop a board state from the stack."""
        last_state = self._stack.pop()
        self._transposition_counter.subtract([last_state.key])
        return last_state

    def to_lc0(
        self, as_side: Color | None = None, *, no_history: bool = False
    ) -> npt.NDArray[np.uint8]:
        """Convert the current board state to the 112 planes used by Leela Chess Zero."""
        n_planes = 112
        n_supplementary_planes = 8
        n_position_planes = n_planes - n_supplementary_planes

        last_state = self._stack[-1]
        as_side = last_state.key.side_to_move if as_side is None else as_side

        planes_stack = []
        n_planes_yielded = 0
        for state in self._stack[::-1]:
            planes = state.position_planes(as_side)
            planes_stack.append(planes)
            n_planes_yielded += planes.shape[0]

            if n_planes_yielded >= n_position_planes:
                break

        if n_planes_yielded < n_position_planes:
            planes_stack.append(
                np.concatenate(
                    [
                        _FLAT_PLANES[0]
                        for _ in range(n_position_planes - n_planes_yielded)
                    ]
                )
            )

        supplementary_planes = last_state.supplementary_planes(as_side)
        planes_stack.append(supplementary_planes)

        planes = np.concatenate(planes_stack)

        if no_history:
            end_last_state = planes_stack[0].shape[0] - 1  # 13 - 1
            planes[end_last_state:n_position_planes] = 0

        return planes

    def __len__(self) -> int:
        return len(self._stack)

    def __getitem__(self, index: int) -> LeelaBoardState:
        return self._stack[index]

    def __iter__(self) -> Iterator[LeelaBoardState]:
        return iter(self._stack)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._stack})"
