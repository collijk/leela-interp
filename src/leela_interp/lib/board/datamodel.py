from typing import NamedTuple

from chess import (
    Bitboard,
    Color,
    Square,
)


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
    # The halfmove clock (number of halfmoves since the last pawn move or capture)
    rule50_count: int
