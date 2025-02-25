# What are the different models?

## lc0

The lc0 model is a finetuned version of #lc0-original. The finetuning was done
specifically to make the model not pay attention to the board state history.

## LD2

## lc0-original

## lc0-random

# What are the model inputs?

The input shape for the model is 112x8x8 where each of the 112 channels represents
a different feature of the board state and the 8x8 grid represents the board squares
themselves.

## Channels

The feature channels consiste of 8 groups of 13 channels each with an additional
8 channels providing additional board state. The 8 groups of 13 channels represent the
board state history in reverse order (ie the most recent board state is the first 13
channels, the second most recent is the second 13 channels, etc). For a single board
state the 13 channels represent the following features:

- 1-6: The board state for the current player for each of the 6 piece types
- 7-12: The board state for the opponent player for each of the 6 piece types
- 13: Whether this board state is a repetition of a previous board state

The pieces are ordered as follows:

1. Pawn
2. Knight
3. Bishop
4. Rook
5. Queen
6. King

Each channel is a binary channel where a 1 represents the presence of a piece and a 0
represents the absence of a piece.

Channel 13 is a constant mask, all 1s if the board state is a repetition of a previous
board state and all 0s otherwise.

The additional 8 channels are all constant masks and represent the following features:

1. Whether the current player has queenside castling rights
2. Whether the current player has kingside castling rights
3. Whether the opponent player has queenside castling rights
4. Whether the opponent player has kingside castling rights
5. 0 if the current player is white, 1 if the current player is black (note that this is
   the opposite of the player to move in the board state)
6. The halfmove clock, i.e. the number of plys since the last pawn move or capture
7. A padding channel of all 0s
8. A padding channel of all 1s
