# What is the chess data?

When we load a chess puzzle, what do we find?

## Raw Lichess DB

- `PuzzleId` (str): Uniqe identifier for the puzzle.
- `FEN` (str): The board state in
    [Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).
    The board state represents the position of the pieces before the opponent makes
    their move. The position to present to the player is after applying the first move
    to the FEN, so the second move is the beginning of the solution.
- `Moves` (str): The optimal solution to the puzzle in
    [UCI notation](https://en.wikipedia.org/wiki/Universal_Chess_Interface). Exceptions
    are made for puzzles with the `"mateIn1"` theme, where there may be multiple
    solutions.
- `Rating` (int): The difficulty rating of the puzzle.
- `RatingDeviation` (int): The uncertainty of the difficulty rating.
- `Popularity` (int): This is a player rating from the Lichess database. It takes values
    between -100 (worst) and 100 (best). It is calculated as `100 * (upvotes - downvotes) / (upvotes + downvotes)`.
    Votes are weighted by various factors such as whether the puzzle was successfully
    solve or the solver puzzle rating in comparison to the puzzle rating.
- `NbPlays` (int): The number of times the puzzle has been attempted.
- `Themes` (str): A list of themes that the puzzle belongs to. Themes are used to
    categorize puzzles and are used in the Lichess puzzle interface. a list of themes
    can be found in this [file](https://github.com/lichess-org/lila/blob/master/translation/source/puzzleTheme.xml)
- `GameUrl` (str): The URL of the game where the puzzle was extracted from.
- `OpeningTags` (str): The opening tags of the game where the puzzle was extracted from.
    Opening tags are only set for puzzles starting before move 20. A list of possible
    opening tags can be found in this [repository](https://github.com/lichess-org/chess-openings)

## Additional Fields

These fields are present in the `unfiltered` dataset and the `interesting` dataset.
Puzzles with a `mateIn1` theme are, contrary to the dataset name, filtered from
the `unfiltered` dataset (in addition to being filtered from the `interesting` dataset).

- `principal_variation` (list of str): This is the optimal move set starting with the
    player's move. It is computed directly from the `Moves` field by converting to
    a list and dropping the initial opponent move.
- `full_pv_probs` (list of float): The conditional probabilities of the optimal move
    set. This is computed by taking the optimal move set and computing the probability
    of each move given the previous moves.
- `full_model_moves` (list of str): The conditional set of moves that the model would
    make given the previous moves. This is computed by taking the optimal move set and
    computing the model move for each move given the previous moves.
- `full_wdl` (list of str): The model's predicted win-draw-loss probabilities at the
    start of the puzzle.
- `sparring_full_pv_probs` (list of float): Same as `full_pv_probs` but for the `LD2`
    model rather than the `lc0` model.
- `sparring_full_model_moves` (list of str): Same as `full_model_moves` but for the
    `LD2` model rather than the `lc0` model.
- `sparring_wdl` (list of str): Same as `full_wdl` but for the `LD2` model rather than
    the `lc0` model.

These fields are only present in the `interesting` dataset.

- `different_targets` (bool): Whether the target square of the optimal move set is
    different for each of the first three moves of the principal variation.
- `corrupted_fen` (str): The FEN of a board state after the original board has been
    corrupted. The corruption is done by either adding/removing a single pawn or by
    moving a single non-pawn piece to an empty square. For each puzzle, all corruptions
    are generated and then a best corruption is selected based on the model's predicted
    win-draw-loss probabilities.
