# Experiment Ideas



## Exploiting Symmetry in Chess Puzzles to elicit reasoning strategies

- Any valid puzzle can be reflected vertically to create a new valid puzzle.
- Puzzles without pawns can be both rotated and reflected to create new valid puzzles
    as all non-pawn pieces have rotationally symmetric move sets. This means we can
    turn each puzzle into 4 puzzles by rotating the board 90 degrees. We can also flip the board horizontally and vertically, so we can turn each puzzle into 8 puzzles and solutions.
- We can then compare the reasoning strategies of the model on these puzzles to the
    reasoning strategies on the original puzzles.

### Questions

1. Do the models achieve the same accuracy on the symmetric puzzles as on the original
    puzzles?
2. Can we identify reasoning strategies that are invariant under these transformations?
3. Can we identify reasoning strategies that are not invariant under these transformations?
4. Can we identify reasoning strategies that are only present in the symmetric puzzles?
