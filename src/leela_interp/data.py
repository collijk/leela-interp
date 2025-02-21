import pickle
from pathlib import Path

from leela_interp.shell_tools import mkdir
from leela_interp import constants as lic


class LeelaData:

    def __init__(self, root: str | Path = lic.DATA_ROOT):
        self._root = Path(root)
        self._create_data_root()

    def _create_data_root(self):
        mkdir(self.root, exist_ok=True)
        mkdir(self.puzzles, exist_ok=True)
        mkdir(self.models, exist_ok=True)

    @property
    def root(self) -> str:
        return self._root

    @property
    def puzzles(self) -> Path:
        return self.root / "puzzles"

    def puzzles_path(self, puzzle_group: str) -> Path:
        return self.puzzles / f"{puzzle_group}.pkl"

    def load_puzzles(self, puzzle_group: str) -> list[str]:
        path = self.puzzles_path(puzzle_group)
        with path.open("rb") as f:
            return pickle.load(f)

    @property
    def models(self) -> Path:
        return self.root / "models"

    def model_path(self, model_name: str) -> Path:
        return self.models / f"{model_name}.onnx"

