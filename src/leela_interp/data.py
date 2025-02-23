from pathlib import Path

import pandas as pd

from leela_interp import constants as lic
from leela_interp import shell_tools


class LeelaData:
    def __init__(self, root: str | Path = lic.DATA_ROOT):
        self._root = Path(root)
        self._create_data_root()

    def _create_data_root(self) -> None:
        shell_tools.mkdir(self.root, exist_ok=True)
        shell_tools.mkdir(self.puzzles, exist_ok=True)
        shell_tools.mkdir(self.models, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def puzzles(self) -> Path:
        return self.root / "puzzles"

    def puzzles_path(self, puzzle_group: str) -> Path:
        return self.puzzles / f"{puzzle_group}.parquet"

    def load_puzzles(self, puzzle_group: str) -> pd.DataFrame:
        path = self.puzzles_path(puzzle_group)
        return pd.read_parquet(path)

    @property
    def models(self) -> Path:
        return self.root / "models"

    def model_path(self, model_name: str) -> Path:
        return self.models / f"{model_name}.onnx"
