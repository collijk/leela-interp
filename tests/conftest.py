from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def puzzle_data() -> pd.DataFrame:
    puzzle_data_path = Path(__file__).parent / "data" / "test_puzzles.parquet"
    puzzles = pd.read_parquet(puzzle_data_path)
    return puzzles


@pytest.fixture(params=list(range(25)))
def puzzle(request: pytest.FixtureRequest, puzzle_data: pd.DataFrame) -> pd.Series[Any]:
    puzzle_idx = request.param
    return puzzle_data.iloc[puzzle_idx]
