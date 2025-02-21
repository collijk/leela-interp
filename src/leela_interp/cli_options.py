from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import click

from leela_interp import constants as lic

_T = TypeVar("_T")
_P = ParamSpec("_P")
_EntryPoint = Callable[_P, _T]
ClickOption = Callable[[_EntryPoint[_P, _T]], _EntryPoint[_P, _T]]


def with_data_root(default: str | Path = lic.DATA_ROOT) -> ClickOption[_P, _T]:
    return click.option(
        "--data-root",
        "-o",
        type=click.Path(file_okay=False, dir_okay=True),
        default=default,
        show_default=True,
        help="Root directory where outputs will be saved.",
    )
