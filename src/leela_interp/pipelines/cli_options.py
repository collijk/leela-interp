from collections.abc import Callable
from pathlib import Path

import click

from leela_interp import constants as lic


def with_data_root[**P, T](
    default: str | Path = lic.DATA_ROOT,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--data-root",
        "-o",
        type=click.Path(file_okay=False, dir_okay=True),
        default=default,
        show_default=True,
        help="Root directory where outputs will be saved.",
    )
