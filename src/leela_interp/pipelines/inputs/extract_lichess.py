import click
import pandas as pd
import zstandard as zstd

from leela_interp import shell_tools
from leela_interp.data import LeelaData
from leela_interp.pipelines import cli_options as clio

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def extract_lichess_main(data_root: str) -> None:
    li_data = LeelaData(data_root)
    dl_path = li_data.root / "lichess_db_puzzle.csv.zst"
    csv_path = li_data.root / "lichess_db_puzzle.csv"
    out_path = li_data.puzzles_path("lichess")

    print(f"Downloading lichess puzzles to {dl_path}")
    shell_tools.wget(URL, dl_path)

    print(f"Decompressing lichess puzzles to {csv_path}")
    with dl_path.open("rb") as ifh, csv_path.open("wb") as ofh:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(ifh, ofh)

    print(f"Converting lichess puzzles to parquet at {out_path}")
    puzzle_df = pd.read_csv(csv_path)
    shell_tools.touch(out_path, clobber=True)
    puzzle_df.to_parquet(out_path)

    dl_path.unlink()
    csv_path.unlink()


@click.command()
@clio.with_data_root()
def extract_lichess(data_root: str) -> None:
    extract_lichess_main(data_root)
