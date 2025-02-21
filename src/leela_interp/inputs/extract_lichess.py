
import click
import zstandard as zstd

from leela_interp import cli_options as clio
from leela_interp.data import LeelaData
from leela_interp.shell_tools import touch, wget

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

def extract_lichess_main(data_root: str):
    li_data = LeelaData(data_root)
    temp_path = li_data.root / "lichess_db_puzzle.csv.zst"
    touch(temp_path, clobber=True)
    print(f"Downloading lichess puzzles to {temp_path}")
    wget(URL, temp_path)

    path = li_data.puzzles_path("lichess_full")
    touch(path, clobber=True)
    print(f"Decompressing lichess puzzles to {path}")

    with temp_path.open("rb") as ifh, path.open("wb") as ofh:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(ifh, ofh)

    temp_path.unlink()


@click.command()
@clio.with_data_root()
def extract_lichess(data_root: str):
    extract_lichess_main(data_root)
