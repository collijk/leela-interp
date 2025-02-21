import click

from leela_interp import constants as lic
from leela_interp.data import LeelaData
from leela_interp.shell_tools import wget, touch
from leela_interp import cli_options as clio


FIGSHARE_URL_TEMPLATE = "https://figshare.com/ndownloader/files/{file_id}?private_link=adc80845c00b67c8fce5"

def figshare_main(data_root: str):
    li_data = LeelaData(data_root)
    
    file_map = {
        46473526: ("puzzles", "interesting"),
        46473550: ("puzzles", "unfiltered"),
        46473529: ("models", "lc0"),
        46473535: ("models", "lc0-random"),
        46473541: ("models", "LD2"),
        52056785: ("models", "lc0-original"),
    }

    for file_id, (subdir, name) in file_map.items():
        url = FIGSHARE_URL_TEMPLATE.format(file_id=file_id)
        if subdir == "puzzles":
            path = li_data.puzzles_path(name)
        elif subdir == "models":
            path = li_data.model_path(name)
        else:
            msg = f"Unknown subdir: {subdir}"
            raise ValueError(msg)

        touch(path, clobber=True)
        print(f"Downloading {name} to {path}")
        wget(url, path)


@click.command()
@clio.with_data_root()
def figshare(data_root: str):
    figshare_main(data_root)
