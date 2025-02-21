from leela_interp.inputs.extract_figshare import (
    extract_figshare,
)
from leela_interp.inputs.extract_lichess import (
    extract_lichess,
)

RUNNERS = {
    "extract_figshare": extract_figshare,
    "extract_lichess": extract_lichess,
}
