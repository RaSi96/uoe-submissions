import logging
import pandas as pd

from datetime import datetime
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def check_max(n: int, max_n: int, what: str) -> None:
    if n > max_n:
        raise ValueError(
            f"{n} {what} requested, but max_{what.replace(' ', '_')}={max_n}. "
            f"Reduce the date range or increase the limit."
        )


def get_file_list(basedir: Path|str, glob: str="*.csv") -> list[Path]:
    """
    Returns a list of all files in `basedir` that end with `regex`.

    Parameters:
    `basedir`: Path|str:
        The directory to scan for files.

    `regex`: str:
        A regular expression search string, if necessary. Defaults to "*.csv",
        which will return a list of all CSVs in `basedir`.

    Returns a list of Path-qualified filepaths.
    """
    _basedir = Path(basedir)
    files = list(_basedir.glob(glob))
    logger.info(f"{datetime.now()}: Found {len(files)} files.")
    return files


def load_processed_bhav(filepath: Path|str) -> pd.DataFrame:
    """
    Loads a single processed Bhavcopy, where "processed" means:
    * Only Nifty options exist,
    * Model-implied volatility has been obtained for each Nifty option, and
    * Only OTM Nifty options and their corresponding IVs have been retained.

    Expects at least the following columns in the CSV located at `filepath`:
    `[date, expiry_date, strike, cp_flag, close, oi, nifty, div_yield,
      years_to_expiry, iv, K]`

    Returns a pd.DataFrame.
    """
    df = pd.read_csv(
        filepath,
        names     = [
            "date",
            "expiry_date",
            "strike",
            "cp_flag",
            "close",
            "oi",
            "nifty",
            "div_yield",
            "years_to_expiry",
            "iv",
            "K"
        ],
        header    = 0,
        parse_dates=[0, 1],
        index_col = [0],
    )

    return df