# name inspired by `hl.exe -map c2a4`

import logging
import numpy as np
import os
import pandas as pd

from datetime import datetime
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def prepare_underlying(
        underlying_path: Path|str,
        div_yield_path: Path|str|None
    ) -> pd.DataFrame:
    """
    Prepares the underlying dataset by loading its data, from `underlying_path`,
    and its dividend yields, from `div_yield_path`, if provided. If no path for
    dividend yields is provided, all dividend yields are assumed to numerically
    be zero.

    Returns a pd.DataFrame with columns ["close", "div_yield"], and a datetime
    index.
    """
    nifty = pd.read_csv(
        underlying_path,
        names       = ["date", "open", "high", "low", "close"],
        usecols     = ["date", "close"],
        header      = 0,
        parse_dates = [0],
        index_col   = [0]
    )

    valid_range = pd.date_range(nifty.index.min(), nifty.index.max(), freq='D')
    nifty = nifty.reindex(index=valid_range, method="ffill")

    # regardless of branch, we must return dividend yield data because it's used
    # downstream in Black-76. If no, or fallacious, data is provided, then a
    # reasonable default is 0 because \forall options then, q=0. This shouldn't
    # interfere with B76 asymptotics.
    if div_yield_path is not None:
        dy = pd.read_csv(
            div_yield_path,
            names       = ["date", "div_yield", "nifty"],
            header      = 0,
            usecols     = ["date", "div_yield"],
            parse_dates = [0],
            index_col   = [0]
        )

        dy = dy.reindex(index=valid_range, method="ffill").dropna()

        if (dy.empty) or (dy.size != nifty.size):
            logger.warning(
                f"{datetime.now()}: Dividend yield dataset is empty or "
                "contains NaNs after attempting to reindex with underlying "
                "data. Setting to zero."
            )
            dy = pd.DataFrame(
                data    = np.zeros_like(nifty),
                index   = nifty.index,
                columns = ["div_yield"]
            )
        else:
            dy /= 100
    else:
        logger.warning(
            f"{datetime.now()}: No dividend yield data provided, setting to "
            " zero."
        )
        dy = pd.DataFrame(
            data    = np.zeros_like(nifty),
            index   = nifty.index,
            columns = ["div_yield"]
        )

    nifty = nifty.merge(dy, left_index=True, right_index=True, how="inner")
    logger.info(
        f"{datetime.now()}: Merged Nifty & dividend yields {nifty.shape}"
    )

    return nifty


def get_file_list(basedir: Path|str) -> list:
    logger.info(f"{datetime.now()}: Scanning `{basedir}`...")

    files = os.listdir(basedir)
    logger.info(f"{datetime.now()}: Found {len(files)} files.")

    return files


def load_bhav(filepath: Path|str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        names     = [
            "instrument",
            "symbol",
            "expiry_date",
            "strike",
            "cp_flag",
            "open",
            "high",
            "low",
            "close",
            "settlement_price",
            "n_contracts",
            "value",
            "oi",
            "oi_chg",
            "date"
        ],
        header    = 0,
        index_col = False
    )

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["instrument"].eq("OPTIDX")
        & df["symbol"].eq("NIFTY")
        & df["oi"].ne(0)
    )

    drop_cols = ["instrument", "symbol",]

    df = (
        df
        .assign(
            date        = pd.to_datetime(df["date"], format="%d-%b-%Y"),
            expiry_date = pd.to_datetime(df["expiry_date"], format="%d-%b-%Y")
        )
        .loc[mask, :]
        .drop(columns=drop_cols)
        .set_index("date")
    )

    logger.info(
        f"{datetime.now()}: Modified dates, masked data, dropped columns."
    )

    return df


def merge_spot(
        raw_bhav: pd.DataFrame,
        spot: pd.DataFrame,
    ) -> pd.DataFrame:
    df = (
        raw_bhav
        .merge(spot, left_index=True, right_index=True, how="inner")
        .rename(columns={"close_y": "nifty", "close_x": "close"})
        .loc[:, raw_bhav.columns.tolist() + ["nifty", "div_yield"]
        ]
    )

    logger.info(f"{datetime.now()}: Merged spot & dividend yields")
    return df