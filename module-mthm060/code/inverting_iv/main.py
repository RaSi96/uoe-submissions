import logging
import pandas as pd

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Iterable

from code.utils import *
from .bhav_processing import *
from .black_scholes import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(
        data_reserve: Path|str,
        years: Iterable[int],
        months: Iterable[int],
        underlying: pd.DataFrame
    ) -> None:
    """
    Main subroutine for obtaining model-implied volatility.

    Parameters:
    `data_reserve`: Path | str:
        The directory of raw Bhavcopies, structured as:
        `./{data_reserve}/{year}/{month}/FILENAME.csv`

        Where `year` and `month` are integers, and `month` ranges from 1..12. So
        for example, one Bhavcopy might be located at, as in [1]:
        `./nifty_bhavs/2010/1/fo04JAN2010bhav.csv`

        This subroutine recursively scans this directory and processes each file
        individually, for all files it detect.

    `years`: Iterable[int]:
        An iterable of integer years, used to discover all relevant Bhavcopies
        in `data_reserve`. For example:
        `[2010, 2011, 2012, ...]`

    `months`: Iterable[int]:
        An iterable of integer months, used to discover all relevant Bhavcopies
        in `data_reserve`. For example:
        `[1, 2, 3, ...]`

    Returns nothing. Saves processed Bhavcopies as CSVs in:
    `./{data_reserve}/{year}/{month}/{year}-{mo}-allbhav.csv"`
    """
    logger.info(f"{datetime.now()}: Loop starting...")
    for year in years:
        for mo in months:
            basedir = Path(data_reserve) / str(year) / str(mo)

            try:
                files = get_file_list(basedir, glob="*bhav.csv")
            except Exception as e:
                logger.exception(
                    f"{datetime.now()}: Error findings files from {basedir}: "
                    f"{e}. Skipping..."
                )
                continue

            if len(files) == 0:
                continue

            try:
                raw_bhav = pd.concat(
                    [load_bhav(f) for f in files]
                )
                logger.info(f"{datetime.now()}: Parsed files.")
            except Exception as e:
                logger.exception(
                    f"{datetime.now()}: Error parsing files from {basedir}: "
                    f"{e}. Skipping..."
                )
                continue

            try:
                raw_bhav = filter_data(raw_bhav)
            except Exception as e:
                logger.exception(
                    f"{datetime.now()}: Error filtering data: {e}. Skipping..."
                )
                continue

            try:
                raw_bhav = merge_spot(raw_bhav, underlying)
            except Exception as e:
                logger.exception(
                    f"{datetime.now()}: Error merging spot: {e}. Skipping..."
                )
                continue

            t_diff = (raw_bhav["expiry_date"].sub(raw_bhav.index))

            raw_bhav = raw_bhav.assign(
                years_to_expiry = t_diff.dt.total_seconds().div(60*60*24*365)
            )

            logger.info(f"{datetime.now()}: Computed tau.")

            try:
                raw_bhav = compute_iv(raw_bhav)
                logger.info(f"{datetime.now()}: Successfully computed IV.")

            except Exception as e:
                logger.exception(
                    f"{datetime.now()}: Error computing IV: {e}, skipping."
                )
                continue

            # edge case for May 2012, noted on 30Apr26 [1]. no idea where this
            # comes from.
            try:
                raw_bhav = filter_otm(raw_bhav)
            except Exception as e:
                logger.exception(
                    f"{datetime.now()}: Error filtering data: {e}, trying to "
                    "data cast..."
                )

                try:
                    raw_bhav = filter_otm(raw_bhav.astype({"strike": "float"}))
                except Exception as e:
                    logger.exception(
                        f"{datetime.now()}: Casting failed, skipping."
                    )
                    continue

            drop_cols = [
                "open",
                "high",
                "low",
                "settlement_price",
                "n_contracts",
                "value",
                "oi_chg",
            ]

            filename = f"{basedir}/{year}-{mo}-allbhav-iv.csv"

            (
                raw_bhav
                .reset_index()
                .drop(columns=drop_cols)
                .to_csv(filename, index=False)
            )

            logger.info(f"{datetime.now()}: Saved file to `{filename}`.\n")
    logger.info(f"{datetime.now()}: Done!")
    return


if __name__=="__main__":
    parser = ArgumentParser(
        description = "Compute B76 model-implied volatility from Bhavcopies."
    )

    parser.add_argument(
        "--underlying",
        type    = Path,
        help    = "Path to the dataset of underlying prices.",
        required = True
    )

    parser.add_argument(
        "--bhavcopies",
        type    = Path,
        help    = "Directory of raw Bhavcopies.",
        required = True
    )

    parser.add_argument(
        "--div_yields",
        type    = Path,
        help    = "Path to the dataset of underlying dividend yields.",
        default = None
    )

    parser.add_argument(
        "--date_from",
        type    = str,
        help    = "Date to begin searching for Bhavcopies from.",
        default = "2010-01-01",
    )

    parser.add_argument(
        "--date_to",
        type    = str,
        help    = "Date to stop searching for Bhavcopies.",
        default = "2019-10-04",
    )

    args = parser.parse_args()

    # these are the dates we (should) have Bhavcopies for
    # dates  = pd.date_range(start="2010-01-01", end="2019-10-04", freq='B')
    dates = pd.date_range(start=args.date_from, end=args.date_to, freq='B')

    nifty  = prepare_underlying(
        underlying_path = args.underlying,
        div_yield_path  = args.div_yields
    )

    main(
        data_reserve = args.bhavcopies,
        years        = set(dates.year),
        months       = set(dates.month),
        underlying   = nifty
    )

# ------------------------------------------------------------------------------
# References:
# [1] https://www.kaggle.com/datasets/rasi96/nse-f-and-o-bhavcopies-2010-2019