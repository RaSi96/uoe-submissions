import logging
import os

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from .detrended_price import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(data_reserve: Path|str, freq: str='B') -> None:
    df = (
        load_underlying(data_reserve, freq=freq)
        .pipe(transform_price)
        .pipe(detrend)
    )

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basedir = os.path.join(os.path.dirname(__file__), "artefacts")
    filename = f"{basedir}/{runtime}_detrended_underlying_price.csv"
    df.reset_index().to_csv(filename, index=False)
    logger.info(
        f"{datetime.now()}: Detrended underlying price data saved to "
        f"`{filename}`."
    )


if __name__=="__main__":
    parser = ArgumentParser(description="Perform FPCA on (a) given dataset(s)")

    parser.add_argument(
        "--data_reserve",
        type     = Path,
        help     = "Directory of the underlying price data..",
        required = True
    )
    parser.add_argument(
        "--freq",
        type     = str,
        help     = (
            "Discrete-time frequency of the underlying price data. Defaults "
            "to 'B', which assumes a Business Day (daily) frequency."
        ),
        default  = 'B'  # original was 'D'
    )

    args = parser.parse_args()

    main(**vars(args))
