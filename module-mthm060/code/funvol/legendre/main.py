import logging
import pandas as pd
import numpy as np
import os

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from .legendre import legendre_OLS

from code.utils import get_file_list

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(data_reserve: Path|str, n_o: int=4) -> None:
    files = get_file_list(
        data_reserve,
        glob       = "*_om-surfaces.csv",
        sort_mtime = True
    )

    df = pd.read_csv(files[0], parse_dates=["date"])

    needed_cols = set(("date", "delta", "time", "cp", "iv"))
    diff = needed_cols.difference(df.columns)
    if len(diff) > 0:
        raise ValueError(
            f"{datetime.now()}: The following columns are missing in the "
            f"dataset loaded from `{data_reserve}`: {diff}. Note that `date` "
            "must be a datetime64 column in this case, not a pd.DatetimeIndex."
        )

    coeffs_ce = legendre_OLS(data=df, cp="CE", n_o=n_o)
    cond_ce = np.linalg.cond(coeffs_ce)
    logger.info(
        f"{datetime.now()}: Call-side Legendre projection condition number: "
        f"{cond_ce:.4f}"
    )

    coeffs_pe = legendre_OLS(data=df, cp="PE", n_o=4)
    cond_pe = np.linalg.cond(coeffs_pe)
    logger.info(
        f"{datetime.now()}: Put-side Legendre projection condition number: "
        f"{cond_pe:.4f}"
    )

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basedir = os.path.join(os.path.dirname(__file__), "artefacts")

    filename_ce = f"{basedir}/{runtime}_legendre-coeffs-ce.csv"
    coeffs_ce.reset_index().to_csv(filename_ce, index=False)
    logger.info(
        f"{datetime.now()}: Call-side Legendre projections saved to "
        f"`{filename_ce}`."
    )

    filename_pe = f"{basedir}/{runtime}_legendre-coeffs-pe.csv"
    coeffs_pe.reset_index().to_csv(filename_pe, index=False)
    logger.info(
        f"{datetime.now()}: Put-side Legendre projections saved to "
        f"`{filename_pe}`."
    )


if __name__=="__main__":
    parser = ArgumentParser(
        description = "Project IV surfaces onto Legendre polynomials."
    )

    parser.add_argument(
        "--data_reserve",
        type     = Path,
        help     = "Directory of the IV surface file artefact.",
        required = True
    )
    parser.add_argument(
        "--n_o",
        type    = int,
        help    = (
            "The maximum degree of Legendre polynomial basis functions to use. "
            "Default is 4, as per the reference implementation."
        ),
        default = 4
    )

    args = parser.parse_args()

    main(**vars(args))
