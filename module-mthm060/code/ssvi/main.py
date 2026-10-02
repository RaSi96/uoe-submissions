import logging
import os
import pandas as pd
import pickle

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from code.utils import *
from .metrics import *
from .ssvi import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(
        data_reserve: Path|str,
        expiries_start: pd.Timestamp|None=None,
        expiries_end: pd.Timestamp|None=None,
    ) -> None:
    files = get_file_list(data_reserve, glob="*-allbhav-iv.csv")
    df = pd.concat([load_processed_bhav(f) for f in files])

    df = (
        df
        .reset_index()
        .sort_values(["date", "expiry_date", "strike"])
        .set_index("date")
    )

    logger.info(f"{datetime.now()}: Loaded processed Bhavcopies: {df.shape}.")

    if expiries_start and expiries_end:
        _mask = df["expiry_date"].between(
            expiries_start,
            expiries_end,
            inclusive = "both"
        )

        df = df.loc[_mask]
        logger.info(
            f"{datetime.now()}: Filtered to expiries only between "
            f"`{expiries_start}` and `{expiries_end}`."
        )

        _mask = None

    dates = df.index.unique().sort_values()
    data_ax = ["strike", "iv", "K", "expiry_date", "years_to_expiry"]

    logger.info(
        f"{datetime.now()}: Starting SSVI calibration loop over {len(dates)} "
        "dates..."
    )

    ssvi_params = {}
    for dt in dates:
        subdf = (
            df
            .loc[:, data_ax]
            .loc[dt]
            .sort_values(["expiry_date", 'K'])
        )

        logger.info(
            f"{datetime.now()}: Data for date `{dt}` shaped {subdf.shape}."
        )

        # for each expiry on that specific date, we need atm variance
        atm_dict = {
            exp: compute_atm_var(group)
            for exp, group in subdf.groupby("expiry_date")
        }

        subdf = subdf.assign(
            theta = subdf["expiry_date"].map(atm_dict),
            market_total_var = (subdf["iv"]**2)*subdf["years_to_expiry"]
        )

        # any day that has 0 ATM variance is a day when all options have already
        # expired. So we can filter that day out WLOG.
        subdf = subdf.loc[~subdf["theta"].eq(0)]

        # remember a benefit of SSVI over previous SVI parameterisations is that
        # SSVI can be fit globally, only locally having to compute ATM total
        # implied var.
        moneyness        = subdf['K'].to_numpy()
        theta            = subdf["theta"].to_numpy()
        market_total_var = subdf["market_total_var"].to_numpy()
        ytd              = subdf["years_to_expiry"].to_numpy()

        try:
            opt_ssvi = fit_ssvi_smiles(moneyness, theta, market_total_var, ytd)
        except Exception as e:
            logger.exception(
                f"{datetime.now()}: Data for {dt} has NaNs, or otherwise "
                "caused an issue. Skipping."
            )
            continue

        logger.info(f"{datetime.now()}: Calibrated SSVI for {dt}.")

        extra_params = {
            "expiry_date": subdf["expiry_date"].to_numpy(),
            "market_iv": subdf["iv"].to_numpy(),
        }

        ssvi_params[str(dt.date())] = extra_params | opt_ssvi

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basedir = os.path.join(os.path.dirname(__file__), "artefacts")
    filename = f"{basedir}/{runtime}_ssvi_fitted_params.pkl"

    with open(filename, "wb") as f:
        pickle.dump(ssvi_params, f)

    logger.info(
        f"{datetime.now()}: SSVI fitted parameters saved to `{filename}`"
    )

    return


if __name__=="__main__":
    parser = ArgumentParser(description = "Fit SSVI surfaces to Bhavcopies.")

    parser.add_argument(
        "--data_reserve",
        type     = Path,
        help     = "Directory of Bhavcopies with IV computed.",
        required = True
    )
    parser.add_argument(
        "--expiries_start",
        type     = pd.Timestamp,
        help     = (
            "Date/datetime string of the start of the expiry window."
        ),
    )
    parser.add_argument(
        "--expiries_end",
        type     = pd.Timestamp,
        help     = (
            "Date/datetime string of the end of the expiry window."
        ),
    )

    args = parser.parse_args()

    main(**vars(args))
