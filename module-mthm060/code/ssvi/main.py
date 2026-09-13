import logging
import pandas as pd

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from dataload import *
from ssvi import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(basedir: Path|str):
    files = find_processed_bhavs(basedir)
    df = pd.concat(
        [load_processed_bhav(f) for f in files]
    )
    logger.info(f"{datetime.now()}: Loaded processed Bhavcopies.")

    dates = df.index.unique()
    data_ax = ["strike", "iv", "K", "expiry_date", "years_to_expiry"]

    ssvi_params = {}
    for dt in dates:
        subdf = (
            df
            .loc[:, data_ax]
            .loc[dt]
            .sort_values(["expiry_date", 'K'])
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

        opt_ssvi = fit_ssvi_smiles(moneyness, theta, market_total_var, ytd)

        extra_params = {
            "expiry_date": subdf["expiry_date"].to_numpy(),
            "market_iv": subdf["iv"].to_numpy(),
        }

        ssvi_params[str(dt.date())] = extra_params | opt_ssvi


if __name__=="__main__":
    parser = ArgumentParser(description = "Fit SSVI surfaces to Bhavcopies.")

    parser.add_argument(
        "--data_reserve",
        type     = Path,
        help     = "Directory of Bhavcopies with IV computed.",
        required = True
    )

    args = parser.parse_args()

    main(args.data_reserve)