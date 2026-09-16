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

    ssvi_params = {}
    for dt in dates:
        subdf = (
            df
            .loc[:, data_ax]
            .loc[dt]
            .sort_values(["expiry_date", 'K'])
        )

        logger.info(f"{datetime.now()}: Data for date `{dt}`: {subdf.shape}.")

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

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basedir = os.path.join(os.path.dirname(__file__), "artefacts")
    filename = f"{basedir}/{runtime}_ssvi_fitted_params.pkl"

    with open(filename, "wb") as f:
        pickle.dump(ssvi_params, f)

    logger.info(
        f"{datetime.now()}: SSVI fitted parameters saved to `{filename}`"
    )

    # if plot_ssvi_params:
    #     expiries = sorted([str(i) for i in df["expiry_date"].unique().date])
    #     risk_reversals = {}

    #     for dt, data in ssvi_params.items():
    #         # need this to compute time to expiry, and subsequently delta for RR
    #         ddiff = pd.to_datetime(data["expiry_date"]) - pd.Timestamp(dt)

    #         _ = pd.DataFrame({
    #             "ssvi"       : data["ssvi_smile"],
    #             'K'          : data["moneyness"],
    #             "expiry_date": data["expiry_date"],
    #             "iv"         : data["market_iv"],
    #             "tau"        : ddiff.total_seconds()/(60*60*24*365)
    #         })

    #         for exp in expiries:
    #             mask = _["expiry_date"].eq(exp)
    #             subdf = _.loc[mask].set_index('K').drop(columns="expiry_date")

    #             if exp == expiries[-1]:
    #                 rr_svi = compute_risk_rev(
    #                     ln_money = subdf.index,
    #                     iv       = subdf["ssvi"],
    #                     tau      = subdf["tau"],
    #                 )

    #                 risk_reversals[dt] = rr_svi

    #     fig = plot_ssvi_parameters(ssvi_params, risk_reversals, 0.25)
    #     plotname = f"{basedir}/{runtime}_ssvi_params.png"
    #     fig.savefig(plotname)

    # if plot_ssvi_surfaces:
    #     fig = plot_ssvi_interp_surfaces(
    #         ssvi_params,
    #         start = plot_window_start,
    #         end   = plot_window_end
    #     )

    #     plotname = f"{basedir}/{runtime}_ssvi_surfaces.png"
    #     fig.savefig(plotname)

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
