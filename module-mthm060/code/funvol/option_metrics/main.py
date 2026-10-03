import logging
import os

from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from pathlib import Path

from code.utils import *
from .greeks import *
from .om_surfacing import ometrics_surface

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(data_reserve: Path|str) -> None:
    files = get_file_list(data_reserve, glob="*-allbhav-iv.csv")
    df = pd.concat([load_processed_bhav(f) for f in files])

    df = (
        df
        .reset_index()
        .sort_values(["date", "expiry_date", "strike"])
        .set_index("date")
    )

    logger.info(f"{datetime.now()}: Loaded processed Bhavcopies.")

    delta = get_option_delta(
        spot   = df["nifty"],
        strike = df["strike"],
        r      = 0.10,
        q      = df["div_yield"],
        iv     = df["iv"],
        t_diff = df["years_to_expiry"],
    )

    vega = get_option_vega(
        spot   = df["nifty"],
        strike = df["strike"],
        r      = 0.10,
        q      = df["div_yield"],
        iv     = df["iv"],
        t_diff = df["years_to_expiry"],
    )

    df = (
        df
        .assign(delta=delta, vega=vega)
        .reset_index()
        .sort_values(["date", "expiry_date", "strike"])
        .set_index("date")
    )

    df.loc[df["cp_flag"].eq("PE"), "delta"] -= 1  # OM's call-equivalent Delta

    mask = (
        (df["delta"].between(-0.90, 0.90, inclusive="both"))
        & (df["years_to_expiry"].mul(365).between(10, 730, inclusive="both"))
        & (df["vega"].ge(0.5))
    )

    filtered = df.loc[mask, :]
    logger.info(
        f"{datetime.now()}: Filtered to Deltas in [-0.9, 0.9], YTEs in "
        f"[10, 730] and Vega >= 0.5: {len(filtered)} records."
    )

    # OptionMetrics, as per [1] ------------------------------------------------
    logger.info(f"{datetime.now()}: Prepping OM grid...")

    delta_start = 0.10
    delta_end = 0.90
    delta_step = 0.05
    grid_delta_ce = np.arange(delta_start, delta_end+delta_step, delta_step)
    grid_delta_pe = -grid_delta_ce

    grid_days = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730])

    grid_ce = list(product(grid_delta_ce, grid_days, ["CE"]))
    grid_pe = list(product(grid_delta_pe, grid_days, ["PE"]))
    surface_grid = grid_ce + grid_pe

    om_surface = ometrics_surface(filtered, surface_grid)

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basedir = os.path.join(os.path.dirname(__file__), "artefacts")
    filename = f"{basedir}/{runtime}_om-surfaces.csv"
    om_surface.to_csv(filename, index=False)
    logger.info(f"{datetime.now()}: OM surface data saved to `{filename}`.")
    return


if __name__=="__main__":
    parser = ArgumentParser(
        description = "Generate IV surfaces using OptionMetrics KDE."
    )

    parser.add_argument(
        "--data_reserve",
        type     = Path,
        help     = "Directory of Bhavcopies with IV computed.",
        required = True
    )

    args = parser.parse_args()

    main(**vars(args))


# ------------------------------------------------------------------------------
# References:
# [1] https://wrds-www.wharton.upenn.edu/documents/2231/IvyDB_US_v7.0_Reference_Manual.pdf