import logging

from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from pathlib import Path

from code.utils import *
from greeks import *
from om_surfacing import ometrics_surface
from plots import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(
        data_reserve: Path|str,
        plot_surface: bool=False,
        plot_option_type: Literal["CE", "PE", "Both"]="Both",
        interpolate_plot: bool=False,
        plot_window_start: pd.Timestamp|None=None,
        plot_window_end: pd.Timestamp|None=None,
    ) -> None:
    files = get_file_list(data_reserve, glob="*-allbhav-iv.csv")
    df = pd.concat(
        [load_processed_bhav(f) for f in files]
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

    if plot_surface:
        plot_om_surfaces(
            om_surface       = om_surface,
            option_type      = plot_option_type,
            interpolate_zero = interpolate_plot,
            start            = plot_window_start,
            end              = plot_window_end
        )

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./{data_reserve}/{runtime}_om-surfaces.csv"
    om_surface.to_csv(filename, index=False)

    logger.info(f"{datetime.now()}: OM surface data saved to `{filename}`")
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
        "--plot_surface",
        help     = "Whether or not to plot OptionMetrics-generated IV surfaces.",
        default  = False,
        action   = "store_true"
    )
    parser.add_argument(
        "--plot_option_type",
        type     = str,
        help     = (
            "Which wing of surfaces must be plotted. May be 'CE' for calls, "
            "'PE' for puts, or 'Both' for both. Only used if `--plot_surface` "
            "is passed."
        )
    )
    parser.add_argument(
        "--interpolate_plot",
        help     = (
            "Whether or not to interpolate the OptionMetrics surface plot "
            "across the 0-Delta grid point. If `--plot_option_type=Both`, the "
            "default plots call and put wings separately with a gap at Delta=0. "
            "Delta=0. If this is true, the gap is linearly interpolated across "
            "the tau axis (time to expiry). Only used if "
            "`--plot_option_type=Both`."
        ),
        default  = False,
        action   = "store_true"
    )
    parser.add_argument(
        "--plot_window_start",
        type     = pd.Timestamp,
        help     = (
            "Date/datetime string of the start of the plotting window. Required "
            "if any of the `--plot_*` arguments are provided as `True`, ignored "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--plot_window_end",
        type     = pd.Timestamp,
        help     = (
            "Date/datetime string of the end of the plotting window. Required "
            "if any of the `--plot_*` arguments are provided as `True`, ignored "
            "otherwise."
        ),
    )

    args = parser.parse_args()

    main(**vars(args))


# ------------------------------------------------------------------------------
# References:
# [1] https://wrds-www.wharton.upenn.edu/documents/2231/IvyDB_US_v7.0_Reference_Manual.pdf