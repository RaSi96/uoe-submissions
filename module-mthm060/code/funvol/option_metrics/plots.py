import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Literal
from matplotlib.figure import Figure

from code.utils import check_max

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def plot_om_surfaces(
        om_surface: pd.DataFrame,
        *,
        option_type: Literal["CE", "PE", "Both"]="Both",
        interpolate_zero: bool=False,
        start: str|pd.Timestamp|None=None,
        end: str|pd.Timestamp|None=None,
        max_surfaces: int=8,
        ncols: int=4,
        figsize: tuple[int, int]|None=None,
    ) -> Figure:
    """
    Plots OptionMetrics-created IV surfaces.

    Parameters:
    `om_surface`: pd.DataFrame:
        Dataset of OM surfaces, expected to have the following columns:
        * `date` (the date of the surface),
        * `cp` (Call/Put identifier: "CE" or "PE"),
        * `delta` (Delta/Moneyness coordinate),
        * `time` (Days to Expiry coordinate), and
        * `iv` (Implied Volatility value)/

    `option_type`: Literal["CE", "PE", "Both"]:
        Determines which option surfaces to plot. "Both" plots calls and puts on
        the same Axes. Default is "Both".

    `interpolate_zero`: bool:
        If True and `option_type` is "Both", linearly interpolates the call- and
        put-side IV surfaces across the 0-Delta grid point across the time axis.
        Results in a single, continuous surface. Default is `False`.

    `start`: str|pd.Timestamp|None:
        The start date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object. Default is None.

    `end`: str|pd.Timestamp|None:
        The end date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object. Default is None.

    `max_surfaces`: int:
        The maximum number of SSVI surfaces to plot. One Axes per surface.
        Default is `8`.

    `ncols`: int:
        The number of Axes columns in the figure. Default is `4`, meaning for a
        selected date range of 8 days (`maxplots=8`, e.g.), a 2x4 image will be
        returned. Default is `4`.

    `figsize`: tuple[int, int]|None:
        Size of the Figure. Default is `None`, which sets a figsize of
        (6*ncols, 6*nrows).

    Returns a matplotlib Figure.
    """
    logger.info(f"{datetime.now()}: Plotting OM surfaces...")

    # Filter by date range
    unique_dates = pd.to_datetime(om_surface["date"]).unique()
    unique_dates = np.sort(unique_dates)

    if not (start and end):
        dates = unique_dates
    elif (start and end):
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

        dates = unique_dates[
            (unique_dates >= start_dt) & (unique_dates <= end_dt)
        ]

        if len(dates) == 0:
            raise ValueError(
                f"{datetime.now()}: No OM surfaces found between "
                f"{start_dt.date()} and {end_dt.date()}."
            )
    else:
        raise ValueError(
            f"{datetime.now()}: Both `start` and `end` must be provided "
            f"together. Received start={start} and end={end}."
        )

    check_max(len(dates), max_surfaces, "surfaces")

    nrows = int(np.ceil(len(dates) / ncols))
    figsize = figsize or (6*ncols, 6*nrows)

    fig, axes = plt.subplots(
        nrows      = nrows,
        ncols      = ncols,
        figsize    = figsize,
        subplot_kw = {"projection": "3d"},
        squeeze    = False,
    )
    axes = axes.ravel()

    cps_to_plot = ["CE", "PE"] if option_type == "Both" else [option_type]
    z_min = om_surface["iv"].min()

    for dt, ax in zip(dates, axes):
        dt_mask = pd.to_datetime(om_surface["date"]) == dt

        if option_type == "Both":
            df_day = om_surface.loc[dt_mask, :]
            if df_day.empty:
                continue

            surface_slice = df_day.pivot_table(
                index   = "delta",
                columns = "time",
                values  = "iv"
            )

            if 0.0 not in surface_slice.index and interpolate_zero:
                surface_slice.loc[0.0] = np.nan
                surface_slice = surface_slice.sort_index()
                surface_slice = surface_slice.interpolate(method="index")

            X, Y = np.meshgrid(
                surface_slice.columns.values,
                surface_slice.index.values
            )
            Z = surface_slice.values

            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
        else:
            # Plot surfaces individually
            for cp in cps_to_plot:
                masks = dt_mask & (om_surface["cp"].eq(cp))
                df_day = om_surface.loc[masks, :]

                if df_day.empty:
                    continue

                surface_slice = df_day.pivot(
                    index   = "delta",
                    columns = "time",
                    values  = "iv"
                )

                X, Y = np.meshgrid(
                    surface_slice.columns.values,
                    surface_slice.index.values
                )
                Z = surface_slice.values

                ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)

        ax.set_xlabel(r"Days To Expiry ($\tau$)")
        ax.set_ylabel(r"Delta ($\Delta$)")
        ax.set_zlabel(r"IV ($\sigma_{\mathrm{OM}}$)")

        ax.set_box_aspect((4, 4, 3), zoom=0.95)
        ax.set_zlim(z_min, 0.50)
        ax.view_init(elev=30, azim=225)
        ax.set_title(f"OM Vol Surface: {pd.Timestamp(dt).date()}")
        ax.invert_yaxis()

    # Hide unused axes
    for ax in axes[len(dates):]:
        ax.set_visible(False)

    fig.tight_layout()

    logger.info(f"{datetime.now()}: Plot generated.")
    return fig
