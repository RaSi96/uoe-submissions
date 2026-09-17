import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm

from collections.abc import Mapping
from datetime import datetime
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy.special import eval_legendre
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from typing import Literal

from ssvi.ssvi import ssvi_smile
from ssvi.metrics import compute_risk_rev
from utils import check_max

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def _sorted_expiries(
        df: pd.DataFrame,
        expiry_start: str|pd.Timestamp|None=None,
        expiry_end: str|pd.Timestamp|None=None
    ) -> np.ndarray:
    """
    Sorts the column `expiry_date` in df. Raises if said column doesn't exist.
    If `expiry_start` && `expiry_end` aren't None, then filters down to only
    those expiries within [expiry_start, expiry_end].

    Returns a np.ndarray, which is a sorted list of expiries.
    """
    col = df["expiry_date"]

    if expiry_start and expiry_end:
        logger.info(
            f"{datetime.now()}: Filtering expiries to lie within "
            f"{expiry_start} and {expiry_end}."
        )
        mask = col.between(expiry_start, expiry_end, inclusive="both")
        col  = col[mask]

    return np.sort(col.unique())


def plot_daily_smiles(
        df_otm: pd.DataFrame,
        *,
        if_var: bool=True,
        ncols: int=4,
        max_plots: int=8,
        start: str|pd.Timestamp|None=None,
        end: str|pd.Timestamp|None=None,
        expiry_start: str|pd.Timestamp|None=None,
        expiry_end: str|pd.Timestamp|None=None,
        figsize: tuple[int, int]|None=None,
    ) -> Figure:
    """
    Plots a set of empirical IV smiles per day, for a range of dates.

    Parameters:
    `df_otm`: pd.DataFrame:
        Dataset of OTM options, which must have at least the following columns:
        * expiry_date, which is the expiry date of all options in `df_otm`;
        * iv, which is model-implied volatility of all options in `df_otm`;
        * years_to_expiry, which is how long until expiry, in years, each option
          in `df_otm` has;
        * K, which is log-forward-moneyness; and
        * `df_otm` must have a datetime index.

    `start`: str|pd.Timestamp|None:
        The start date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object. Default is None.

    `end`: str|pd.Timestamp|None:
        The end date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object. Default is None.

    `expiry_start`: str|pd.Timestamp|None:
        The start date of the expiry range. Default is None, meaning all expiry
        IV curves will be used. Note that this filter only works if both
        `expiry_start`

    `expiry_end`: str|pd.Timestamp|None:
        The end date of the expiry range. Default is None, meaning all expiry IV
        curves will be used. Note that this filter only works if both
        `expiry_start` and `expiry_end` are passed.

    `if_var`: bool:
        Boolean flag determining whether to plot total implied variance or model
        -implied volatility. Total implied variance is sigma^2*tau. Default is
        `True`.

    `ncols`: int:
        The number of Axes columns in the figure. Default is `4`, meaning for a
        selected date range of 8 days (`maxplots=8`, e.g.), a 2x4 image will be
        returned. Default is `4`.

    `max_plots`: int:
        The maximum number of Axes to plot. If a date range of 10 days is
        provided but max_plots is less than 10, only those many Axes will be
        plotted. Default is `8`.

    `figsize`: tuple[int, int]|None:
        Size of the Figure. Default is `None`, which sets a figsize of
        (5*ncols, 5*nrows).

    Returns a matplotlib Figure.

    Note: this function does not select specific expiries to plot. All expiries
    sent in under `df_otm` will be plotted. If you want to see IV curves only
    for specific expiries, pre-filter `df_otm` before passing it here.
    """
    dates = df_otm.loc[start:end].index.unique()
    check_max(len(dates), max_plots, "plots")

    expiries = _sorted_expiries(df_otm, expiry_start, expiry_end)
    _msg = "total implied variance" if if_var else "implied volatility"
    logger.info(
        f"{datetime.now()}: Plotting daily {_msg} curves for {len(expiries)} "
        "expiries..."
    )

    colourgrid = np.linspace(0, 1, len(expiries))
    colours = plt.get_cmap("Paired")(colourgrid)

    nrows = int(np.ceil(len(dates) / ncols))
    figsize = figsize or (5*ncols, 5*nrows)

    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = figsize,
        sharex  = True,
        sharey  = True,
        squeeze = False,
    )
    axes = axes.ravel()

    for i, (date, ax) in enumerate(zip(dates, axes)):
        df_day = df_otm.loc[date]
        collect = []

        for colour, expiry in zip(colours, expiries):
            expiry = pd.to_datetime(expiry)
            subdf = df_day.loc[df_day["expiry_date"].eq(expiry)]

            if subdf.empty:
                logger.info(
                    f"{datetime.now()}: Expiry date {expiry} has no data."
                )
                continue

            y = (
                (subdf["iv"]**2)*subdf["years_to_expiry"]
                if if_var else subdf["iv"]
            )

            ax.plot(
                subdf["K"],
                y,
                label = str(expiry.date()),
                alpha = 0.35,
                c     = colour
            )

            if len(y) >= 2:
                collect.append(y.to_numpy())

        # if collect:
        #     twp_avg = twp_multi_average(collect)
        #     lo, hi = ax.get_xlim()
        #     twp_idx = np.linspace(lo, hi, len(twp_avg))
        #     ax.plot(twp_idx, twp_avg, ls="-.", c="tab:gray", label="twp_avg")

        quantity = (
            r"$\sigma_{\mathrm{B76}}^2\,\tau$" if if_var else
            r"$\sigma_{\mathrm{B76}}$"
        )

        ax.set_title(f"{quantity} as of {date.date()}")
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(0.00, 0.012 if if_var else 0.50)
        ax.grid(alpha=0.3)
        ax.set_xlabel("")

        col = i % ncols
        if col == 0:
            ax.set_ylabel(
                "Total Implied Variance" if if_var else "Implied Volatility"
            )
            ax.tick_params(axis="y", labelleft=True, labelright=False)
        elif col == ncols - 1:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelleft=False, labelright=True)
        else:
            ax.tick_params(axis="y", labelleft=False, labelright=False)

        ax.legend()

    # hide unused axes
    for ax in axes[len(dates):]:
        ax.set_visible(False)

    fig.supxlabel("Log-forward-moneyness", fontsize="x-large")
    fig.tight_layout()

    logger.info(f"{datetime.now()}: Plot generated.")
    return fig


def plot_ssvi_curves(
        ssvi_params: dict[str, dict[str, np.ndarray|float]],
        df_otm: pd.DataFrame,
        *,
        max_dates: int=8,
        start: str|pd.Timestamp|None=None,
        end: str|pd.Timestamp|None=None,
        expiry_start: str|pd.Timestamp|None=None,
        expiry_end: str|pd.Timestamp|None=None,
        figsize: tuple[int, int]|None=None,
    ) -> Figure:
    """
    Plots SSVI fits to total implied variance curves. Individual expiries are
    plotted vertically (i.e. along the y-axis of the entire figure), and each
    date SSVI has been fit to is plotted horizontally (i.e., along the x-axis
    of the entire figure).

    Parameters:
    `ssvi_params`: dict[str, dict[str, np.ndarray|float]]:
        Dictionary of fitted SSVI parameters. Expects structure:
        `{"date": {"ssvi_param": value(s)}}`

        Where the inner dictionary's keys are of type:
        {"expiry_date": numpy.ndarray,
         "market_iv"  : numpy.ndarray,
         "rho"        : float,
         "eta"        : float,
         "gamma"      : float,
         "theta"      : numpy.ndarray,
         "ssvi_smile" : numpy.ndarray,
         "moneyness"  : numpy.ndarray,
         "time_to_exp": numpy.ndarray,
         "loss"       : float}

        And all `np.ndarrays` are the same length.

    `df_otm`: pd.DataFrame:
        Dataset of OTM options, which must have at least the following columns:
        * expiry_date, which is the expiry date of all options in `df_otm`;
        * iv, which is model-implied volatility of all options in `df_otm`;
        * years_to_expiry, which is how long until expiry, in years, each option
          in `df_otm` has;
        * K, which is log-forward-moneyness; and
        * `df_otm` must have a datetime index.

    `start`: str|pd.Timestamp|None:
        The start date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object. Default is `None`.

    `end`: str|pd.Timestamp|None:
        The end date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object. Default is `None`.

    `expiry_start`: str|pd.Timestamp|None:
        The start date of the expiry range. Default is None, meaning all expiry
        IV curves will be used. Note that this filter only works if both
        `expiry_start`

    `expiry_end`: str|pd.Timestamp|None:
        The end date of the expiry range. Default is None, meaning all expiry IV
        curves will be used. Note that this filter only works if both
        `expiry_start` and `expiry_end` are passed.

    `max_dates`: int=8:
        The maximum number of dates to plot SSVI fits across. Each date is
        plotted horizontally (i.e. along the x-axis of the entire figure).

    `expiries`: Iterable|None=None:
        An Iterable of expiry dates. All expiries are plotted vertically (i.e.
        along the y-axis of the entire Figure).

    `figsize`: tuple[int, int]|None:
        Size of the Figure. Default is `None`, which sets a figsize of
        (5*ncols, 5*nrows).

    Returns a matplotlib Figure.
    """
    dates = df_otm.loc[start:end].index.unique().sort_values()
    check_max(len(dates), max_dates, "dates")

    expiries = _sorted_expiries(df_otm, expiry_start, expiry_end)

    logger.info(
        f"{datetime.now()}: Plotting for {len(dates)} days and "
        f"{len(expiries)} expiries."
    )

    ncols = len(dates)
    nrows = len(expiries)
    figsize = figsize or (3*ncols, 3*nrows)

    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = figsize,
        sharex  = True,
        sharey  = True,
        squeeze = False,
    )

    iv_min, iv_max = df_otm["iv"].min(), df_otm["iv"].max()

    _params = {k: v for k, v in ssvi_params.items() if k in dates}

    for i, (dt, data) in enumerate(_params.items()):
        df = pd.DataFrame({
            "ssvi"       : data["ssvi_smile"],
            "K"          : data["moneyness"],
            "expiry_date": data["expiry_date"],
            "iv"         : data["market_iv"],
            "tau"        : data["time_to_exp"],
        })

        for j, exp in enumerate(expiries):
            ax = axes[j, i]

            subdf = (
                df
                .loc[df["expiry_date"].eq(exp)]
                .set_index("K")
                .drop(columns="expiry_date")
            )

            if subdf.empty: continue

            ax.plot(
                subdf["iv"],
                label  = "emp",
                marker = "x",
                ls     = "none",
                alpha  = 0.5,
            )
            ax.plot(subdf["ssvi"], label="ssvi", lw=2)

            ax.set_ylim(iv_min, iv_max)
            ax.grid(alpha=0.5)

            if j == 0:
                ax.set_title(str(dt), fontsize="x-large")

            if i == 0:
                ax.set_ylabel(f"Expiry: {exp.item().date()}")

            if j == nrows-1:
                ax.set_xlabel(r"Log-forward-moneyness ($k$)")

            if i == ncols-1:
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", labelleft=False, labelright=True)
            else:
                ax.tick_params(axis="y", labelleft=False, labelright=False)

    for ax in axes.flat:
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)

    fig.supylabel("IV", x=0.995, rotation=90, va="center")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc            = "upper center",
        ncols          = 2,
        frameon        = False,
        bbox_to_anchor = (0.5, 1.01),
        fontsize       = "x-large",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    logger.info(f"{datetime.now()}: Plot generated.")
    return fig


def plot_ssvi_parameters(
        ssvi_params: dict[str, dict[str, np.ndarray|float]],
        rr_Delta: float
    ) -> Figure:
    """
    `ssvi_params`: dict[str, dict[str, np.ndarray|float]]:
        Dictionary of fitted SSVI parameters. Expects structure:
        `{"date": {"ssvi_param": value(s)}}`

        Where the inner dictionary's keys are of type:
        {"expiry_date": numpy.ndarray,
         "market_iv"  : numpy.ndarray,
         "rho"        : float,
         "eta"        : float,
         "gamma"      : float,
         "theta"      : numpy.ndarray,
         "ssvi_smile" : numpy.ndarray,
         "moneyness"  : numpy.ndarray,
         "time_to_exp": numpy.ndarray,
         "loss"       : float}

        And all `np.ndarrays` are the same length.

    `rr_Delta`: float
        The Delta that a RR is calculated at.

    Returns a matplotlib Figure.
    """
    if not (0<rr_Delta<1):
        raise ValueError(
            f"{datetime.now()}: `rr_Delta` must be in [0, 1], got {rr_Delta} "
            "instead."
        )

    logger.info(
        f"{datetime.now()}: Computing {rr_Delta*100:.0f}-Delta Risk "
        "Reversals..."
    )

    risk_reversals = {}
    for (dt, data) in ssvi_params.items():
        _e = data["expiry_date"]
        _k = data["moneyness"]
        _s = data["ssvi_smile"]
        _t = data["time_to_exp"]

        last_expiry = np.unique(_e)[-1]
        mask = (_e==last_expiry)
        risk_reversals[dt] = compute_risk_rev(_k[mask], _s[mask], _t[mask])

    logger.info(f"{datetime.now()}: Plotting fitted SSVI parameters...")

    params = (
        pd
        .DataFrame
        .from_dict(
            {
                dt: {
                    "rho"  : data["rho"],
                    "eta"  : data["eta"],
                    "gamma": data["gamma"],
                }
                for dt, data in ssvi_params.items()
            },
            orient="index",
        )
        .sort_index()
    )

    rr = (
        pd
        .Series(risk_reversals, name=f"{rr_Delta*100:.0f}RR")
        .to_frame()
        .sort_index()
    )

    ssvi_df = params.join(rr, how="left")

    fig, axes = plt.subplots(
        nrows   = len(ssvi_df.columns),
        ncols   = 1,
        figsize = (12, 2.5*len(ssvi_df.columns)),
        sharex  = True,
        squeeze = False,
    )
    axes = axes.ravel()

    titles = {
        "rho"  : r"SSVI parameter $\rho$",
        "eta"  : r"SSVI parameter $\eta$",
        "gamma": r"SSVI parameter $\gamma$",
        "ssvi" : f"{rr_Delta}"+r"$\Delta$ RR with $\sigma_{\mathrm{SSVI}}$",
    }

    for col, ax in zip(ssvi_df.columns, axes):
        ax.plot(ssvi_df[col])
        ax.set_title(titles.get(col, col))
        ax.grid(alpha=0.5)
        ax.tick_params(axis="x", labelrotation=45)
        ax.yaxis.tick_right()

    fig.supxlabel("Date", fontsize="x-large")
    fig.supylabel("Parameter Level", fontsize="x-large")
    fig.tight_layout()

    logger.info(f"{datetime.now()}: Plot generated.")
    return fig


def plot_ssvi_interp_surfaces(
        ssvi_params: dict[str, dict[str, np.ndarray|float]],
        start: str|pd.Timestamp|None=None,
        end: str|pd.Timestamp|None=None,
        *,
        max_surfaces: int=8,
        ncols: int=4,
        figsize: tuple[int, int]|None=None,
    ) -> Figure:
    """
    Interpolates fitted SSVI parameters into a 3D implied volatility surface.
    Plots the SSVI surface.

    Parameters:
    `ssvi_params`: dict[str, dict[str, np.ndarray|float]]:
        Dictionary of fitted SSVI parameters. Expects structure:
        `{"date": {"ssvi_param": value(s)}}`

        Where the inner dictionary's keys are of type:
        {"expiry_date": numpy.ndarray,
         "market_iv"  : numpy.ndarray,
         "rho"        : float,
         "eta"        : float,
         "gamma"      : float,
         "theta"      : numpy.ndarray,
         "ssvi_smile" : numpy.ndarray,
         "moneyness"  : numpy.ndarray,
         "time_to_exp": numpy.ndarray,
         "loss"       : float}

        And all `np.ndarrays` are the same length.

    `start`: str|pd.Timestamp|None:
        The start date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object.

    `end`: str|pd.Timestamp|None:
        The end date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object.

    `max_surfaces`: int:
        The maximum number of SSVI surfaces to plot. One Axes per surface.
        Default is `8`.

    `ncols`: int:
        The number of Axes columns in the figure. Default is `4`, meaning for a
        selected date range of 8 days (`max_surfaces=8`, e.g.), a 2x4 image will
        be returned. Default is `4`.

    `figsize`: tuple[int, int]|None:
        Size of the Figure. Default is `None`, which sets a figsize of
        (6*ncols, 6*nrows).

    Returns a matplotlib Figure.
    """
    logger.info(f"{datetime.now()}: Plotting SSVI surfaces...")

    if not (start and end):                     # if start and end are both None
        dates = sorted(dt for dt in ssvi_params.keys())
    elif (start and end):                   # if start and end are both provided
        start, end = pd.Timestamp(start), pd.Timestamp(end)

        dates = sorted(
            dt for dt in ssvi_params
            if start <= pd.Timestamp(dt) <= end
        )

        if not dates:
            raise ValueError(
                f"{datetime.now()}: No SSVI surfaces found between "
                f"{start.date()} and {end.date()}."
            )
    else:                                # one of start and end weren't provided
        raise ValueError(
            f"{datetime.now()}: Both `start` and `end` must be provided "
            "together. Received start={start} and end={end}."
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

    logger.info(f"{datetime.now()}: Beginning interpolation...")
    for ax, dt in zip(axes, dates):
        data: dict = ssvi_params[dt]
        iv_min = data["market_iv"].min()

        tau = np.asarray(data["time_to_exp"])
        theta = np.asarray(data["theta"])
        money = np.asarray(data["moneyness"])

        k = np.linspace(money.min(), money.max(), 50)
        t = np.linspace(tau.min(), tau.max(), 50)
        K, T = np.meshgrid(k, t)

        order = np.argsort(tau)
        tau_sorted = tau[order]
        theta_sorted = theta[order]

        tau_unique, idx = np.unique(tau_sorted, return_index=True)
        theta_unique = theta_sorted[idx]

        thetas = (
            np
            .interp(T.ravel(), tau_unique, theta_unique)
            .reshape(T.shape)
        )

        w_mesh = ssvi_smile(K, thetas, data["rho"], data["eta"], data["gamma"])
        iv_mesh = np.sqrt(w_mesh / T)

        ax.plot_surface(K, T, iv_mesh, cmap="viridis", alpha=0.8)
        ax.scatter(money, tau, data["ssvi_smile"], c="red", s=5)

        ax.set(
            title  = f"Volatility Surface: {dt}",
            xlabel = r"Log-forward-moneyness ($k$)",
            ylabel = r"Time to expiry ($\tau$)",
            zlabel = r"IV ($\sigma_{\mathrm{SSVI}}$)",
            zlim   = (iv_min, 0.50),
        )
        ax.set_box_aspect((4, 4, 3), zoom=0.90)

    for ax in axes[len(dates):]:
        ax.set_visible(False)

    for ax in axes:
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)

    fig.tight_layout()

    logger.info(f"{datetime.now()}: Plot generated.")
    return fig


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

            if (0.0 not in surface_slice.index) and (interpolate_zero):
                logger.info(
                    f"{datetime.now()}: Interpolating 0-Delta coordinate "
                    "across tau..."
                )

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


def plot_projection_coeffs(
        coeffs: pd.DataFrame,
        daily_ticks: bool=False
    ) -> Figure:
    """
    Plots a time series of projection coefficients, where each time t is a
    vector of coefficients.

    Parameters:
    `coeffs`: pd.DataFrame:
        The dataset of projection coefficients. Expected to be shaped (T, N)
        where N is the number of features.

    `daily_ticks`: bool:
        Boolean controlling whether or not to display daily ticklabels along the
        x- (time) axis. If true, will display a very dense grid at the daily
        level alongside corresponding xticks and xlabels. If false, uses monthly
        frequencies.

    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(25, 7))

    coeffs.plot(ax=ax, grid=True, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    if daily_ticks:
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
        ax.tick_params(axis='x', which='minor', labelsize=6)
    else:
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))

    ax.set_xlim(coeffs.index.min(), coeffs.index.max())
    ax.legend(loc="upper left")
    ax.grid(which="minor")
    ax.set_title("Basis coefficients over time")
    ax.set_ylabel("Level")
    ax.set_xlabel("Time")
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")

    fig.tight_layout()
    return fig


def plot_vifs(
        dataframes: Mapping[str, pd.DataFrame],
        *,
        ncols: int = 2,
        figsize_per_axes: tuple[float, float] = (5.5, 5),
    ) -> Figure:
    """
    Plot the Variance Inflation Factors of all features in a given set of data-
    frames.

    Parameters:
    `dataframes`: Mapping[str, pd.DataFrame]:
        Mapping between subplot titles and dataframes.

    `ncols`: int:
        Maximum number of subplot columns to draw.

    `figsize_per_axes`: tuple[float, float]:
        Width and height allocated to each subplot. Defaults to (5.5in, 5in).

    Returns a matplotlib Figure, containing one subplot per dataframe.
    """
    if not dataframes:
        raise ValueError("dataframes must contain at least one dataframe.")

    if ncols < 1:
        raise ValueError("ncols must be at least 1.")

    nplots = len(dataframes)
    ncols = min(ncols, nplots)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = (figsize_per_axes[0] * ncols, figsize_per_axes[1] * nrows),
        sharey  = True,
        squeeze = False,
    )

    axes = axes.ravel()

    for ax, (title, df) in zip(axes, dataframes.items()):
        vifs = {
            column: vif(df, index)
            for index, column in enumerate(df.columns)
        }

        pd.Series(vifs).plot.bar(ax=ax, grid=True)

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    # Hide unused axes when the number of dataframes does not fill the grid.
    for ax in axes[nplots:]:
        ax.set_visible(False)

    # Show the shared y-axis labels only on the last visible subplot.
    axes[nplots - 1].tick_params(
        axis       = "y",
        right      = True,
        labelright = True,
    )

    fig.supxlabel("Basis functions", fontsize="x-large")
    fig.supylabel(r"VIF = $\frac{1}{1-R^2}$", fontsize="x-large")
    fig.suptitle("Variance Inflation Factors", fontsize="x-large")

    fig.tight_layout()
    return fig


def plot_eigensurfaces(
        princomps: np.ndarray,
        eigenvariances: np.ndarray,
        n_o: int=4
    ) -> Figure:
    degrees = np.arange(0, n_o+1, 1)
    norm = 1/np.sqrt( 2/(2*degrees +1) )

    outer_sum = np.add.outer(degrees, degrees)
    i, j = np.where( (0<outer_sum) & (outer_sum<= 4) )

    xg = np.linspace(start=-1, stop=1, num=101)
    yg = np.linspace(start=-1, stop=1, num=101)
    X, Y = np.meshgrid(xg, yg)

    co_delta = eval_legendre(degrees, X.ravel()[:, None]) * norm
    co_tau   = eval_legendre(degrees, Y.ravel()[:, None]) * norm
    Lgrid = (co_delta[:,:,None] * co_tau[:,None,:])[:, i, j]
    Lgrid = sm.add_constant(Lgrid)

    Zs = [
        (Lgrid @ princomps[k]).reshape(X.shape)
        for k in range(len(princomps))
    ]

    # Shared symmetric Z-axis across all eigensurfaces.
    zmax = max(np.max(np.abs(Z)) for Z in Zs)

    norm_Z = Normalize(-zmax, zmax)
    cmap = plt.colormaps["viridis"]

    fig, axes = plt.subplots(
        nrows=2, ncols=6, figsize=(14, 6.5), subplot_kw={"projection": "3d"},
    )

    for ax, Z, k in zip(np.ravel(axes), Zs, range(len(Zs))):
        ax.plot_surface(
            X, Y, -Z,
            facecolors  = cmap(norm_Z(Z)),
            linewidth   = 0,
            antialiased = True,
            shade       = False,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        ax.set_box_aspect((1, 1, 0.65))
        ax.set_title(rf"$\psi_{{{k+1}}}={eigenvariances[k] * 100:.2f}\%$", pad=2,)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Keep the common coordinate system visible without labels.
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

    # Hide unused panels.
    for ax in np.ravel(axes)[len(Zs):]:
        ax.set_axis_off()

    fig.tight_layout(pad=0.5)
    return fig


