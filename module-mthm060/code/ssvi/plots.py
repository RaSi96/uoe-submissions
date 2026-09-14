import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from matplotlib.figure import Figure
from typing import Literal, Iterable

from ssvi import ssvi_smile

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def _check_max(
        n: int,
        max_n: int,
        what: Literal["plots", "dates", "surfaces"]="plots"
    ) -> None:
    if n > max_n:
        raise ValueError(
            f"{n} {what} requested, but max_{what.replace(' ', '_')}={max_n}. "
            f"Reduce the date range or increase the limit."
        )


def _sorted_expiries(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["expiry_date"].unique())


def _ssvi_frame(data: dict, dt: str) -> pd.DataFrame:
    tau = (
        pd.to_datetime(data["expiry_date"]) - pd.Timestamp(dt)
    ).total_seconds() / (24 * 60 * 60 * 365)

    return pd.DataFrame({
        "ssvi"       : data["ssvi_smile"],
        "K"          : data["moneyness"],
        "expiry_date": data["expiry_date"],
        "iv"         : data["market_iv"],
        "tau"        : tau,
    })


def plot_daily_smiles(
        df_otm: pd.DataFrame,
        *,
        if_var: bool=True,
        ncols: int=4,
        max_plots: int=8,
        start: str|pd.Timestamp|None=None,
        end: str|pd.Timestamp|None=None,
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
    """
    dates = df_otm.loc[start:end].index.unique()
    _check_max(len(dates), max_plots)

    expiries = _sorted_expiries(df_otm)

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
            subdf = df_day.loc[df_day["expiry_date"].eq(expiry)]

            if subdf.empty:
                logger.info(
                    f"{datetime.now()}: data for expiry date {expiry} is empty."
                )
                continue

            y = (
                (subdf["iv"]**2)*subdf["years_to_expiry"]
                if if_var else subdf["iv"]
            )

            ax.plot(
                subdf["K"],
                y,
                label=str(expiry.date()),
                alpha=0.35,
                c=colour,
            )

            if len(y) >= 2:
                collect.append(y.to_numpy())

        if collect:
            twp_avg = twp_multi_average(collect)
            lo, hi = ax.get_xlim()
            twp_idx = np.linspace(lo, hi, len(twp_avg))
            ax.plot(twp_idx, twp_avg, ls="-.", c="tab:gray", label="twp_avg")

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
    return fig


def plot_ssvi_fits(
        ssvi_params: dict[str, dict[str, np.ndarray|float]],
        df_otm: pd.DataFrame,
        *,
        max_dates: int=8,
        start: str|pd.Timestamp|None=None,
        end: str|pd.Timestamp|None=None,
        expiries: Iterable|None=None,
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
    dates = (
        pd
        .Index(ssvi_params.keys())
        .intersection(
            df_otm
            .loc[start:end]
            .index
            .unique()
            .tolist()
        )
        .sort_values()
    )

    _check_max(len(dates), max_dates, "dates")

    if expiries is None:
        expiries = _sorted_expiries(df_otm)
    else:
        expiries = np.sort(np.asarray(expiries))

    ncols = len(dates)
    nrows = len(expiries)
    figsize = figsize or (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = figsize,
        sharex  = True,
        sharey  = True,
        squeeze = False,
    )

    k_min, k_max = df_otm["K"].min(), df_otm["K"].max()
    iv_min, iv_max = df_otm["iv"].min(), df_otm["iv"].max()

    for i, dt in enumerate(dates):
        data = ssvi_params[dt]
        df = _ssvi_frame(data, dt)

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

            ax.set_xlim(k_min, k_max)
            ax.set_ylim(iv_min, iv_max)
            ax.grid(alpha=0.5)

            if j == 0:
                ax.set_title(str(dt), fontsize="x-large")

            if i == 0:
                ax.set_ylabel(f"Expiry: {exp}")

            if j == nrows-1:
                ax.set_xlabel(r"Log-forward-moneyness ($k$)")

            if i == ncols-1:
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", labelleft=False, labelright=True)
            else:
                ax.tick_params(axis="y", labelleft=False, labelright=False)

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
    return fig


def plot_ssvi_parameters(
        ssvi_params: dict[str, dict[str, np.ndarray|float]],
        risk_reversals: dict[str, dict[str, float]],
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

    `risk_reversals`: dict[str, dict[str, float]]:
        Dictionary of Risk Reversal metrics for each day. Expects structure:
        `{"date": rr_value}`

    `rr_Delta`: float
        The Delta that a RR is calculated at.

    Returns a matplotlib Figure.
    """
    params = pd.DataFrame.from_dict(
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

    rr = pd.DataFrame(risk_reversals).T
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
    return fig


def plot_ssvi_surfaces(
        ssvi_params: dict[str, dict[str, np.ndarray|float]],
        start: str|pd.Timestamp,
        end: str|pd.Timestamp,
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

    `start`: str|pd.Timestamp:
        The start date of the date range. For each day in the date range, all IV
        curves are plotted together on a shared Axes object.

    `end`: str|pd.Timestamp:
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
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    dates = sorted(
        dt for dt in ssvi_params
        if start <= pd.Timestamp(dt) <= end
    )

    if not dates:
        raise ValueError(
            f"{datetime.now()}: No SSVI surfaces found between {start.date()} "
            f"and {end.date()}."
        )

    _check_max(len(dates), max_surfaces, "surfaces")

    nrows = int(np.ceil(len(dates) / ncols))
    figsize = figsize or (6 * ncols, 6 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, dt in zip(axes, dates):
        data = ssvi_params[dt]
        k_min, k_max = data["moneyness"].min(), data["moneyness"].max()
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
            xlim   = (k_min, k_max),
            zlim   = (iv_min, 0.50),
        )
        ax.set_box_aspect((4, 4, 3), zoom=0.90)

    for ax in axes[len(dates):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig