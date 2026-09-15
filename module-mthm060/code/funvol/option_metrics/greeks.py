import logging
import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from scipy.stats import norm

from code.utils import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def get_option_delta(
        spot: pd.Series,
        strike: pd.Series,
        r: pd.Series|float,
        q: pd.Series|float,
        iv: pd.Series,
        t_diff: pd.Series
    ) -> np.ndarray:
    """
    Computes the Black-76 Delta of a given set of option contracts.

    Parameters:
    `spot`: pd.Series:
        Underlying price data series. One price per timestamp, per option.

    `strike`: pd.Series:
        Option strikes.

    `r`: pd.Series|float:
        Risk-free rate levels (r_t, not dr_t). One rate per timestamp.

    `q`: pd.Series|float:
        Underlying dividend yields. One yield per timestamp.

    `iv`: pd.Series:
        Model-implied volatilities, one per timestamp, per option.

    `t_diff`: pd.Series:
        The time to expiry, in years, of a given option contract. One value
        per timestamp, per option.

    Returns a np.ndarray, which is the Black-76 Delta for each strike/option
    contract in the dataset.
    """
    F = spot*np.exp( (r-q)*t_diff )
    numer = np.log(F/strike) + (0.5*iv**2)*t_diff
    denom = iv*np.sqrt(t_diff)

    n_zeroes = numer.eq(0).sum()
    d_zeroes = denom.eq(0).sum()
    if n_zeroes+d_zeroes > 0:
        logger.warning(
            f"{datetime.now()}: {d_zeroes} zeroes in denominator, "
            f"{n_zeroes} zeroes in numerator."
        )

    d1 = numer/denom

    # OM uses Delta for calls, Delta+1 for puts
    delta = norm.cdf(d1)
    return delta


def get_option_vega(
        spot: pd.Series,
        strike: pd.Series,
        r: pd.Series|float,
        q: pd.Series|float,
        iv: pd.Series,
        t_diff: pd.Series
    ) -> pd.Series:
    """
    Computes the Black-76 Vega of a given set of option contracts.

    Parameters:
    `spot`: pd.Series:
        Underlying price data series. One price per timestamp, per option.

    `strike`: pd.Series:
        Option strikes.

    `r`: pd.Series|float:
        Risk-free rate levels (r_t, not dr_t). One rate per timestamp.

    `q`: pd.Series|float:
        Underlying dividend yields. One yield per timestamp.

    `iv`: pd.Series:
        Model-implied volatilities, one per timestamp, per option.

    `t_diff`: pd.Series:
        The time to expiry, in years, of a given option contract. One value
        per timestamp, per option.

    Returns a pd.Series, which is the Black-76 Vega for each strike/option
    contract in the dataset.
    """
    F = spot*np.exp( (r-q)*t_diff )
    numer = np.log(F/strike) + (0.5*iv**2)*t_diff
    denom = iv*np.sqrt(t_diff)

    n_zeroes = numer.eq(0).sum()
    d_zeroes = denom.eq(0).sum()
    if n_zeroes+d_zeroes > 0:
        logger.warning(
            f"{datetime.now()}: {d_zeroes} zeroes in denominator, "
            f"{n_zeroes} zeroes in numerator."
        )

    d1 = numer/denom
    vega = F * norm.pdf(d1) * np.sqrt(t_diff)
    return vega


def compute_bhav_greeks(data_reserve: Path|str) -> pd.DataFrame:
    """
    Computes the Black-76 Delta and Vega greeks for all processed Bhavcopies in
    `data_reserve`.

    Parameters:
    `data_reserve`: Path|str:
        The location where processed Bhavcopies exist in. "Processed" means:
        * Only Nifty options exist,
        * Model-implied volatility has been obtained for each Nifty option, and
        * Only OTM Nifty options and their corresponding IVs have been retained.

    Returns a pd.DataFrame, which is a collection of all Bhavcopies (unsorted)
    with columns `delta` and `vega` assigned, for each of the 2 Greeks
    respectively.
    """
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
    return df
