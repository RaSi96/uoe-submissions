import logging
import numpy as np
import pandas as pd

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
    d1 = numer/denom

    # OM uses Delta for calls, Delta+1 for puts
    delta = norm.cdf(d1)
    logger.info(f"{datetime.now()}: Computed option Delta(s).")
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
    d1 = numer/denom

    vega = F * norm.pdf(d1) * np.sqrt(t_diff)
    logger.info(f"{datetime.now()}: Computed option Vega(s).")
    return vega
