import logging
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.stats import norm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def get_option_delta(
        K: pd.Series,
        iv: pd.Series,
        t_diff: pd.Series
    ) -> pd.Series | np.ndarray:
    """
    Computes the delta (Greek) of a given option.

    Parameters:
    `K`: pd.Series:
        Series of option strike prices.

    `iv`: pd.Series:
        Model-implied volatilities of a set of options.

    `t_diff`: pd.Series:
        Time to expiry, in years, of a set of options.
    """
    numer = -K + (0.5*iv**2)*t_diff
    denom = iv*np.sqrt(t_diff)

    n_zeroes = numer.eq(0).sum()
    d_zeroes = denom.eq(0).sum()
    if n_zeroes+d_zeroes > 0:
        logger.warning(
            f"{datetime.now()}: {d_zeroes} zeroes in denominator, "
            f"{n_zeroes} zeroes in numerator."
        )

    d1 = numer/denom

    # can infer this behaviour from the B-S boundary condition for t\to\tau
    d1.loc[abs(d1).eq(np.inf)] = np.inf

    delta = norm.cdf(d1)
    return delta


def compute_risk_rev(
        ln_money: pd.Series,
        iv: pd.Series,
        tau: pd.Series,
        Delta: float=0.25,
    ) -> float:
    """
    Computes the Risk Reversal (RR) of a set of options. Risk Reversal is:
    $$ RR(x) = \sigma_{x\Delta}^{CE} - \sigma_{x\Delta}^{PE} $$

    Parameters:
    `ln_money`: pd.Series:
        Series of log-forward-moneyness for a set of corresponding options.

    `iv`: pd.Series:
        Model-implied volatilities of a set of options.

    `tau`:  pd.Series:
        Time to expiry, in years, of a set of options.

    `Delta`: pd.Series:
        The Delta to compute the RR at. Default=0.25, or the 25 Delta RR.

    Returns a float, which is the RR of a given option.

    Note: the parameters (except `Delta`) may be vectors, but the returned value
    is a float. This is because RR is only computed from the two options at a
    specific Delta, but the entire IV curve is needed to select options at that
    particular Delta.
    """
    # recall that \Delta=0.50 is at the money (ATM)
    deltas = get_option_delta(K=-ln_money, iv=iv, t_diff=tau)

    iv_c = np.interp(Delta, deltas, iv)
    iv_p = np.interp(1-Delta, deltas, iv)
    return iv_c - iv_p