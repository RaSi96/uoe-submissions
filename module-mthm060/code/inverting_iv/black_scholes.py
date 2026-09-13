import logging
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize_scalar

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def black_76(
        spot: float,
        strike: float,
        r: float,
        q: float,
        sigma: float,
        t_diff: float,
        call: bool
    ) -> float:
    """
    Computes the Black-76 price of a given option. Please ensure all provided
    measurements are as of the same day.

    Parameters:
    `spot`: float
        Scalar spot price of the given option's underlying.

    `strike`: float
        Strike price of the given option.

    `r`: float
        Risk-free rate.

    `q`: float
        Dividend yield, in the same units as `r` (usually a percentage). If a
        bank quotes a 6% yield on some fixed-income instrument whilst an asset's
        dividend yield is 1.44, then `r=0.06` and `q=0.0144`.

    `sigma`: float
        Volatility of the given option's underlying asset. This quantity is
        usually optimised for.

    `t_diff`: float
        Time remaining to expiry, in years (ACT/365). Usually computed as:
        $$ \frac{t_T-t_0}{365} $$

        Where $t_T$ is the expiry date of the given option, $t_0$ is some time
        before $t_T$ (i.e. now).

    `call`: bool
        Boolean flag indicating whether the given option is a call or put.

    Returns a scalar float, which is the Black-76 price of the given option.
    """
    F = spot*np.exp( (r-q)*t_diff )
    numer = np.log(F/strike) + (0.5*sigma**2)*t_diff
    denom = sigma*np.sqrt(t_diff)

    # can infer this behaviour from the B76 asymptotic t\to\tau
    if denom == 0:
        d1 = np.inf
    else:
        d1 = numer/denom

    d2 = d1 - sigma*np.sqrt(t_diff)

    if call:
        payoff = F*norm.cdf(d1) - strike*norm.cdf(d2)
    else:
        payoff = strike*norm.cdf(-d2) - F*norm.cdf(-d1)

    return np.exp(-r*(t_diff))*payoff


def sigma_loss(sigma: float, record: pd.Series) -> float:
    """
    Computes the squared-error between an estimated `sigma` and the market-
    observed price of an option `record`. This subroutine is usually used in
    numerical inversion of B76, to obtain model-implied volatility.

    Parameters:
    `sigma`: float
        Guesstimate for the volatility of an option contract, given its observed
        market price in `record`.

    `record`: float
        Data record of an option contract. Must contain at least the following
        accessible key-value pairs (either as a pd.Series, pd.DataFrame, dict,
        etc.):
        * "nifty": spot price of the option's underlying,
        * "strike": strike price of the given option,
        * "div_yield": dividend yield of the option's underlying,
        * "years_to_expiry": years to expiry of the given option,
        * "cp_flag": Literal["CE", "PE"] to help identify whether the given
          option is a call ("CE") or put ("PE"), and
        * "close": last traded market-obseved price of the given option.

    Returns a scalar float, which is the squared-error.
    """
    computed = black_76(
        spot   = record["nifty"],
        strike = record["strike"],
        r      = 0.10,
        q      = record["div_yield"],
        sigma  = sigma,
        t_diff = record["years_to_expiry"],
        call   = True if record["cp_flag"]=="CE" else False
    )
    loss = (computed-record["close"])**2
    return loss


def compute_iv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Meta function that tries to compute the Black-76 model-implied volatility of
    all option contracts in `df`.

    Parmeters:
    `df`: pd.DataFrame:
        Dataset of all option market data.Must contain at least the following
        accessible key-value pairs (either as a pd.Series, pd.DataFrame, dict,
        etc.):
        * "nifty": spot price of the option's underlying,
        * "strike": strike price of the given option,
        * "div_yield": dividend yield of the option's underlying,
        * "years_to_expiry": years to expiry of the given option,
        * "cp_flag": Literal["CE", "PE"] to help identify whether the given
          option is a call ("CE") or put ("PE"), and
        * "close": last traded market-obseved price of the given option.

    Returns `df` with an added column `iv`, which is the optimal model-implied
    volatility for each option contract in `df`.
    """
    logger.info(
        f"{datetime.now()}: Beginning IV optimisation over {len(df)} rows..."
    )

    ivs = np.empty(shape=len(df))

    for i, record in enumerate(df.iterrows()):
        res = minimize_scalar(
            fun = lambda x: sigma_loss(x, record=record[1]),
            bounds = (1e-3, 2)
        )

        if not res.success:
            raise ValueError(f"Error with record {i}:\n`{res.message}`\n")

        ivs[i] = res.x

    return df.assign(iv=ivs)


def filter_otm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters given option contracts in `df` to out of the money (OTM) contracts.
    At expiry, calls are OTM if spot<strike whilst puts are OTM if spot>strike.

    Returns a modified `df`, filtered to only include OTM options.
    """
    otm_calls = df.loc[
        (df["cp_flag"].eq("CE"))
        & (df["strike"].gt(df["nifty"]))
    ]

    otm_puts = df.loc[
        (df["cp_flag"].eq("PE"))
        & (df["strike"].lt(df["nifty"]))
    ]

    df = (
        pd
        .concat([otm_calls, otm_puts], axis=0)
        .reset_index()
        .sort_values(by=["date", "expiry_date", "strike"])
        .set_index("date")
        .pipe(lambda x: x.assign( K=np.log(x["strike"]/x["nifty"])) )
    )

    logger.info(f"{datetime.now()}: Filtered OTM only, assigned moneyness.")
    return df
