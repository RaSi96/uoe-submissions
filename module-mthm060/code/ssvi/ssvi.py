import logging
import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, NonlinearConstraint

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def compute_atm_var(df: pd.DataFrame) -> float:
    """
    Computes ATM total implied variance for a given dataframe `df`. Assumes `df`
    contains data of all options for one expiry. Then, ATM options are obtained
    as follows:
    1. Find the option with strike K closest to 0, with rtol=atol=1e-03:
        1. If there is a result from (1), use that option's corresponding IV; or
        2. Interpolate K across x=0 and get the corresponding IV.
    2. Compute (iv**2)*tau, where tau=time to expiry (in years).

    Parameters:
    `df`: pd.DataFrame:
        Dataframe of options data (including IV), where all options in `df` are
        assumed to have the same expiry.

    Returns a float, which is the ATM total implied variance.
    """
    # this is per expiry, and each option has the same expiry so its irrelevant
    # what we choose here
    time_to_atm = df["years_to_expiry"].iloc[0]

    if abs(time_to_atm) == 0:
        # ATM options at expiry have zero intrinsic value (also in [1]), so
        # \theta_0 := \lim_{t\to 0} \theta_t = 0
        return 0.0

    atm = np.isclose(df['K'], 0.0, rtol=1e-03, atol=1e-03)

    if atm.any():
        theta = df.loc[atm, "iv"].values.item()
    else:
        theta = np.interp(
            x     = 0,
            xp    = df['K'],
            fp    = df["iv"],
            left  = df["iv"].iloc[0],
            right = df["iv"].iloc[-1]
        )

    return (theta**2)*time_to_atm


def ssvi_phi(eta: float, theta: np.ndarray, gamma: float) -> np.ndarray:
    """
    The SSVI function for ATM variance, as written in [1] eq. 4.5.
    """
    denom = (theta**gamma) * (1+theta)**(1-gamma)
    return eta / denom


def static_arb_constraint(params: tuple[float, float, float]) -> float:
    """
    As mentioned in [1] rem. 4.4, this constraint is necessary for the function
    used in `ssvi_phi()` to remain free of static arbitrage. eta(1+abs(rho))<=2
    is the inequality implemented here as a minimum bound.

    Params:
    `params`: tuple[float, float, float]:
        Parameter tuple of (rho, eta, gamma) SSVI variables.

    Returns a float, which is the amount of violation.
    """
    rho, eta, gamma = params
    return 2.0 - eta*(1+abs(rho))


def ssvi_smile(
        K: np.ndarray,
        theta: np.ndarray,
        rho: float,
        eta: float,
        gamma: float
    ) -> np.ndarray:
    """
    Returns a hyperbola using Surface-SVI's parameterisation.
    """
    if np.all(abs(theta)==0):
        # if we have no ATM variance, which occurs when \tau=0, then shouldn't
        # have a curve. see def. 4.1 in [1]
        return np.zeros_like(theta)

    phi = ssvi_phi(eta, theta, gamma)
    radicand = np.sqrt( (phi*K + rho)**2 + (1-rho**2) )
    w = (theta/2) * (1 + rho*phi*K + radicand)
    return w


def ssvi_loss(
        params: tuple[float, float, float],
        moneyness: np.ndarray,
        theta: np.ndarray,
        market_smile: np.ndarray
    ) -> float:
    """
    Sum of squared error between an estimated SSVI smile and the market-observed
    smile.

    Parameters:
    `params`: tuple[float, float, float]:
        Parameter tuple of (rho, eta, gamma) SSVI variables.

    `moneyness`: np.ndarray:
        Array of log-forward-moneyness values corresponding to each option in
        `market_smile`.

    `theta`: np.ndarray:
        ATM total implied variance for a set of smiles in `market_smile`.

    `market_smile`: np.ndarray:
        A set of market-observed IV smiles. In other words, a set of IV curves
        where one curve corresponds to one expiry date.

    Returns a float, which is the sum of squared error loss.

    Note: the SSE loss is, of course, pointwise.
    """
    rho, eta, gamma = params
    computed = ssvi_smile(moneyness, theta, rho, eta, gamma)
    loss = np.sum((computed-market_smile)**2)

    # need to magnify the loss because IV as it is is probably very small,
    # ATM IV is even smaller
    return 1e6*loss


def fit_ssvi_smiles(
        moneyness: np.ndarray,
        theta: np.ndarray,
        market_smile: np.ndarray,
        time_to_exp: np.ndarray
    ) -> dict:
    """
    Simultaneously fits SSVI parameters (rho, eta, gamma) to a given market-
    observed set of IV smiles, and the array of each smiles' ATM total implied
    variance `theta`.

    Parameters:
    `moneyness`: np.ndarray:
        Array of log-forward-moneyness values corresponding to each option in
        `market_smile`.

    `theta`: np.ndarray:
        ATM total implied variance for a given `market_smile`.

    `market_smile`: np.ndarray:
        A set of market-observed IV smiles. In other words, a set of IV curves
        where one curve corresponds to one expiry date.

    `time_to_exp`: np.ndarray:
        An array of how long, in years, each option in `market_smile` will take
        to expire.

    Returns a dict containing:
    * Optimal SSVI surface parameters, applicable for an entire day
      (rho, eta, gamma);
    * `theta`,
    * The SSVI smile, computed with optimal parameters;
    * `moneyness`,
    * `time_to_exp`, and
    * The loss function evaluation for the fit.
    """
    bounds = [
        (-1.0, 1.0),  # rho
        (1e-6, 5.0),  # eta
        (1e-6, 1.0)   # gamma
    ]

    static_arb = NonlinearConstraint(static_arb_constraint, 0.0, np.inf)

    res = differential_evolution(
        ssvi_loss,
        bounds      = bounds,
        args        = (moneyness, theta, market_smile),
        constraints = (static_arb,),
        polish      = True
    )

    rho, eta, gamma = res.x
    opt_smile = ssvi_smile(moneyness, theta, rho, eta, gamma)
    opt_loss = ssvi_loss(res.x, moneyness, theta, market_smile)

    # we never end up with a case with time_to_exp==0 because we pre-filter
    # those expired options out.
    opt_smile = np.sqrt(opt_smile/time_to_exp)

    param_dict = {
        "rho"        : rho,
        "eta"        : eta,
        "gamma"      : gamma,
        "theta"      : theta,
        "ssvi_smile" : opt_smile,
        "moneyness"  : moneyness,
        "time_to_exp": time_to_exp,
        "loss"       : opt_loss,
        # "arb_violate": static_arb_constraint(res.x)
    }

    return param_dict

# ------------------------------------------------------------------------------
# References:
# [1] https://doi.org/10.1080/14697688.2013.819986