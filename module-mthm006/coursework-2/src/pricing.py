import numpy as np

from scipy.stats import norm

# ------------------------------------------------------------------------------

def eu_payoff(spot: np.ndarray, strike: float, r: float, T: int) -> np.ndarray:
    """
    Computes the payoff of a European call option: $max(0, S-E)$.
    """
    payoffs = np.maximum(0, spot-strike)
    return payoffs.mean() * np.exp(-r*T)


def exact_black_scholes(
        spot: float,
        strike: float,
        r: float=0.01,
        sigma: float=0.1,
        T: int=5,
        t: int=0,
    ):
    """
    Computes the exact Black-Scholes price of a European call option.
    """
    tau = T-t
    root_tau = np.sqrt(tau)

    strike_change = np.log(spot/strike)
    rf_vol_increment = (r + 0.5*sigma**2)*tau
    d1 = (strike_change+rf_vol_increment) / (sigma*root_tau)
    d2 = d1 - sigma*root_tau

    mprice_qtile = spot*norm.cdf(d1)
    strike_qtile = strike*np.exp(-r*tau)*norm.cdf(d2)
    cbs0 = mprice_qtile - strike_qtile
    return cbs0


def dno_payoff(
        price_paths: np.ndarray,
        barrier: float,
        strike: float,
        r: float|None = None,
        T: int|None = None,
        return_sum: bool=False,
    ) -> float:
    """
    Computes the discounted price of a Down-and-Out call option:
    $$
        Payoff per path = {
            S(T)-E:  S(T)>E && S(t) >= barrier ∀t
            0     :  otherwise
        }
    $$

    If `return_sum` is true, returns the undiscounted sum of payoffs of a Down-
    and-Out call option. Useful for MC DnO pricing.
    """
    survived = ~np.any(price_paths < barrier, axis=1)
    terminal = price_paths[:, -1]
    eu_payoff = np.maximum(terminal-strike, 0.0)
    eu_prices = np.where(survived, eu_payoff, 0.0)

    if return_sum:
        return eu_prices.sum()
    else:
        if (r is None) or (T is None):
            raise ValueError(
                "Requested discounted prices but `r`, `T` not provided."
            )
        return eu_prices.mean() * np.exp(-r*T)
