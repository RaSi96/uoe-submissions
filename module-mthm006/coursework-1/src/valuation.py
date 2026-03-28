import numpy as np

# ------------------------------------------------------------------------------

def option_payoff(
        strike: float,
        price: np.ndarray,
        ce: bool=True,
    ) -> np.ndarray:
    """
    Option payoffs. If we have a call option (ce), returns
    `max(price-strike, 0)`. If we have a put option, set `ce=False` and will
    return `max(strike-price, 0)`.
    """
    if ce:
        payoff = price-strike
    else:
        payoff = strike-price

    payoff[payoff<0] = 0
    return payoff


def get_option_value(
        payoff: np.ndarray,
        p_star: float,
        rf: float,
        T: int,
        steps: int
    ) -> float:
    """
    Computes the price of a European option today.
    """
    dt = T/steps
    discount = np.exp(-rf*dt)
    terminal_values = payoff[-(steps+1):]

    for _ in range(steps):
        up = p_star*terminal_values[1:]
        down = (1-p_star)*terminal_values[:-1]
        terminal_values = (up+down)*discount

    return terminal_values[0]