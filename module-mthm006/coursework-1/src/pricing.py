import numpy as np

# ------------------------------------------------------------------------------

def binomial_price(
        seed_price: float,
        rf: float,
        sigma: float,
        T: int,
        steps: int,
    ) -> tuple[np.ndarray, float]:
    """
    Binomial options pricing model, Cox-Ross-Rubinstein implementation.
    """
    dt = T/steps

    # CRR spec
    up = np.exp(sigma*np.sqrt(dt))
    down = 1/up
    p_star = (np.exp(rf*dt)-down) / (up-down)

    n_leafs = ((steps+1)*(steps+2))/2
    prices = np.zeros(int(n_leafs)-1)
    t = 0

    for i in range(1, steps+1):
        for j in range(0, i+1):
            prices[t] = seed_price*up**j*down**(i-j)
            t += 1

    return prices, p_star


def binomial_price_jr(
        seed_price: float,
        rf: float,
        sigma: float,
        T: int,
        steps: int,
    ) -> tuple[np.ndarray, float]:
    """
    Binomial options pricing model, Jarrow-Rudd implementation.
    """
    dt = T/steps

    # JR spec
    p_star = 0.5
    drift = (rf - 0.5*sigma**2)*dt
    diffn = sigma*np.sqrt(dt)
    up = np.exp(drift+diffn)
    down = np.exp(drift-diffn)

    n_leafs = ((steps+1)*(steps+2))/2
    prices = np.zeros(int(n_leafs)-1)
    t = 0

    for i in range(1, steps+1):
        for j in range(0, i+1):
            prices[t] = seed_price*up**j*down**(i-j)
            t += 1

    return prices, p_star