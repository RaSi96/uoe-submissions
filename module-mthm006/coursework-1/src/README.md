# MTHM006 Assignment 1 Code
Greetings! This document briefly sets out how to use the code in this folder.

## Folder Structure
Ensure all `.py` files are placed with each other, because `main.py` imports from `pricing.py` and `valuation.py`. The top-level `__init__.py` ensures `main` can see & fetch things from `pricing, valuation`, so all `.py` files need to be placed in the same folder. If you extracted the submission from a `.zip` archive, there's a good chance everything's setup already. For convenience, we'll call the folder with all Python source files `/src/`:

```
/student_id
    /src
        ./__init__.py
        ./main.py
        ./pricing.py
        ./valuation.py
```

## Generating Submission Plots
Assuming you are in the `/src` folder, here are some examples that demonstrate how to use the program & generate the plots as shown in the submission. Use the exact parameters from here to obtain the discussed plots.

1. This first example demonstrates the calibration exercise from Question 4.A:
    ```bash
    $ /src> python main.py --price0=100 --strike=100 --rf=0.1 --sigma=0.1 --maxtime=1 --nsteps=100
    Discounted price: 10.296671413695828
    $ /src> python main.py --price0=100 --strike=100 --rf=0.1 --sigma=0.1 --maxtime=1 --nsteps=100 --put
    Discounted price: 0.7804132172913312
    ```
    The flag `--put` ensures we price a put option. Pricing a call option is the default; just leave it blank with no trailing `--call`.

2. This example demonstrates how to generate the EU put option vs. instant exercise (American) payoff graph as per Question 4.B:
    ```bash
    $ /src> python main.py --price0=100 --strike=110 --rf=0.05 --sigma=0.4 --maxtime=1 --nsteps=100 --put --test_eu_us
    Discounted price: 18.646037657218947
    INFO:__main__:2026-03-02 23:34:30.806883: This is a calibration exercise. Given seed price argument `--price0` will not be used. All other parameters will be.
    ```

3. This example demonstrates how to generate the convergence plot between the CRR binomial tree and a given Black-Scholes exact option price, $P_{\text{ex}}$, as per Question 4.C. Use the flag `--bs_price=<price>`:
    ```bash
    $ /src> python main.py --price0=75 --strike=110 --rf=0.05 --sigma=0.4 --maxtime=1 --nsteps=100 --put --bs_price=33.608551966084981
    Discounted price: 18.646037657218947
    INFO:__main__:2026-03-02 23:34:30.806883: This is a calibration exercise. Given num steps argument `--nsteps` will not be used. All other parameters will be.
    WARNING:__main__:2026-03-02 23:52:36.103056: This will take ~1min 30s.
    Estimated convergence rate c: -1.0914
    ```
    After generating the error array $\epsilon$, a 1D polynomial is also fit between $\ln(N), \ln(\epsilon)$. **Note** that we compare $\epsilon$ to $N$ rather than to $\delta t = 1/N$, because as mentioned in the submission, this way we get a downward sloping graph matching our intuition that "error _decreases_ as $\delta t \to 0$" (which means error decreases as $N \to \infty$).

4. This example demonstrates how to generate the convergence plot between the CRR binomial tree and the JR binomial tree, as per question 4.D:
    ```bash
    $ /src> python main.py --price0=100 --strike=110 --rf=0.05 --sigma=0.4 --maxtime=1 --nsteps=100 --put --compare_jr
    Discounted price: 18.646037657218947
    INFO:__main__:2026-03-02 23:34:30.806883: This is a calibration exercise. Given num steps argument `--nsteps` will not be used. All other parameters will be.
    WARNING:__main__:2026-03-02 23:56:39.533771: This will take ~1min 30s.
    ```

## Command Line Parameters
If you choose to run the program from the command line, here is the full list of accepted parameters:
- `--price0`: the seed price, $S(0)$. Float value. Note that this value will not be used when comparing EU discounted payoffs vs. US discounted payoffs, a fixed array of prices as specified in Quetion 4.B will be used instead: $S(0) \in [1, 200]$.
- `--strike=<value>`: the strike price of the option. Float value.
- `--rf=<value>`: the risk free rate. Float value.
- `--sigma=<value>`: scale of the variance. Float value.
- `--maxtime=<value>`: $\tau$; expiry time/maturity time. Integer. Default is 1.
- `--nsteps=<value>`: $N$, the number of steps to take between $[0, \tau]$. Default is 252. Note that this value will not be used when comparing CRR vs. JR, or CRR vs. an exact Black Scholes price. A fixed array of $N$ will be used instead: $N \in [2^2, 2^3, 2^4, \dots, 2^{14}]$.
- `--put`: whether or not we're valuing a put option (PE). Boolean flag. If we are valuing a PE, use this flag. If we are valuing a call option (CE), do not use this flag. CE valuation is the default. No value must be provided against this flag, it's simply a boolean switch. See example (1) under [Generating Submission Plots](#generating-submission-plots).
- `--method=<value>`: whether to use the CRR method (default) or JR method of the binomial tree. String. Accepted options are `--method=crr` or `--method=jr`. Anything else will raise an error.
- `--bs_price=<value>`: a specific, exact Black-Scholes price to compare convergence of the CRR tree with. Float value. If not provided, no comparison subroutine is invoked. If provided, a comparison subroutine will be called & a convergence-error plot generated. See example (3) under [Generating Submission Plots](#generating-submission-plots).
- `--compare_jr`: whether to compare the CRR method vs. the JR implementation and produce a graph. Boolean flag. No value must be provided against this flag, it's simply a boolean switch. See example (4) under [Generating Submission Plots](#generating-submission-plots).
- `--test_eu_us`: whether to compare discounted EU payoffs (the full binomial tree) vs. immediate exercise decisions (emulating US option payoffs) and produce a graph. Boolean flag. No value must be provided against this flag, it's simply a boolean switch. See example (2) under [Generating Submission Plots](#generating-submission-plots).

## If Dissecting Code
If you instead wish to dissect this program & inspect each code piece yourself, then provided here is the skeleton layout of the R&D Jupyter Notebook. `#cell ---` comments indicate which code snippets were in their own cell:

```python
# cell -------------------------------------------------------------------------
import logging
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from typing import Literal

# required for local testing
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rng = np.random.default_rng(seed=42)

# cell -------------------------------------------------------------------------
def option_payoff(
        strike: float,
        price: np.ndarray,
        ce: bool=True,
    ) -> float|np.ndarray:
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

# cell -------------------------------------------------------------------------
prices, p_star = binomial_price(
    seed_price = 100,
    rf         = 0.03,
    sigma      = 0.03,
    T          = 4,
    steps      = 5
)

payoff = option_payoff(strike=105, price=prices, ce=True)

print(f"p_star={p_star:.4f}")
print(prices)
print(payoff)

# Expected output:
# p_star=0.9459
# [ 97.35239858 102.71960574  94.7748951  100.         105.51317403
#   92.26563363  97.35239858 102.71960574 108.38271637  89.8228074
#   94.7748951  100.         105.51317403 111.33029894  87.44465748
#   92.26563363  97.35239858 102.71960574 108.38271637 114.35804414]
# [0.         0.         0.         0.         0.51317403 0.
#  0.         0.         3.38271637 0.         0.         0.
#  0.51317403 6.33029894 0.         0.         0.         0.
#  3.38271637 9.35804414]

# cell -------------------------------------------------------------------------
def get_option_value(
        payoff: np.ndarray,
        p_star: float,
        rf: float,
        T: int,
        steps: int
    ) -> np.ndarray:
    """
    Computes the price of a European option today.
    """
    # grid = np.linspace(0, T, steps)
    # dt = np.diff(grid)[0]
    dt = T/steps
    discount = np.exp(-rf*dt)
    terminal_values = payoff[-(steps+1):]

    for _ in range(steps):
        up = p_star*terminal_values[1:]
        down = (1-p_star)*terminal_values[:-1]
        terminal_values = (up+down)*discount

    return terminal_values[0]

# cell -------------------------------------------------------------------------
get_option_value(
    payoff = payoff,
    p_star = p_star,
    rf     = 0.03,
    T      = 4,
    steps  = 5
)

# Expected output:
# np.float64(6.933613606225594)

# cell -------------------------------------------------------------------------
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

# cell -------------------------------------------------------------------------
def main(
        seed_price: float,
        strike_price: float,
        rf: float,
        sigma: float,
        T: float,
        steps: float,
        ce: bool=True,
        method: Literal["crr", "jr"]="crr",
    ) -> float:
    """
    Computes the discounted price of a European option. Supports Jarrow-Rudd and
    Cox-Ross-Rubinstein implementations via `method`.
    """
    if method=="crr":
        prices, p_star = binomial_price(
            seed_price = seed_price,
            rf         = rf,
            sigma      = sigma,
            T          = T,
            steps      = steps,
        )
    elif method=="jr":
        prices, p_star = binomial_price_jr(
            seed_price = seed_price,
            rf         = rf,
            sigma      = sigma,
            T          = T,
            steps      = steps,
        )
    else:
        raise ValueError(
            f"{datetime.now()}: Unsupported method: {method}. Only `crr` "
            "(Cox-Ross-Rubinstein) and `jr` (Jarrow-Rudd) are supported."
        )

    payoff = option_payoff(strike=strike_price, price=prices, ce=ce)

    final = get_option_value(
        payoff = payoff,
        p_star = p_star,
        rf     = rf,
        T      = T,
        steps  = steps,
    )

    return final

# cell -------------------------------------------------------------------------
main(
    seed_price   = 100,
    strike_price = 105,
    rf           = 0.03,
    sigma        = 0.03,
    T            = 4,
    steps        = 5,
    ce           = True
)

# Expected output:
# np.float64(6.933613606225594)

# cell -------------------------------------------------------------------------
main(
    seed_price   = 100,
    strike_price = 100,
    rf           = 0.1,
    sigma        = 0.1,
    T            = 1,
    steps        = 100,
    ce           = False
)

# Expected output:
# np.float64(0.7804132172913312)

# cell -------------------------------------------------------------------------
seed_prices = np.arange(1, 201, 1)
option_prices = np.empty(len(seed_prices))
immediate_pay = np.empty(len(seed_prices))

for i, s in enumerate(seed_prices):
    option_prices[i] = main(
        seed_price   = s,
        strike_price = 110,
        rf           = 0.05,
        sigma        = 0.4,
        T            = 1,
        steps        = 100,
        ce           = False
    )

    immediate_pay[i] = max(0, 110-s)

plt.plot(seed_prices, option_prices, label="EU")
plt.plot(seed_prices, immediate_pay, label="US")
plt.title("EU vs. American PE payoffs")
plt.xlabel("Stock price")
plt.ylabel("Payoff")
plt.legend()
plt.grid()
plt.show()

# cell -------------------------------------------------------------------------
pows = np.arange(2, 15, 1)
Ns = 2**pows
p_ex = 33.608551966084981
eps = np.empty(len(pows))

for i, n in enumerate(Ns):
    option_price = main(
        seed_price   = 75,
        strike_price = 110,
        rf           = 0.05,
        sigma        = 0.4,
        T            = 1,
        steps        = n,
        ce           = False
    )

    eps[i] = np.abs(option_price-p_ex)

plt.figure(figsize=(12, 8))
plt.loglog(Ns, eps)
plt.title(
    r"Log-Log plot of $\epsilon=|P_{\text{CRR}} - P_{\text{ex}}|$ vs. "
    r"$N \in [2^2, 2^3, \dots, d^{14}]$"
)
plt.xlabel("N (log scale)")
plt.ylabel(r"$\epsilon$ (log scale)")
plt.grid()
plt.show()

# cell -------------------------------------------------------------------------
delta_ts = 1.0 / Ns  # Assuming T=1
slope, intercept = np.polyfit(np.log(Ns), np.log(eps), 1)
print(f"Estimated convergence rate c: {slope:.4f}")

# Expected output:
# Estimated convergence rate c: -1.0914

# cell -------------------------------------------------------------------------
eps_jr = np.empty(len(pows))

for i, n in enumerate(Ns):
    p_jr = main(
        seed_price   = 75,
        strike_price = 110,
        rf           = 0.05,
        sigma        = 0.4,
        T            = 1,
        steps        = n,
        ce           = False,
        method       = "jr",
    )

    p_crr = main(
        seed_price   = 75,
        strike_price = 110,
        rf           = 0.05,
        sigma        = 0.4,
        T            = 1,
        steps        = n,
        ce           = False,
        method       = "crr",
    )

    eps_jr[i] = np.abs(p_crr-p_jr)

# cell -------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
plt.loglog(Ns, eps_jr)
plt.title(
    r"Log-Log plot of $\delta=|P_{\text{CRR}} - P_{\text{JR}}|$ vs. "
    r"$N \in [2^2, 2^3, \dots, d^{14}]$"
)
plt.xlabel("N (log scale)")
plt.ylabel(r"$\delta$ (log scale)")
plt.grid()
plt.show()
```