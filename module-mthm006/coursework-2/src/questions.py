import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from pricing import *
from simulation import *

# ------------------------------------------------------------------------------

def question_1(
        r: float,
        T: int,
        S0: float,
        E:float,
        sigma: float,
        rng: np.random.Generator,
        maxsims: int=11,
    ) -> Figure:
    """
    Plots the error between a MC-estimated EU option price vs. the exact Black-
    Scholes price for said option.
    """
    # r = 0.01
    # T = 5
    # S0 = 100
    # E = 110
    k = np.arange(1, maxsims, 1)
    M = np.pow(4, k)

    exact_bs = exact_black_scholes(spot=S0, strike=E)
    errs = []

    for m in M:
        sims = gbm_terminal(
            initial_value = S0,
            mu            = r,
            sigma         = sigma,
            T             = T,
            n_sims        = m,
            quantise      = True,
            rng           = rng,
        )

        eu_price = eu_payoff(spot=sims, strike=E, r=r, T=T)
        errs.append(np.abs(eu_price - exact_bs))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.loglog(M, [i.item() for i in errs])
    ax.grid()
    ax.set_xlabel("M simulated paths")
    ax.set_ylabel("Error")
    ax.set_title("Convergence of MC-estimated EU CE price vs. exact B-S")
    fig.tight_layout()
    return fig


def question_2(
        r: float,
        T: int,
        S0: float,
        E: float,
        sigma: float,
        rng: np.random.Generator,
        M: int,
    ) -> Figure:
    """
    Plots the dependence of a MC-estimated EU CE price vs. the number of steps,
    N, in GBM-simulated price paths. The relationship between N and δt is δt=T/N.

    Note that we've used the exact GBM solution, so is independent of N by
    design.
    """
    # M = 10**6
    # r = 0.01
    # T = 5
    # S0 = 100
    # E = 110
    k = np.arange(0, 11, 1)
    Ns = np.pow(2, k)

    eu_prices = []

    for n in Ns:
        sims = gbm_terminal(
            initial_value   = S0,
            mu              = r,
            sigma           = sigma,
            T               = T,
            n_sims          = M,
            quantise        = True,
            rng             = rng,
        )

        eu_price = eu_payoff(spot=sims, strike=E, r=r, T=T)
        eu_prices.append(eu_price)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(Ns, [i.item() for i in eu_prices])
    ax.grid()
    ax.set_xlabel("N steps")
    ax.set_ylabel("EU CE price")
    ax.set_title("Dependence of MC-simulated EU CE price and num steps")
    fig.tight_layout()
    return fig


def question_4(
        r: float,
        T: int,
        S0: float,
        E: float,
        sigma: float,
        N: int,
        M: int,
        rng: np.random.Generator,
    ) -> Figure:
    """
    Plots the MC-estimated price of a Down-and-Out CE vs. its barrier, B, as it
    gets closer to S(0). Uses a grid of [0, S0) in steps of 10.
    """
    # M = 10**5
    # N = 512
    # r = 0.01
    # T = 5
    Bs = np.arange(0, S0, 10, dtype=float)

    dno_prices = []

    for b in Bs:
        sims_dno = gbm(
            mu            = r,
            sigma         = sigma,
            initial_value = S0,
            T             = T,
            n_steps       = N,
            n_sims        = M,
            quantise      = True,
            rng           = rng,
        )

        dno_price = dno_payoff(
            price_paths = sims_dno,
            barrier     = b,
            strike      = E,
            r           = r,
            T           = T
        )

        dno_prices.append(dno_price)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(Bs, [i.item() for i in dno_prices])
    ax.set_xlabel("Barrier B")
    ax.set_ylabel("Down-n-out CE price")
    ax.set_title(r"Down and out CE price as $B \to S(0)$")
    ax.grid()
    fig.tight_layout()
    return fig


def question_5(
        r: float,
        T: int,
        S0: float,
        E: float,
        B: float,
        sigma: float,
        M: int,
        rng: np.random.Generator,
        maxsteps: int=11,
    ) -> Figure:
    """
    Plots the dependence of an MC-estimated Down-and-Out CE price vs. the number
    of steps, N, in GBM-simulated price paths. The relationship between N and δt
    is δt=T/N.
    """
    # B = 95
    # M = 10**6
    # E = 110
    # r = 0.01
    # T = 5
    k = np.arange(0, maxsteps, 1)
    Ns = np.pow(2, k, dtype=int)

    dno_prices = []
    chunk_size = 10**3
    n_chunks = M//chunk_size
    discounter = np.exp(-r*T)

    for n in Ns:
        total_payoff = 0

        for _ in range(n_chunks):
            sims_dno = gbm(
                mu            = r,
                sigma         = sigma,
                initial_value = S0,
                T             = T,
                n_steps       = n,
                n_sims        = chunk_size,
                quantise      = True,
                rng           = rng,
            )

            dno_price = dno_payoff(
                price_paths = sims_dno,
                barrier     = B,
                strike      = E,
                return_sum  = True
            )

            total_payoff += dno_price

        average = (total_payoff/M)*discounter
        dno_prices.append(average)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(Ns, dno_prices)
    ax.set_xlabel("N steps")
    ax.set_ylabel("DnO CE Price")
    ax.set_title(r"Dependence of MC-simulated down-n-out price and $N$ steps")
    ax.grid()
    fig.tight_layout()
    return fig