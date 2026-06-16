import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from questions import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rng = np.random.default_rng(seed=42)

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--price0",      type=float, default=100)
parser.add_argument("--strike",      type=float, default=110)
parser.add_argument("--rf",          type=float, default=0.01)
parser.add_argument("--sigma",       type=float, default=0.1)
parser.add_argument("--maxtime",     type=int,   default=5)
parser.add_argument("--nsteps",      type=int,   default=252)
parser.add_argument("--nsims",       type=int,   default=10**6)
parser.add_argument("--dno_barrier", type=float, default=95)

parser.add_argument("--plot_gbm_path", default=False, action="store_true")
parser.add_argument("--question_1",    default=False, action="store_true")
parser.add_argument("--question_2",    default=False, action="store_true")
parser.add_argument("--question_4",    default=False, action="store_true")
parser.add_argument("--question_5",    default=False, action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    seed_price = args.price0
    strike_price = args.strike
    rf = args.rf
    sigma = args.sigma
    T = args.maxtime
    B = args.dno_barrier
    n_steps = args.nsteps
    n_sims = args.nsims

    logger.info(
        f"{datetime.now()}: Running with params `S0={seed_price}`, "
        f"`E={strike_price}`, `rf={rf}`, `sigma={sigma}`, `T={T}`, `B={B}`, "
        f"`n_steps (N)={n_steps}`, `n_sims (M)={n_sims}`."
    )

    if args.plot_gbm_path:
        # mainly just used to ensure the implementation's working right
        sims = gbm(
            mu            = rf,
            sigma         = sigma,
            initial_value = seed_price,
            T             = T,
            n_steps       = n_steps,
            n_sims        = 1,
            rng           = rng,
        )

        plt.figure(figsize=(6, 4))
        plt.plot(sims.flatten())
        plt.xlabel("Time")
        plt.ylabel("Price level")
        plt.title("GBM Simulated Price Path (exact solution)")
        plt.grid()
        plt.show()

    if args.question_1:
        question_1(
            r       = rf,
            T       = T,
            S0      = seed_price,
            E       = strike_price,
            sigma   = sigma,
            maxsims = 11,
            rng     = rng
        )

        plt.show()

    if args.question_2:
        question_2(
            r     = rf,
            T     = T,
            S0    = seed_price,
            E     = strike_price,
            sigma = sigma,
            M     = n_sims,
            rng   = rng
        )

        plt.show()

    if args.question_4:
        question_4(
            r     = rf,
            T     = T,
            S0    = seed_price,
            E     = strike_price,
            sigma = sigma,
            N     = n_steps,
            M     = n_sims,
            rng   = rng
        )

        plt.show()

    if args.question_5:
        question_5(
            r        = rf,
            T        = T,
            S0       = seed_price,
            E        = strike_price,
            B        = B,
            sigma    = sigma,
            M        = n_sims,
            maxsteps = 11,
            rng      = rng
        )

        plt.show()


