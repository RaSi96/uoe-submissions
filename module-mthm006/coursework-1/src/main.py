import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from typing import Literal

from pricing import *
from valuation import *

# required for local testing
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rng = np.random.default_rng(seed=42)

# ------------------------------------------------------------------------------

def main(
        seed_price: float,
        strike_price: float,
        rf: float,
        sigma: float,
        T: int,
        steps: int,
        ce: bool=True,
        method: Literal["crr", "jr"]="crr",
    ) -> float:
    """
    Computes the discounted price of a European option. Supports Jarrow-Rudd and
    Cox-Ross-Rubinstein implementations via `method`. If valuing a call option,
    set `ce=True`. If valuing a put option, set `ce=False`.
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


parser = argparse.ArgumentParser()
parser.add_argument("--price0",     type=float,    required=True)
parser.add_argument("--strike",     type=float,    required=True)
parser.add_argument("--rf",         type=float,    required=True)
parser.add_argument("--sigma",      type=float,    required=True)
parser.add_argument("--maxtime",    type=int,      required=True, default=1.0)
parser.add_argument("--nsteps",     type=int,      required=True, default=252)
parser.add_argument("--put",        default=True,  action="store_false")
parser.add_argument("--method",     type=str,      default="crr")
parser.add_argument("--bs_price",   type=float,    default=None)
parser.add_argument("--compare_jr", default=False, action="store_true")
parser.add_argument("--test_eu_us", default=False, action="store_true")

if __name__=="__main__":
    args = parser.parse_args()
    seed_price = args.price0
    strike_price = args.strike
    rf = args.rf
    sigma = args.sigma
    T = args.maxtime
    steps = args.nsteps
    ce = args.put
    method = args.method
    p_ex = args.bs_price
    compare_jr = args.compare_jr
    test_eu_us = args.test_eu_us

    discounted_price = main(
        seed_price   = seed_price,
        strike_price = strike_price,
        rf           = rf,
        sigma        = sigma,
        T            = T,
        steps        = steps,
        ce           = ce,
        method       = method
    )

    print(f"Discounted price: {discounted_price}")

    if p_ex is not None:
        logger.info(
            f"{datetime.now()}: This is a calibration exercise. Given num "
            "steps argument `--nsteps` will not be used. All other parameters "
            "will be."
        )
        logger.warning(f"{datetime.now()}: This will take ~1min 30s.")

        Ns = 2**np.arange(2, 15, 1, dtype=int)
        eps = np.empty(len(Ns))

        for i, n in enumerate(Ns):
            option_price = main(
                seed_price   = seed_price,
                strike_price = strike_price,
                rf           = rf,
                sigma        = sigma,
                T            = T,
                steps        = n,
                ce           = ce,
                method       = method
            )

            eps[i] = np.abs(option_price-p_ex)

        delta_ts = T / Ns
        slope, intercept = np.polyfit(np.log(Ns), np.log(eps), 1)
        print(f"Estimated convergence rate c: {slope:.4f}")

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

    if compare_jr:
        logger.info(
            f"{datetime.now()}: This is a calibration exercise. Given num "
            "steps argument `--nsteps` will not be used. All other parameters "
            "will be."
        )
        logger.warning(f"{datetime.now()}: This will take ~1min 30s.")

        Ns = 2**np.arange(2, 15, 1, dtype=int)
        eps_jr = np.empty(len(Ns))

        for i, n in enumerate(Ns):
            p_jr = main(
                seed_price   = seed_price,
                strike_price = strike_price,
                rf           = rf,
                sigma        = sigma,
                T            = T,
                steps        = n,
                ce           = ce,
                method       = "jr",
            )

            p_crr = main(
                seed_price   = seed_price,
                strike_price = strike_price,
                rf           = rf,
                sigma        = sigma,
                T            = T,
                steps        = n,
                ce           = ce,
                method       = "crr",
            )

            eps_jr[i] = np.abs(p_crr-p_jr)

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

    if test_eu_us:
        logger.info(
            f"{datetime.now()}: This is a calibration exercise. Given seed "
            "price argument `--price0` will not be used. All other parameters "
            "will be."
        )

        seed_prices = np.arange(1, 201, 1, dtype=float)
        option_prices = np.empty(len(seed_prices))
        immediate_pay = np.empty(len(seed_prices))

        for i, s in enumerate(seed_prices):
            option_prices[i] = main(
                seed_price   = s,
                strike_price = strike_price,
                rf           = rf,
                sigma        = sigma,
                T            = T,
                steps        = steps,
                ce           = ce,
            )

            immediate_pay[i] = max(0, strike_price-s)

        plt.plot(seed_prices, option_prices, label="EU")
        plt.plot(seed_prices, immediate_pay, label="US")
        plt.title("EU vs. American PE payoffs")
        plt.xlabel("Stock price")
        plt.ylabel("Payoff")
        plt.legend()
        plt.grid()
        plt.show()

# ------------------------------------------------------------------------------
# Jupyter notebook extract

# prices, p_star = binomial_price(
#     seed_price = 100,
#     rf         = 0.03,
#     sigma      = 0.03,
#     T          = 4,
#     steps      = 5
# )
# payoff = option_payoff(strike=105, price=prices, ce=True)
# print(f"p_star={p_star:.4f}")
# print(prices)
# print(payoff)

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

# -----

# get_option_value(
#     payoff = payoff,
#     p_star = p_star,
#     rf     = 0.03,
#     T      = 4,
#     steps  = 5
# )

# Expected output:
# np.float64(6.933613606225594)

# -----

# main(
#     seed_price   = 100,
#     strike_price = 105,
#     rf           = 0.03,
#     sigma        = 0.03,
#     T            = 4,
#     steps        = 5,
#     ce           = True
# )

# Expected output:
# np.float64(6.933613606225594)

# main(
#     seed_price   = 100,
#     strike_price = 100,
#     rf           = 0.1,
#     sigma        = 0.1,
#     T            = 1,
#     steps        = 100,
#     ce           = False
# )

# Expected output:
# np.float64(0.7804132172913312)

# ------------------------------------------------------------------------------

"EU vs. instant exercise payoff graphs"

# seed_prices = np.arange(1, 201, 1)
# option_prices = np.empty(len(seed_prices))
# immediate_pay = np.empty(len(seed_prices))

# for i, s in enumerate(seed_prices):
#     option_prices[i] = main(
#         seed_price   = s,
#         strike_price = 110,
#         rf           = 0.05,
#         sigma        = 0.4,
#         T            = 1,
#         steps        = 100,
#         ce           = False
#     )
#     immediate_pay[i] = max(0, 110-s)

# plt.plot(seed_prices, option_prices, label="EU")
# plt.plot(seed_prices, immediate_pay, label="US")
# plt.title("EU vs. American PE payoffs")
# plt.xlabel("Stock price")
# plt.ylabel("Payoff")
# plt.legend()
# plt.grid()
# plt.show()


"Log-Log plot of CRR vs. B-S"

# pows = np.arange(2, 15, 1)
# Ns = 2**pows
# p_ex = 33.608551966084981
# eps = np.empty(len(pows))

# for i, n in enumerate(Ns):
#     option_price = main(
#         seed_price   = 75,
#         strike_price = 110,
#         rf           = 0.05,
#         sigma        = 0.4,
#         T            = 1,
#         steps        = n,
#         ce           = False
#     )
#     eps[i] = np.abs(option_price-p_ex)

# plt.figure(figsize=(12, 8))
# plt.loglog(Ns, eps)
# plt.title(
#     r"Log-Log plot of $\epsilon=|P_{\text{CRR}} - P_{\text{ex}}|$ vs. "
#     r"$N \in [2^2, 2^3, \dots, d^{14}]$"
# )
# plt.xlabel("N (log scale)")
# plt.ylabel(r"$\epsilon$ (log scale)")
# plt.grid()
# plt.show()

"Log-Log plot of CRR vs. JR"

# eps_jr = np.empty(len(pows))
# for i, n in enumerate(Ns):
#     p_jr = main(
#         seed_price   = 75,
#         strike_price = 110,
#         rf           = 0.05,
#         sigma        = 0.4,
#         T            = 1,
#         steps        = n,
#         ce           = False,
#         method       = "jr",
#     )
#     p_crr = main(
#         seed_price   = 75,
#         strike_price = 110,
#         rf           = 0.05,
#         sigma        = 0.4,
#         T            = 1,
#         steps        = n,
#         ce           = False,
#         method       = "crr",
#     )
#     eps_jr[i] = np.abs(p_crr-p_jr)

# plt.figure(figsize=(12, 8))
# plt.loglog(Ns, eps_jr)
# plt.title(
#     r"Log-Log plot of $\delta=|P_{\text{CRR}} - P_{\text{JR}}|$ vs. "
#     r"$N \in [2^2, 2^3, \dots, d^{14}]$"
# )
# plt.xlabel("N (log scale)")
# plt.ylabel(r"$\delta$ (log scale)")
# plt.grid()
# plt.show()