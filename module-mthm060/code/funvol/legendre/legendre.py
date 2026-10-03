import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

from datetime import datetime
from scipy.special import eval_legendre
from typing import Literal

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def legendre_OLS(
        data: pd.DataFrame,
        cp: Literal["CE", "PE"],
        n_o: int=4,
    ) -> pd.DataFrame:
    if cp not in ["CE", "PE"]:
        raise ValueError(
            f"{datetime.now()}: Invalid `cp` flag. Must be either 'CE' or 'PE' "
            f", received {cp} instead."
        )

    df = data.loc[data["cp"].eq(cp), :]

    # degrees = [0, 1, 2, 3, 4]
    degrees = np.arange(0, n_o+1, 1)

    # norm = (1x5) array/vector
    norm = 1/np.sqrt( 2/(2*degrees +1) )

    # `np.add.outer` is the outer *addition*, not outer *product*. basically the
    # exact same thing as the outer product, just with the op being addition
    # rather than multiplication. see e.g. https://stackoverflow.com/a/33848817
    outer_sum = np.add.outer(degrees, degrees)
    # i, j = np.where( (0<outer_sum) & (outer_sum<= 4) )
    i, j = np.where(outer_sum<= 4)

    # calls and puts, regardless of day, have the same ±Δ grid, so we can just
    # collocate ±Δ once outside the loop. also since the grid is standardised,
    # shapes will be the same for all days. τ is fixed regardless of call/put.
    dates = df["date"].unique()
    x: np.ndarray = df.loc[df["date"].eq(dates[0]), "delta"].to_numpy()
    y: np.ndarray = df.loc[df["date"].eq(dates[0]), "time"].to_numpy()

    x = 2*x -1 if cp=="CE" else -2*x -1
    y = 2*( np.sqrt(y/y.max()) )-1

    co_delta = eval_legendre(degrees, x[:, None])*norm           # [nrows, 5, 1]
    co_tau = eval_legendre(degrees, y[:, None])*norm             # [nrows, 5, 1]
    legendres = (co_delta[:,:,None] * co_tau[:,None,:])[:,i,j]   # [nrows, 5, 5]
    legendres = sm.add_constant(legendres)
    logger.info(
        f"{datetime.now()}: L-shape={legendres.shape}, "
        f"L-cond={np.linalg.cond(legendres)}."
    )

    # actually at this point the loop collapses into just repeated OLS over iv
    # per day. If only it were 2026 and we had a vectorised implementation for
    # repeated OLS...🤔 Also because we've split calls and puts, the `legendres`
    # matrix has condition number ~4. so we can *really* leverage the hell out
    # of our hoists here
    Z = (
        df[["date", "iv"]]
        .assign(row=lambda x: x.groupby("date").cumcount())
        .pivot(index="row", columns="date", values="iv")        # [nrows, ndays]
    )

    Q, R = np.linalg.qr(legendres, mode="reduced")
    logger.info(
        f"{datetime.now()}: Regression shapes: Z={Z.shape}, Q={Q.shape}, "
        f"R={R.shape}."
    )

    coef = np.linalg.solve(R, Q.T @ Z).T

    cols = [f"L{i}L{j}" for i, j in zip(i, j)]
    ret = pd.DataFrame(coef, index=dates, columns=cols)
    return ret