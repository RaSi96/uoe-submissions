import logging
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Iterable

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def ometrics_surface(
        df: pd.DataFrame,
        grid: Iterable,
        h1: float=0.05,
        h2: float=0.005,
        h3: float=0.001
    ) -> pd.DataFrame:
    rows = []
    logger.info(f"{datetime.now()}: Constructing OM surfaces...")

    for dt, subdf in df.groupby(df.index):
        T_i     = subdf["years_to_expiry"].values
        delta_i = subdf["delta"].values
        cp_i    = subdf["cp_flag"].values
        iv_i    = subdf["iv"].values
        vega_i  = subdf["vega"].values

        if any(delta_i == 0):
            logger.warning(
                f"{datetime.now()}: Options with 0 Delta detected. This may "
                "lead to inaccurate results,"
            )

        for point in grid:
            delta_j, T_j, cp_j = point  # delta, days, cp
            T_j /= 365

            # distances
            x = np.log(T_i/T_j)
            y = delta_i-delta_j
            z = (cp_i!=cp_j).astype(float)

            K_x = (x**2)/(h1)  # distance between empirical and grid time
            K_y = (y**2)/(h2)  # distance between empirical and grid delta
            K_z = (z**2)/(h3)  # put/call flag
            gaussian = 1/np.sqrt(2*np.pi) * np.exp(-0.5*(K_x + K_y + K_z))
            w_numer = np.sum(vega_i*iv_i*gaussian)
            w_denom = np.sum(vega_i*gaussian)

            sigma_hat = np.nan if w_denom==0 else w_numer/w_denom

            rows.append({
                "date": dt,
                "delta": delta_j,
                "time": T_j,
                "cp": cp_j,
                "iv": sigma_hat,
            })

    om_surface = (
        pd
        .DataFrame(rows)
        .astype({
            "date" : "datetime64[ns]",
            "delta": "float",
            "time" : "float",
            "iv"   : "float",
        })
    )

    logger.info(f"{datetime.now()}: Constructed OM surface dataframe.")
    return om_surface

# ------------------------------------------------------------------------------
# References:
# [1] https://wrds-www.wharton.upenn.edu/documents/2231/IvyDB_US_v7.0_Reference_Manual.pdf