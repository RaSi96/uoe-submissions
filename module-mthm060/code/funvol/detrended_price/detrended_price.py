import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path

# ------------------------------------------------------------------------------

def load_underlying(underlying_dir: Path|str, freq: str) -> pd.Series:
    nifty = pd.read_csv(
        underlying_dir,
        names       = ["date", "open", "high", "low", "close"],
        usecols     = ["date", "close"],
        header      = 0,
        parse_dates = [0],
        index_col   = [0]
    )

    # need to fill in any missing values, just in case
    valid_range = pd.date_range(nifty.index.min(), nifty.index.max(), freq=freq)
    nifty = nifty.reindex(index=valid_range, method="ffill").loc[:, "close"]
    return nifty


def transform_price(data: pd.Series) -> pd.Series:
    # funvol section 5.1
    q_9 = data.quantile(0.9)
    q_1 = data.quantile(0.1)
    c_0 = (q_9+q_1) / (q_9-q_1)
    c_1 = 2 / (q_9-q_1)
    return c_0 + c_1*data


def detrend(data: pd.Series) -> pd.Series:
    y = data.reset_index(drop=True)

    t = np.arange(1, len(y)+1, dtype=np.int64)
    X = sm.add_constant(t, has_constant="add")

    mod = sm.OLS(endog=y, exog=X, hasconst=True)
    res = mod.fit()

    trend = res.predict(X)

    ret = pd.Series(
        y.to_numpy() - np.asarray(trend),
        index = data.index,
        name  = data.name
    )

    return ret