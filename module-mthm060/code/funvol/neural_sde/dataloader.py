import logging
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from torch.utils.data import Dataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def train_test_split(
        data: pd.DataFrame,
        train_pct: float=0.90,
        lags: int=0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into a train/test split. `train_pct` is the amount of data to
    retain for training. The ratio is `train_pct`:(1-`train_pct`), train:test.
    """
    N = int( len(data)*train_pct )
    train = data.iloc[:N, :]
    test = data.iloc[N-lags:, :]

    if lags != 0:
        logger.info(
            f"{datetime.now()}: Prepended last {lags} observations from "
            "`train` to `test`."
        )

    logger.info(
        f"{datetime.now()}: **Train:** "
        f"shape={train.shape}, "
        f"start={train.index.min().date()}, "
        f"end={train.index.max().date()}."
    )

    logger.info(
        f"{datetime.now()}: **Test:** "
        f"shape={test.shape}, "
        f"start={test.index.min().date()}, "
        f"end={test.index.max().date()}. "
    )

    return (train, test)


class NeuralSDEDataset(Dataset):
    """
    Returns, in order, as a tuple:
    * history : (seq_len, n_features)
    * dX      : (n_features,)
    * dT      : scalar (years)
    """
    def __init__(self, X: pd.DataFrame, lags: int=10) -> None:
        # B = batch_size,
        # T = timestamp,
        # N = n_features,
        dT = X.index.diff().dropna().days.to_numpy(dtype=np.float32) / 365

        self.X_median = X.median()
        self.X_iqr = X.quantile(0.75) - X.quantile(0.25)
        _X = (
            X
            .sub(self.X_median)
            .div(self.X_iqr)
            .to_numpy(dtype=np.float32)
        )

        self.seq_len = lags
        self.X = torch.from_numpy(_X)
        self.dT = torch.from_numpy(dT)

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(
            self,
            idx
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The way the FuNVol authors have crafted this, it appears as though the
        # increment to forecast, dX_t, is a part of the history. This shouldn't
        # the case, so my implementation here just moves dX_t ahead by 1.
        history = self.X[idx : idx+self.seq_len]
        _next   = self.X[idx + self.seq_len]

        # next observation - current observation (dX==history[-1] - history[-2])
        dX = _next - history[-1]

        # correctly index the time increment
        dT = self.dT[idx + self.seq_len-1]

        return history, dX, dT


