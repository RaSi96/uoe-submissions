import logging
import numpy as np
import os
import pandas as pd
import torch

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Literal

from code.utils import get_file_list
from .dataloader import *
from .training import *
from .neural_model import NeuralSDE

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


rng = np.random.default_rng(seed=42)

# ------------------------------------------------------------------------------

def main(
        data_reserve: Path|str,
        detrended_price_reserve: Path|str,
        wing: Literal["ce", "pe"],
        num_epochs: int=1_000,
        train_pct: float=0.90,
        nlags: int=10,
    ) -> None:
    if wing not in ["ce", "pe"]:
        raise ValueError(
            f"{datetime.now()}: `wing` must be either one of 'ce' or 'pe'. "
            f"Received `{wing}` instead."
        )

    if num_epochs <= 0:
        raise ValueError(
            f"{datetime.now()}: `n_epochs` must be greater than 0, received "
            f"`{num_epochs}` instead."
        )

    if not (0 <= train_pct <= 1):
        raise ValueError(
            f"{datetime.now()}: Cannot use 100% or more of data for training. "
            f"Ensure `train_pct` is in (0, 1) (received `{train_pct}`)."
        )

    n_epochs = np.arange(1, num_epochs+1, 1)
    logger.info(
        f"{datetime.now()}: num epochs: {len(n_epochs)} (start={n_epochs[0]}, "
        f"end={n_epochs[-1]})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{datetime.now()}: Training on device `{device}`.")

    # data_reserve = "./funvol/fpca/artefacts/"
    files = get_file_list(
        basedir    = data_reserve,
        glob       = f"*_{wing}.csv",
        sort_mtime = True
    )

    neural_df = pd.read_csv(files[0], parse_dates=[0], index_col=[0])
    logger.info(f"{datetime.now()}: Loaded neural df.")

    files = get_file_list(
        basedir    = detrended_price_reserve,
        glob       = "*_detrended_underlying_price.csv",
        sort_mtime = True
    )

    price = pd.read_csv(files[0], parse_dates=[0], index_col=[0])
    logger.info(f"{datetime.now()}: Loaded detrended underlying price.")

    neural_df = neural_df.join(price, how="inner")

    # --------------------------------------------------------------------------

    train, test = train_test_split(neural_df, train_pct=train_pct, lags=nlags)

    train_dataset = NeuralSDEDataset(train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # test_dataset = NeuralSDEDataset(test)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    logger.info(f"{datetime.now()}: Prepared dataloaders.")

    # --------------------------------------------------------------------------

    N = len(neural_df.T)

    nn_params_ce = {
        "n_in"    : N,
        "n_hidden": N,
        "n_layers": 3,
        "n_out"   : N,
    }

    nsde = NeuralSDE(nn_params_ce).to(device)
    logger.info(f"{datetime.now()}: Prepared neural SDE.")

    # --------------------------------------------------------------------------

    logger.info(f"{datetime.now()}: Commencing stage 1 training...")
    nsde, drift_eloss, drift_stats = train_stage_1(
        model    = nsde,
        n_epochs = n_epochs,
        loader   = train_loader,
        device   = device,
    )

    drift_eloss = pd.Series(drift_eloss, name="drift_eloss")

    drift_stats = (
        pd
        .DataFrame([s for s in drift_stats if s is not None])
        .sort_values(by=["epoch", "name"])
    )
    logger.info(
        f"{datetime.now()}: Stage 1 training completed, commencing stage 2..."
    )

    nsde, diffn_eloss, diffn_stats, alpha = train_stage_2(
        model    = nsde,
        n_epochs = n_epochs,
        loader   = train_loader,
        device   = device,
    )

    diffn_eloss = pd.Series(diffn_eloss, name="drift_eloss")

    diffn_stats = (
        pd
        .DataFrame([s for s in diffn_stats if s is not None])
        .sort_values(by=["epoch", "name"])
    )
    logger.info(
        f"{datetime.now()}: Stage 2 training completed, commencing stage 3..."
    )

    nsde, cmb_eloss, cmb_stats = train_stage_3(
        model    = nsde,
        n_epochs = n_epochs,
        loader   = train_loader,
        alpha    = alpha,
        device   = device,
    )

    cmb_eloss = pd.Series(cmb_eloss, name="drift_eloss")

    # Welcome...to City 17! It's safer here...
    cmb_stats = (
        pd
        .DataFrame([s for s in cmb_stats if s is not None])
        .sort_values(by=["epoch", "name"])
    )
    logger.info(f"{datetime.now()}: Stage 3 training completed.")

    # --------------------------------------------------------------------------

    runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    basedir = os.path.join(os.path.dirname(__file__), "artefacts")
    filename = f"{basedir}/{runtime}_neural_model_{wing}.pt"
    torch.save(nsde.state_dict(), filename)
    logger.info(
        f"{datetime.now()}: Trained neural model saved to `{filename}`."
    )

    filename = f"{basedir}/{runtime}_drift_epoch_losses_{wing}.csv"
    drift_eloss.to_csv(filename)
    logger.info(
        f"{datetime.now()}: Stage 1 epoch-wise losses saved to {filename}."
    )

    filename = f"{basedir}/{runtime}_drift_stats_{wing}.csv"
    drift_stats.reset_index().to_csv(filename, index=False)
    logger.info(
        f"{datetime.now()}: Stage 1 training statistics saved to {filename}."
    )

    filename = f"{basedir}/{runtime}_diffusion_epoch_losses_{wing}.csv"
    diffn_eloss.to_csv(filename)
    logger.info(
        f"{datetime.now()}: Stage 2 epoch-wise losses saved to {filename}."
    )

    filename = f"{basedir}/{runtime}_diffusion_stats_{wing}.csv"
    diffn_stats.reset_index().to_csv(filename, index=False)
    logger.info(
        f"{datetime.now()}: Stage 2 training statistics saved to {filename}."
    )

    filename = f"{basedir}/{runtime}_combined_epoch_losses_{wing}.csv"
    cmb_eloss.to_csv(filename)
    logger.info(
        f"{datetime.now()}: Stage 3 epoch-wise losses saved to {filename}."
    )

    filename = f"{basedir}/{runtime}_combined_stats_{wing}.csv"
    cmb_stats.reset_index().to_csv(filename, index=False)
    logger.info(
        f"{datetime.now()}: Stage 3 training statistics saved to {filename}."
    )

    filename = f"{basedir}/{runtime}_test_df_{wing}.csv"
    test.reset_index().to_csv(filename, index=False)
    logger.info(
        f"{datetime.now()}: Testing dataset for inference saved to {filename}."
    )

    return


if __name__=="__main__":
    parser = ArgumentParser(
        description = (
            "Train the FuNVol neural SDE on a given dataset. This entire "
            "subroutine outputs optimal weights, per-epoch loss, training "
            "statistics, and a test dataset as artefacts; the latter used for "
            "subsequent inference. Optimal weight tensors can simply be "
            "reloaded using PyTorch, with the same neural model spec used here."
        )
    )

    parser.add_argument(
        "--data_reserve",
        type     = Path,
        help     = "Directory of the dataset to train on",
        required = True,
    )
    parser.add_argument(
        "--detrended_price_reserve",
        type     = Path,
        help     = (
            "Directory of the underlying price time series, to use a covariate."
        ),
        required = True,
    )
    parser.add_argument(
        "--wing",
        type     = str,
        help     = (
            "Which wing of options is being trained. Calls ('ce') or "
            "puts('pe')? Must be only one of either 'ce' or 'pe'. Defaults to "
            "'ce'."
        ),
        default  = "ce",
    )
    parser.add_argument(
        "--num_epochs",
        type     = int,
        help     = "The number of epochs to train for. Defaults to 1_000.",
        default  = 1_000,
    )
    parser.add_argument(
        "--train_pct",
        type     = float,
        help     = (""
            "The percentage of data to retain for training. Defaults to 0.90, or "
            "90%."
        ),
        default  = 0.90,
    )
    parser.add_argument(
        "--nlags",
        type     = int,
        help     = (
            "How many lags to consider in the non-Markovian modelling of the "
            "given dataset. Defaults to 10 for t-10 lags."
        ),
        default  = 10,
    )

    args = parser.parse_args()

    main(**vars(args))

