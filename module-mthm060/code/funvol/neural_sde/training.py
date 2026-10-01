import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datetime import datetime
from torch.utils.data import DataLoader
from typing import Iterable

from .neural_model import NeuralSDE
from .diffusion_losses import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def track_weight_stats(
        name: str,
        vals: np.ndarray,
        epoch: int,
        log: bool=False
    ) -> dict|None:
    if vals.ndim != 2:
        return None

    cvar = vals.T @ vals                       # covariance matrix
    eigs = np.linalg.eigvalsh(cvar)            # eigvals, because square mat
    eigs = np.maximum(eigs, 1e-12)             # to prevent degeneracy
    eigs = np.sort(eigs)[::-1]                 # sort in ascending
    prop = eigs/eigs.sum()                     # proportion of eigvals
    eig1 = prop[0]                             # largest eig contrib    (REPORT)
    reff = np.exp(-(prop*np.log(prop)).sum())  # effective rank         (REPORT)

    fro_norm = np.linalg.norm(vals, ord="fro")**2
    two_norm = np.linalg.norm(vals, ord=2)**2
    stable_rank = fro_norm/two_norm            # stable rank            (REPORT)
    matrix_rank = np.linalg.matrix_rank(vals)  # actual matrix rank     (REPORT)
    matrix_size = vals.shape[1]

    if log:
        logger.info(
            f"{datetime.now()}: [{name}] "
            f"eig1={eig1:.4f}, reff={reff:.4f}, "
            f"stable_rank={stable_rank:.4f}, matrix_rank={matrix_rank}, "
            f"num_columns={matrix_size}"
        )

    stats = {
        "epoch"      : epoch,
        "name"       : name,
        "eig1"       : eig1,
        "reff"       : reff,
        "stable_rank": stable_rank,
        "matrix_rank": matrix_rank,
        "num_columns": matrix_size
    }

    return stats


def train_stage_1(
        model: NeuralSDE,
        n_epochs: np.ndarray,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[NeuralSDE, list[float], list[dict]]:
    """
    Trains drift only. Returns model, epoch_losses, weight_stats.
    """
    if n_epochs <= 0:
        # is someone trying something funny?
        raise ValueError(
            f"{datetime.now()}: `n_epochs` must be greater than 0, received "
            f"`{n_epochs}` instead."
        )

    if len(loader) <= 0:
        # is someone trying something funnier?
        raise ValueError(
            f"{datetime.now()}: `loader` must have at least one batch, received "
            f"`{len(loader)}` instead."
        )

    optim_drift = torch.optim.AdamW(
        model.drift.parameters(),
        lr=1e-3,
    )

    epoch_losses = []
    weight_stats = []

    model.train()

    for epoch in n_epochs:
        running_loss = 0.0

        for history, dX, dT in loader:
            history = history.to(device)
            dX = dX.to(device)
            dT = dT.to(device)

            optim_drift.zero_grad()
            mu = model.drift(history)
            mse_loss = F.mse_loss(input=mu*dT[:, None], target=dX)
            mse_loss.backward()
            optim_drift.step()
            loss = mse_loss.item()
            running_loss += loss

        epoch_loss = running_loss / len(loader)
        epoch_losses.append(epoch_loss)

        if epoch % 10 == 0:
            logger.info(
                f"{datetime.now()}: Epoch={epoch}, epoch loss={epoch_loss:.4f}"
            )

        for p in model.drift.named_parameters():
            name = p[0]
            vals = p[1].detach().cpu().numpy()
            stats = track_weight_stats(name, vals, epoch, log=False)
            weight_stats.append(stats)

    return model, epoch_losses, weight_stats


def train_stage_2(
        model: NeuralSDE,
        n_epochs: np.ndarray,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[NeuralSDE, list[float], list[dict], float]:
    """
    Trains diffusion only. Returns model, epoch_losses, weight_stats, alpha.
    """
    if n_epochs <= 0:
        # is someone trying something funny?
        raise ValueError(
            f"{datetime.now()}: `n_epochs` must be greater than 0, received "
            f"`{n_epochs}` instead."
        )

    if len(loader) <= 0:
        # is someone trying something funnier?
        raise ValueError(
            f"{datetime.now()}: `loader` must have at least one batch. Received "
            f"`{len(loader)}` instead."
        )

    # some linters detect paths where these variables are possibly unbounded.
    llf_loss = torch.empty()
    pit_loss = torch.empty()

    # stage 2 trains the diffusion only and disables drift.
    for p in model.drift.parameters():
        p.requires_grad_(False)

    optim_diffusion = torch.optim.AdamW(
        model.diffusion.parameters(),
        lr=1e-3,
    )

    epoch_losses = []
    weight_stats = []

    model.train()

    for epoch in n_epochs:
        running_loss = 0.0

        for history, dX, dT in loader:
            history = history.to(device)
            dX = dX.to(device)
            dT = dT.to(device)

            optim_diffusion.zero_grad()

            with torch.no_grad():
                mu = model.drift(history)

            Sigma = model.diffusion(history)
            llf_loss: torch.Tensor = log_likelihood_loss(mu, Sigma, dT, dX)
            pit_loss: torch.Tensor = density_loss(mu, Sigma, dT, dX)
            dif_loss = llf_loss + pit_loss
            dif_loss.backward()
            optim_diffusion.step()
            loss = dif_loss.item()
            running_loss += loss

        epoch_loss = running_loss / len(loader)
        epoch_losses.append(epoch_loss)

        if epoch % 10 == 0:
            logger.info(
                f"{datetime.now()}: Epoch={epoch}, epoch loss={epoch_loss:.4f}"
            )

        for p in model.diffusion.named_parameters():
            name = p[0]
            vals = p[1].detach().cpu().numpy()
            stats = track_weight_stats(name, vals, epoch, log=False)
            weight_stats.append(stats)

    order = torch.log10( torch.abs(llf_loss/pit_loss) ).floor().item()
    alpha = 10**order
    logger.info(f"{datetime.now()}: alpha'={alpha}")

    return model, epoch_losses, weight_stats, alpha


def train_stage_3(
        model: NeuralSDE,
        n_epochs: np.ndarray,
        loader: DataLoader,
        alpha: float,
        device: torch.device,
    ) -> tuple[NeuralSDE, list[float], list[dict]]:
    """
    Trains drift & diffusion. Returns model, epoch_losses, weight_stats.
    """
    if n_epochs <= 0:
        # is someone trying something funny?
        raise ValueError(
            f"{datetime.now()}: `n_epochs` must be greater than 0, received "
            f"`{n_epochs}` instead."
        )

    if len(loader) <= 0:
        # is someone trying something funnier?
        raise ValueError(
            f"{datetime.now()}: `loader` must have at least one batch. Received "
            f"`{len(loader)}` instead."
        )

    # stage 3 trains both drift && diffusion, but does so with diffusion losses
    # ONLY.
    for p in model.drift.parameters():
        p.requires_grad_(True)

    optim_combined = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    epoch_losses = []
    weight_stats = []

    model.train()

    for epoch in n_epochs:
        running_loss = 0.0

        for history, dX, dT in loader:
            history = history.to(device)
            dX = dX.to(device)
            dT = dT.to(device)

            optim_combined.zero_grad()

            mu = model.drift(history)
            Sigma = model.diffusion(history)

            _llf_loss = log_likelihood_loss(mu, Sigma, dT, dX)
            _pit_loss = density_loss(mu, Sigma, dT, dX)
            net_loss = _llf_loss + alpha*_pit_loss
            net_loss.backward()
            optim_combined.step()
            loss = net_loss.item()
            running_loss += loss

        epoch_loss = running_loss / len(loader)
        epoch_losses.append(epoch_loss)

        if epoch % 10 == 0:
            logger.info(
                f"{datetime.now()}: Epoch={epoch}, epoch loss={epoch_loss:.4f}"
            )

        for p in model.named_parameters():
            name = p[0]
            vals = p[1].detach().cpu().numpy()
            stats = track_weight_stats(name, vals, epoch, log=False)
            weight_stats.append(stats)

    return model, epoch_losses, weight_stats


def train_all(
        nsde: NeuralSDE,
        device: torch.device,
        n_epochs: Iterable,
        loader: DataLoader
    ) -> tuple[NeuralSDE, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nsde, drift_loss, drift_stats = train_stage_1(
        model    = nsde,
        n_epochs = n_epochs,
        loader   = loader,
        device   = device,
    )
    drift_stats = (
        pd
        .DataFrame([s for s in drift_stats if s is not None])
        .sort_values(by=["epoch", "name"])
    )

    nsde, diffn_eloss, diffn_stats, alpha = train_stage_2(
        model    = nsde,
        n_epochs = n_epochs,
        loader   = loader,
        device   = device,
    )
    diffusion_stats = (
        pd
        .DataFrame([s for s in diffn_stats if s is not None])
        .sort_values(by=["epoch", "name"])
    )

    nsde, cmb_eloss, cmb_stats = train_stage_3(
        model    = nsde,
        n_epochs = n_epochs,
        loader   = loader,
        alpha    = alpha,
        device   = device,
    )
    combined_stats = (
        pd
        .DataFrame([s for s in cmb_stats if s is not None])
        .sort_values(by=["epoch", "name"])
    )

    return (nsde, drift_stats, diffusion_stats, combined_stats)