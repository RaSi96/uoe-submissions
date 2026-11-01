import logging
import pandas as pd
import torch

from torch.utils.data import DataLoader

from ..neural_sde.dataloader import NeuralSDEDataset
from ..neural_sde.neural_model import NeuralSDE

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

@torch.no_grad()
def predict_teacher_forcing(
        model: NeuralSDE,
        dataloader: DataLoader,
        out_index: pd.Series,
        out_columns: list,
        device: torch.device,
        sample: bool=False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """
    Returns (forecasts, mus, Sigmas)
    """
    model.eval()
    forecasts = []
    mus = []
    Sigmas = []

    for history, dX, dT in dataloader:
        history = history.to(device)
        dX = dX.to(device)
        dT = dT.to(device)

        mu = model.drift(history)
        Sigma = model.diffusion(history)

        if sample:
            Z = torch.distributions.MultivariateNormal(
                loc               = mu*dT[:, None],
                covariance_matrix = Sigma*dT[:, None, None],
            )
            dX = Z.sample()
        else:
            dX = mu*dT[:, None]

        fcast = history[:, -1, :] + dX
        forecasts.append( fcast.squeeze(0).cpu().numpy() )
        mus.append( mu.squeeze(0).cpu().numpy() )
        Sigmas.append( Sigma.squeeze(0).cpu().numpy() )

    forecasts = (
        pd
        .DataFrame(
            data    = forecasts,
            index   = out_index,
            columns = out_columns
        )
        .mul(dataloader.dataset.X_iqr)
        .add(dataloader.dataset.X_median)
    )

    mus = pd.DataFrame(mus, index=out_index, columns=out_columns)
    Sigmas = [
        pd.DataFrame(S, index=out_columns, columns=out_columns)
        for S in Sigmas
    ]

    return forecasts, mus, Sigmas


@torch.no_grad()
def predict_recursive(
        model: NeuralSDE,
        dataset: NeuralSDEDataset,
        out_index: pd.Series,
        out_columns: list,
        horizon: int,
        device: torch.device,
        sample: bool=False,
        scale_fcasts: bool=True
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """
    Returns (forecasts, mus, Sigmas)
    """
    model.eval()
    forecasts = []
    mus = []
    Sigmas = []

    # seed with the first history only. that's the initial condition
    history = dataset.X[:dataset.seq_len].unsqueeze(0).to(device)

    # dT from the end of the initial history onward
    dT = dataset.dT[
        dataset.seq_len - 1 :
        dataset.seq_len - 1 + horizon
    ].to(device)

    if len(dT) < horizon:
        raise ValueError(
            f"Need {horizon} dT values, but only have {len(dT)}. "
            f"X has {len(dataset.X)} observations and "
            f"seq_len={dataset.seq_len}."
        )

    # recursive forecasting
    for k in range(horizon):
        mu = model.drift(history)
        Sigma = model.diffusion(history)

        if sample:
            Z = torch.distributions.MultivariateNormal(
                loc               = mu*dT[k],
                covariance_matrix = Sigma*dT[k],
            )
            dX = Z.sample()
        else:
            dX = mu*dT[k]

        fcast = history[:, -1, :] + dX

        forecasts.append( fcast.squeeze(0).cpu().numpy() )
        mus.append( mu.squeeze(0).cpu().numpy() )
        Sigmas.append( Sigma.squeeze(0).cpu().numpy() )

        # remember to feed the model's prediction back as the next observation
        history = torch.cat(
            (
                history[:, 1:, :],
                fcast.unsqueeze(1),
            ),
            dim=1,
        )

    forecasts = pd.DataFrame(
        data    = forecasts,
        index   = out_index[:horizon],
        columns = out_columns,
    )

    if scale_fcasts:
        forecasts = forecasts.mul(dataset.X_iqr).add(dataset.X_median)

    mus = pd.DataFrame(mus, index=out_index[:horizon], columns=out_columns)
    Sigmas = [
        pd.DataFrame(S, index=out_columns, columns=out_columns)
        for S in Sigmas
    ]

    return forecasts, mus, Sigmas

