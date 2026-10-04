import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from code.funvol.neural_sde.dataloader import NeuralSDEDataset
from code.funvol.neural_sde.neural_model import NeuralSDE
from code.utils import get_file_list

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rng = np.random.default_rng(seed=42)

# ------------------------------------------------------------------------------

def main(neural_data_reserve: Path|str, wing: str) -> None:
    files = get_file_list(
        basedir    = neural_data_reserve,
        glob       = f"*test_df_{wing}.csv",
        sort_mtime = True
    )

    test_df = pd.read_csv(files[0], parse_dates=[0], index_col=[0])
    logger.info(f"{datetime.now()}: Loaded test dataset.")

    N = len(test_df.T)

    nn_params = {
        "n_in"    : N,
        "n_hidden": N,
        "n_layers": 3,
        "n_out"   : N,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{datetime.now()}: Infering on device `{device}`.")
    nsde = NeuralSDE(nn_params).to(device)

    files = get_file_list(
        basedir    = neural_data_reserve,
        glob       = f"*neural_model_{wing}.pt",
        sort_mtime = True
    )

    weights = torch.load(files[0], weights_only=True)
    nsde.load_state_dict(weights)
    nsde.eval()
    logger.info(f"{datetime.now()}: Loaded NSDE weights successfully.")

    test_dataset = NeuralSDEDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    logger.info(f"{datetime.now()}: Prepared testing dataloader.")

    # how are we creating forecasts?
    # recursively?
    # teacher forcing?
    # how long ahead for, a specific period? Or the entire test length?
    # are we doing LSA?
    # are we comparing with an AR(1)-GARCH(1,1)?
    #   If so, must this be only analytically? Or progammatic, as part of the
    #   pipleine?