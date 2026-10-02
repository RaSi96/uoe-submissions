import logging
import numpy as np
import torch

from datetime import datetime

from .legendre import *
from .fpca import *
from .detrended_price import *
from .neural_sde.dataloader import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rng = np.random.default_rng(seed=42)

# ------------------------------------------------------------------------------

def main(num_epochs: int=1_000) -> None:
    # NSDE ---------------------------------------------------------------------
    n_epochs = np.arange(1, num_epochs+1, 1)
    logger.info(
        f"{datetime.now()}: num epochs: {len(n_epochs)} (first={n_epochs[0]}, "
        f"last={n_epochs[-1]})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{datetime.now()}: Device={device}")

    nsde_ce, train_loader_ce, test_loader_ce = prepare_neural_sde(
        df     = neural_ce_price,
        device = device
    )
    logger.info(f"{datetime.now()}: Prepared call-side NSDE.")

    nsde_pe, train_loader_pe, test_loader_pe = prepare_neural_sde(
        df     = neural_pe_price,
        device = device
    )
    logger.info(f"{datetime.now()}: Prepared put-side NSDE.")

    nsde_ce, drift_stats_ce, diffusion_stats_ce, combined_stats_ce = train_all(
        nsde:    = nsde_ce,
        n_epochs = n_epochs,
        loader   = train_loader_ce
    )
    logger.info(f"{datetime.now()}: Call-side neural optimistaion completed.")

    nsde_pe, drift_stats_pe, diffusion_stats_pe, combined_stats_pe = train_all(
        nsde:    = nsde_pe,
        n_epochs = n_epochs,
        loader   = train_loader_pe
    )
    logger.info(f"{datetime.now()}: Put-side neural optimistaion completed.")