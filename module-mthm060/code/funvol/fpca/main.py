import logging
import pandas as pd
import numpy as np
import os

from argparse import ArgumentParser
from datetime import datetime
from joblib import dump
from pathlib import Path

from .fpca import functional_PCA
from code.utils import get_file_list

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

def main(
        data_reserve_calls: Path|str|None=None,
        data_reserve_puts: Path|str|None=None,
        var_threshold: float=0.996
    ) -> None:
    # no `needed_cols` check, just trust the caller to know what they're sending
    # in. This part of the pipeline just performs regular FPCA on a dataset.
    if not (data_reserve_calls and data_reserve_puts):
        raise ValueError(
            f"{datetime.now()}: At least one directory out of "
            "`data_reserve_calls` or `data_reserve_puts` must be specified."
        )

    for id, dir in zip(["ce", "pe"], [data_reserve_calls, data_reserve_puts]):
        files = get_file_list(dir, glob=f"*_legendre-coeffs-{id}.csv")
        df = pd.concat([pd.read_csv(f, index_col=[0]) for f in files])

        princomps, eigsurfs, pca = functional_PCA(df, var_threshold)

        neural_df = pca.transform(df.sub(df.mean()))
        neural_df = pd.DataFrame(
            data = neural_df,
            index = df.index,
        columns = [f"psi_{i+1}" for i in range(len(neural_df.T))]
        )

        logger.info(
            f"{datetime.now()}: Neural {id} condition number: "
            f"{np.linalg.cond(neural_df):.4f}."
        )

        runtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        basedir = os.path.join(os.path.dirname(__file__), "artefacts")

        filename = f"{basedir}/{runtime}_neural_{id}.csv"
        neural_df.reset_index().to_csv(filename, index=False)
        logger.info(
            f"{datetime.now()}: FPCA coefficient data saved to `{filename}`."
        )

        filename = f"{basedir}/{runtime}_princomps_{id}.npy"
        np.save(filename, princomps)
        logger.info(
            f"{datetime.now()}: Eigenvalues '{id}' saved to `{filename}`."
        )

        filename = f"{basedir}/{runtime}_eigsurfs_{id}.npy"
        np.save(filename, eigsurfs)
        logger.info(
            f"{datetime.now()}: Eigensurfaces '{id}' saved to `{filename}`."
        )

        filename = f"{basedir}/{runtime}_pca_{id}.joblib"
        dump(pca, filename)
        logger.info(
            f"{datetime.now()}: PCA object '{id}' saved to `{filename}`."
        )

        logger.info(f"{datetime.now()}: Processed '{id}'.")


if __name__=="__main__":
    parser = ArgumentParser(description="Perform FPCA on (a) given dataset(s)")

    parser.add_argument(
        "--data_reserve_calls",
        type     = Path,
        help     = "Directory of the call-side file artefact.",
    )
    parser.add_argument(
        "--data_reserve_puts",
        type     = Path,
        help     = "Directory of the put-side file artefact.",
    )
    parser.add_argument(
        "--var_threshold",
        type     = float,
        help     = (
            "Variance explained threshold, determining how many princomps to "
            "retain. Defaults to 99.6% (0.996), meaning only those many "
            "principal components that explain 99.6% of data variance will be "
            "retained."
        ),
        default  = 0.996
    )

    args = parser.parse_args()

    main(**vars(args))
