import logging
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.decomposition import PCA

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

# equivalent manual implementation, just without retaining the top M eigvals:
# def eigbreak_covariance(df: pd.DataFrame):
#     n = len(df)
#     df_c = df.sub(df.mean())
#
#     covar = (df_c.T @ df_c) / (n-1)
#     eigvals, eigvecs = np.linalg.eigh(covar)
#
#     # projection -- eigenvectors are in columns, not rows.
#     proj = df_c @ eigvecs
#     return proj


def functional_PCA(
        data: np.ndarray|pd.DataFrame,
        var_threshold: float=0.996,
    ) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Returns (princomps, explained_var_ratios, fitted PCA object), filtered only
    to the number of princomps that explain up to `var_threshold` amount of
    variance.
    """
    if not 0<var_threshold<=1:
        raise ValueError(
            f"{datetime.now()}: Received invalid `var_threshold` value: "
            f"{var_threshold}. Amount of explained variance retained must be "
            "in (0, 1]."
        )

    pca = PCA()
    pca.fit(data - data.mean())

    # i'm clobbering variables here because we really don't need so many of them
    # all around. they can afford to be reused WLOG. first get the explained var
    # ratios, select only those components' indices that explain up to threshold
    # ; report the cumulative explained variance ratio, and return only those
    # components.
    eigenvars = pca.explained_variance_ratio_
    princomps = np.flatnonzero( np.cumsum(eigenvars)<var_threshold )
    eigenvars = eigenvars[princomps]
    princomps = pca.components_[princomps, :]

    logger.info(
        f"{datetime.now()}: PCA total eigenvariance (n={len(eigenvars)})="
        f"{eigenvars.sum():.4f}."
    )

    # i think it's principled to first get the full eigendecomp, find out how
    # many eigenvs we need to explain `threshold` amt of variance, then refit
    # with those many components and return that object instead of the full one.
    # that way with `pca.transform(X)`, we don't need to worry about
    # n_components.
    pca = PCA(n_components=len(eigenvars))
    pca.fit(data - data.mean())

    return (princomps, eigenvars, pca)