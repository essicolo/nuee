"""
Principal Component Analysis (PCA) implementation without third-party
dependencies. The full singular value decomposition is retained so that
scores can be re-scaled post hoc to match vegan conventions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

from .base import OrdinationResult, OrdinationMethod
from .rda import _build_rda_scaling_backend


class PCA(OrdinationMethod):
    """Principal Component Analysis for ordination."""

    def __init__(self, n_components: Optional[int] = None,
                 scale: bool = True, center: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.center = center

    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> OrdinationResult:
        """Fit PCA to the data using an explicit SVD."""
        species_names = None
        if hasattr(X, 'columns'):
            species_names = list(X.columns)

        matrix, column_means, column_scale = self._prepare_matrix(X)
        n_samples, n_features = matrix.shape

        denom = max(n_samples - 1, 1)
        normalization = np.sqrt(denom)
        normalized = matrix / normalization

        U, singular, Vt = np.linalg.svd(normalized, full_matrices=False)

        if singular.size == 0:
            eigenvalues = singular
        else:
            eigenvalues = singular ** 2

        if self.n_components is not None:
            k = min(self.n_components, singular.size)
            U = U[:, :k]
            singular = singular[:k]
            Vt = Vt[:k, :]
            eigenvalues = eigenvalues[:k]

        scores = U * singular if singular.size else U
        scores = scores * normalization if singular.size else scores

        row_weights = np.full(n_samples, 1.0 / denom, dtype=float)
        column_weights = np.ones(n_features, dtype=float)

        call_info = {
            "method": "PCA",
            "scale": self.scale,
            "center": self.center,
            "n_components": self.n_components
        }

        scaling_backend = _build_rda_scaling_backend(
            cca_wa=U if U.size else None,
            ca_u=None,
            cca_v=Vt.T if Vt.size else None,
            ca_v=None,
            eigenvalues=eigenvalues,
            tot_chi=float(np.sum(eigenvalues)),
            n_samples=n_samples
        )

        result = OrdinationResult(
            points=scores,
            species=Vt.T,
            eigenvalues=eigenvalues,
            call=call_info,
            site_u=U if singular.size else None,
            species_v=Vt.T if singular.size else None,
            singular_values=singular if singular.size else None,
            row_weights=row_weights,
            column_weights=column_weights,
            scaling_backend=scaling_backend
        )

        if species_names is not None:
            result._species_names = species_names
        result.column_means = column_means
        result.column_scale = column_scale
        result.row_masses = np.full(n_samples, 1.0 / max(n_samples, 1), dtype=float)
        total_variance = float(np.sum(eigenvalues))
        result.total_variance = total_variance
        result.tot_chi = total_variance
        return result

    def _prepare_matrix(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate, centre, and scale the data matrix."""
        X_arr = self._validate_data(X)
        column_means = np.mean(X_arr, axis=0)
        if self.center:
            centred = X_arr - column_means
        else:
            centred = X_arr.copy()
        column_scale = np.ones_like(column_means, dtype=float)
        if self.scale:
            column_scale = np.std(centred, axis=0, ddof=1)
            zero_mask = column_scale == 0
            column_scale[zero_mask] = 1.0
        transformed = centred / column_scale
        return transformed, column_means, column_scale


def pca(X: Union[np.ndarray, pd.DataFrame],
        n_components: Optional[int] = None,
        scale: bool = True,
        center: bool = True,
        **kwargs) -> OrdinationResult:
    """
    Principal Component Analysis.

    Parameters
    ----------
    X:
        Data matrix (samples x variables).
    n_components:
        Number of components to keep.
    scale:
        Whether to scale variables (unit variance) prior to analysis.
    center:
        Whether to subtract column means prior to analysis.

    Returns
    -------
    OrdinationResult with PCA results.
    """
    pca_obj = PCA(n_components=n_components, scale=scale, center=center)
    return pca_obj.fit(X, **kwargs)
