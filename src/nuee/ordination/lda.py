"""
Linear Discriminant Analysis (LDA) for ordination.

LDA finds axes that maximise between-group separation relative to
within-group variance.  It wraps scikit-learn's
``LinearDiscriminantAnalysis`` and returns an ``OrdinationResult``
compatible with ``biplot()``.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base import OrdinationResult, OrdinationMethod


class LDA(OrdinationMethod):
    """Linear Discriminant Analysis for ordination."""

    def __init__(self, n_components: Optional[int] = None,
                 solver: str = "svd"):
        self.n_components = n_components
        self.solver = solver

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> OrdinationResult:
        """Fit LDA to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix (samples x variables).
        y : array-like of shape (n_samples,)
            Group labels for each sample.

        Returns
        -------
        OrdinationResult
            Result with site scores, species loadings, and group labels.
        """
        species_names = None
        site_names = None
        if isinstance(X, pd.DataFrame):
            species_names = list(X.columns)
            site_names = list(X.index)

        X_arr = self._validate_data(X)

        # Coerce y to 1-D numpy (handles Polars, pandas, lists, etc.)
        if hasattr(y, "to_series"):    # pl.DataFrame → first column
            y = y.to_series()
        if hasattr(y, "to_numpy"):     # pl.Series / pd.Series
            y_arr = y.to_numpy()
        else:
            y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        n_classes = len(np.unique(y_arr))
        max_components = min(n_classes - 1, X_arr.shape[1])

        n_comp = self.n_components
        if n_comp is None:
            n_comp = max_components
        else:
            n_comp = min(n_comp, max_components)

        model = LinearDiscriminantAnalysis(
            n_components=n_comp, solver=self.solver
        )
        scores = model.fit_transform(X_arr, y_arr)

        loadings = model.scalings_[:, :n_comp]
        explained_variance_ratio = model.explained_variance_ratio_[:n_comp]

        call_info = {
            "method": "LDA",
            "n_components": n_comp,
            "solver": self.solver,
            "n_classes": n_classes,
        }

        result = OrdinationResult(
            points=scores,
            species=loadings,
            eigenvalues=explained_variance_ratio,
            call=call_info,
        )

        if site_names is not None:
            result._site_names = site_names
        if species_names is not None:
            result._species_names = species_names

        result.groups = y_arr
        result.model = model
        result.explained_variance_ratio_ = explained_variance_ratio

        return result


def lda(X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_components: Optional[int] = None,
        solver: str = "svd",
        **kwargs) -> OrdinationResult:
    """
    Linear Discriminant Analysis.

    Parameters
    ----------
    X : array-like
        Data matrix (samples x variables).
    y : array-like
        Group labels for each sample.
    n_components : int, optional
        Number of discriminant axes to keep.
        Defaults to min(n_classes - 1, n_features).
    solver : str
        Solver for sklearn LDA. Default ``"svd"``.

    Returns
    -------
    OrdinationResult with LDA results.
    """
    lda_obj = LDA(n_components=n_components, solver=solver)
    return lda_obj.fit(X, y, **kwargs)
