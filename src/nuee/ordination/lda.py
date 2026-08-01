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


#: Solvers that expose the transform and the attributes this wrapper needs.
#: sklearn's ``"lsqr"`` implements neither ``transform`` nor ``scalings_``/
#: ``explained_variance_ratio_``, so it cannot back an ordination.
SUPPORTED_SOLVERS = ("svd", "eigen")


def _scale_loadings(loadings: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Put discriminant loadings on the same extent as the site scores.

    Returns the loadings scaled so their longest arrow reaches 80% of the
    largest score coordinate. The factor is a ratio of two quantities that
    both track the units of X, so the result is invariant to rescaling the
    input features.
    """
    loadings = np.asarray(loadings, dtype=float)
    if loadings.size == 0 or scores is None or np.asarray(scores).size == 0:
        return loadings

    max_loading = float(np.max(np.abs(loadings)))
    max_score = float(np.max(np.abs(np.asarray(scores, dtype=float))))
    if max_loading <= 0.0 or max_score <= 0.0:
        return loadings

    return loadings * (0.8 * max_score / max_loading)


class LDA(OrdinationMethod):
    """Linear Discriminant Analysis for ordination."""

    def __init__(self, n_components: Optional[int] = None,
                 solver: str = "svd"):
        if solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"solver must be one of {SUPPORTED_SOLVERS}, got {solver!r}. "
                "sklearn's 'lsqr' solver provides no transform and no "
                "discriminant scalings, so it cannot produce an ordination."
            )
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
        **kwargs
            Rejected. Accepting these silently would let a misspelled
            ``n_components`` pass unnoticed and leave the ordination at its
            default width.

        Returns
        -------
        OrdinationResult
            Result with site scores, species loadings, and group labels.

        Raises
        ------
        TypeError
            If any unexpected keyword argument is supplied.
        """
        if kwargs:
            raise TypeError(
                "lda() got unexpected keyword argument(s): "
                + ", ".join(repr(name) for name in sorted(kwargs))
            )

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

        raw_loadings = model.scalings_[:, :n_comp]
        # scalings_ carries the units of X, so its magnitude is arbitrary
        # relative to the scores: rescaling the features by 1000 would leave
        # the scores untouched while shrinking every arrow to nothing. Bring
        # the loadings onto the score extent, the same 80% convention the
        # constrained biplot arrows use.
        loadings = _scale_loadings(raw_loadings, scores)

        # sklearn normalises explained_variance_ratio_ over *all* discriminant
        # axes and only then truncates to n_components, so these values are
        # already shares of the whole between-group variance and their total
        # is 1.0. Recording that total keeps proportion_explained honest when
        # n_components truncates the set -- left to itself it renormalises
        # over the retained axes alone, and a single retained axis then
        # reports 100%.
        explained_variance_ratio = np.asarray(
            model.explained_variance_ratio_, dtype=float)[:n_comp]

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
        result.tot_chi = 1.0

        if site_names is not None:
            result._site_names = site_names
        if species_names is not None:
            result._species_names = species_names

        result.groups = y_arr
        result.model = model
        result.explained_variance_ratio_ = explained_variance_ratio
        #: Unscaled discriminant coefficients, in the units of X.
        result.raw_loadings = raw_loadings

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
    solver : {"svd", "eigen"}
        Solver for sklearn LDA. Default ``"svd"``. sklearn's ``"lsqr"`` is
        rejected: it provides neither a transform nor discriminant scalings,
        so it cannot produce an ordination.

    Returns
    -------
    OrdinationResult
        ``points`` holds the discriminant scores and ``species`` the loadings,
        rescaled onto the score extent so biplot arrows stay visible whatever
        the units of *X* (the unscaled coefficients remain on
        ``result.raw_loadings``). ``proportion_explained`` gives each retained
        axis' share of the between-group variance across *all* axes, so it
        sums to less than 1 when ``n_components`` truncates the set.
    """
    lda_obj = LDA(n_components=n_components, solver=solver)
    return lda_obj.fit(X, y, **kwargs)
