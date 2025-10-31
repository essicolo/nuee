"""
Protest (Procrustean randomization test) for comparing ordination configurations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Union

from ..ordination.procrustes import procrustes
from .mantel import _draw_permutation, _normalize_rng  # reuse internal helpers


ArrayLike = Union[np.ndarray, pd.DataFrame]


def _as_matrix(values: ArrayLike) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values.values
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Input configuration must be a 2-D array.")
    return array


def protest(x: ArrayLike,
            y: ArrayLike,
            permutations: int = 999,
            *,
            random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
            scale: bool = True) -> Dict[str, Union[float, int, np.ndarray]]:
    """
    Perform a Procrustean randomization test (PROTEST) between two ordinations.

    Parameters
    ----------
    x, y:
        Configuration matrices with matched observations (rows) and axes (columns).
    permutations:
        Number of row permutations applied to ``y`` to approximate the null
        distribution. Set to ``0`` to skip permutation testing.
    random_state:
        Seed or numpy RNG used for the permutation stream. When ``None`` the
        global numpy RNG is used.
    scale:
        Forwarded to :func:`nuee.procrustes` to control symmetric scaling.

    Returns
    -------
    dict
        Dictionary capturing the observed correlation, permutation p-value,
        number of permutations, and Procrustes transformation details.
    """
    X = _as_matrix(x)
    Y = _as_matrix(y)

    if X.shape != Y.shape:
        raise ValueError(f"Configurations must share the same shape; received {X.shape} vs {Y.shape}.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        observed = procrustes(X, Y, scale=scale)
    r_obs = float(observed["correlation"])

    p_value = np.nan
    if permutations and permutations > 0:
        rng = _normalize_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = _draw_permutation(Y.shape[0], rng)
            Y_perm = Y[perm, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                permuted = procrustes(X, Y_perm, scale=scale)
            if float(permuted["correlation"]) >= r_obs - 1e-12:
                exceed += 1
        p_value = (exceed + 1) / (permutations + 1)

    return {
        "correlation": r_obs,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "permutations": permutations,
        "rotation": observed["rotation"],
        "ss": observed["ss"],
        "dilation": observed["dilation"],
        "translation": observed["translation"],
        "scale": scale,
    }
