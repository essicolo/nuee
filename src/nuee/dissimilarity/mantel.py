"""
Mantel test implementation (Mantel, 1967).

Supports Pearson and Spearman correlations with permutation p-values,
mirroring ``vegan::mantel``. A partial Mantel test is also provided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from typing import Dict, Optional, Tuple, Union


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
GeneratorType = getattr(np.random, "Generator", None)


def _as_numpy_distance(matrix: ArrayLike) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = squareform(matrix)
    if matrix.ndim != 2:
        raise ValueError("Distance matrix must be 1-D condensed or 2-D square.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric.")
    return matrix


def _flatten(matrix: np.ndarray, method: str) -> np.ndarray:
    vec = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if method == "spearman":
        vec = pd.Series(vec).rank().to_numpy()
    return vec


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    r, _ = pearsonr(x, y)
    return float(r)


def _normalize_rng(random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]) -> Optional[Union[np.random.RandomState, np.random.Generator]]:
    if random_state is None:
        return None
    if GeneratorType is not None and isinstance(random_state, GeneratorType):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def _uniform_drawer(rng: Optional[Union[np.random.RandomState, np.random.Generator]]):
    if rng is None:
        return np.random.random
    if GeneratorType is not None and isinstance(rng, GeneratorType):
        return rng.random
    return rng.random_sample


def _draw_permutation(n: int,
                      rng: Optional[Union[np.random.RandomState, np.random.Generator]]) -> np.ndarray:
    drawer = _uniform_drawer(rng)
    perm = np.arange(n)
    for idx in range(n - 1, 0, -1):
        u = float(drawer())
        j = int(u * (idx + 1))
        if j > idx:
            j = idx
        perm[idx], perm[j] = perm[j], perm[idx]
    return perm


def mantel(x: ArrayLike,
           y: ArrayLike,
           method: str = "pearson",
           permutations: int = 999,
           random_state: Optional[Union[int, np.random.Generator]] = None,
            **kwargs) -> Dict[str, Union[float, int, str]]:
    method = method.lower()
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    X = _as_numpy_distance(x)
    Y = _as_numpy_distance(y)
    if X.shape != Y.shape:
        raise ValueError("Matrices must have the same shape.")

    x_vec = _flatten(X, method)
    y_vec = _flatten(Y, method)

    r = _correlation(x_vec, y_vec)

    p_value = np.nan
    if permutations and permutations > 0:
        rng = _normalize_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = _draw_permutation(X.shape[0], rng)
            Y_perm = Y[perm][:, perm]
            y_perm_vec = _flatten(Y_perm, method)
            r_perm = _correlation(x_vec, y_perm_vec)
            if r_perm >= r - 1e-12:
                exceed += 1
        p_value = (exceed + 1) / (permutations + 1)

    return {
        "r_statistic": r,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "permutations": permutations,
        "method": method,
    }


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float, float]:
    rxy = _correlation(x, y)
    rxz = _correlation(x, z)
    ryz = _correlation(y, z)
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        return np.nan, rxy, rxz, ryz
    return (rxy - rxz * ryz) / denom, rxy, rxz, ryz


def mantel_partial(x: ArrayLike,
                   y: ArrayLike,
                   z: ArrayLike,
                   method: str = "pearson",
                   permutations: int = 999,
                   random_state: Optional[Union[int, np.random.Generator]] = None,
                   **kwargs) -> Dict[str, Union[float, int, str]]:
    method = method.lower()
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    X = _as_numpy_distance(x)
    Y = _as_numpy_distance(y)
    Z = _as_numpy_distance(z)
    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("All matrices must have the same shape.")

    x_vec = _flatten(X, method)
    y_vec = _flatten(Y, method)
    z_vec = _flatten(Z, method)

    r, rxy, rxz, ryz = _partial_correlation(x_vec, y_vec, z_vec)

    p_value = np.nan
    if permutations and permutations > 0 and np.isfinite(r):
        rng = _normalize_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = _draw_permutation(X.shape[0], rng)
            X_perm = X[perm][:, perm]
            x_perm_vec = _flatten(X_perm, method)
            r_perm, *_ = _partial_correlation(x_perm_vec, y_vec, z_vec)
            if np.isfinite(r_perm) and r_perm >= r - 1e-12:
                exceed += 1
        p_value = (exceed + 1) / (permutations + 1)

    return {
        "r_statistic": r,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "permutations": permutations,
        "method": method,
    }
