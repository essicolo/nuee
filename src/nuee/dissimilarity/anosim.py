"""
ANOSIM (Analysis of Similarities).

This module provides an implementation of ANOSIM that follows the
approach used in ``vegan::anosim``: it ranks distances, computes the
between- vs. within-group statistic, and estimates p-values by permutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from typing import Dict, Optional, Union


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _as_numpy_distance(matrix: ArrayLike) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = squareform(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square or condensed.")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric.")
    return matrix


def _prepare_grouping(grouping: ArrayLike, n: int) -> np.ndarray:
    if isinstance(grouping, pd.Series):
        grouping = grouping.values
    grouping = np.asarray(grouping)
    if grouping.ndim != 1:
        raise ValueError("Grouping vector must be one-dimensional.")
    if grouping.shape[0] != n:
        raise ValueError("Grouping vector length must match the distance matrix.")
    return grouping


def anosim(distance_matrix: ArrayLike,
           grouping: ArrayLike,
           permutations: int = 999,
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs) -> Dict[str, Union[float, int]]:
    """
    Analysis of similarities (ANOSIM).

    Parameters
    ----------
    distance_matrix:
        Square or condensed distance matrix.
    grouping:
        Group assignments for each observation.
    permutations:
        Number of permutations used to estimate the p-value. Set to 0 to skip.
    random_state:
        Seed controlling permutation reproducibility.

    Returns
    -------
    dict
        Dictionary with ``r_statistic``, ``p_value`` and ``permutations``.
    """
    D = _as_numpy_distance(distance_matrix)
    n = D.shape[0]
    grouping = _prepare_grouping(grouping, n)

    iu = np.triu_indices(n, k=1)
    distances = D[iu]
    ranks = distances.argsort().argsort().astype(float) + 1.0

    group_labels, inverse = np.unique(grouping, return_inverse=True)
    n_groups = len(group_labels)
    if n_groups < 2:
        raise ValueError("ANOSIM requires at least two groups.")

    between_values = []
    within_values = []
    idx = 0
    for i in range(n - 1):
        gi = inverse[i]
        for j in range(i + 1, n):
            gj = inverse[j]
            if gi == gj:
                within_values.append(ranks[idx])
            else:
                between_values.append(ranks[idx])
            idx += 1

    between_values = np.asarray(between_values, dtype=float)
    within_values = np.asarray(within_values, dtype=float)
    rb = between_values.mean() if between_values.size else 0.0
    rw = within_values.mean() if within_values.size else 0.0
    m = ranks.size
    denominator = m / 2 if m > 0 else 1.0
    R_obs = (rb - rw) / denominator

    p_value = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            permuted = rng.permutation(inverse)
            between_p = []
            within_p = []
            idx = 0
            for i in range(n - 1):
                gi = permuted[i]
                for j in range(i + 1, n):
                    gj = permuted[j]
                    if gi == gj:
                        within_p.append(ranks[idx])
                    else:
                        between_p.append(ranks[idx])
                    idx += 1
            between_p = np.asarray(between_p, dtype=float)
            within_p = np.asarray(within_p, dtype=float)
            denom = m / 2 if m > 0 else 1.0
            rb_p = between_p.mean() if between_p.size else 0.0
            rw_p = within_p.mean() if within_p.size else 0.0
            R_perm = (rb_p - rw_p) / denom
            if R_perm >= R_obs - 1e-12:
                exceed += 1
        p_value = (exceed + 1) / (permutations + 1)

    return {
        "r_statistic": float(R_obs),
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "permutations": permutations,
    }
