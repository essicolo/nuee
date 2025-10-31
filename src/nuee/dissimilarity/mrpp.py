"""
MRPP (Multi-Response Permutation Procedure).

Implementation following the approach of ``vegan::mrpp``. The routine
computes the observed weighted mean within-group distance, its expected
value under permutations, the chance-corrected agreement statistic ``A``,
and a permutation-based p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from typing import Dict, Optional, Union


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _as_numpy_distance(matrix: ArrayLike, method: str = "euclidean") -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = squareform(matrix)
    if matrix.ndim != 2:
        raise ValueError("Distance matrix must be 1-D condensed, 2-D square, or a data matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        from .distances import vegdist
        return vegdist(matrix, method=method)
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


def mrpp(distance_matrix: ArrayLike,
         grouping: ArrayLike,
         permutations: int = 999,
         random_state: Optional[Union[int, np.random.Generator]] = None,
         distance_method: str = "euclidean",
         **kwargs) -> Dict[str, Union[float, int]]:
    """
    Multi-Response Permutation Procedure (MRPP).

    Parameters
    ----------
    distance_matrix:
        Square or condensed distance matrix.
    grouping:
        Group assignments for each observation.
    permutations:
        Number of permutations for p-value estimation. Set to 0 to skip permutations.
    random_state:
        Seed controlling permutation reproducibility.

    Returns
    -------
    dict
        Contains ``delta`` (observed within-group distance),
        ``expected_delta`` (mean delta under permutations), ``a_statistic``
        (chance-corrected agreement), ``p_value``, and ``permutations``.
    """
    D = _as_numpy_distance(distance_matrix, method=distance_method)
    n = D.shape[0]
    grouping = _prepare_grouping(grouping, n)

    iu = np.triu_indices(n, k=1)
    distances = D[iu]
    perm_indices = list(zip(iu[0], iu[1]))

    labels, inverse = np.unique(grouping, return_inverse=True)
    sizes = np.bincount(inverse)
    n_groups = len(labels)
    if n_groups < 2:
        raise ValueError("MRPP requires at least two groups.")

    pair_counts = (sizes * (sizes - 1) / 2).astype(float)
    weights = sizes.astype(float)
    weight_sum = weights.sum()
    with np.errstate(invalid="ignore"):
        weights = np.where(pair_counts > 0, weights, 0.0)

    within_sums = np.zeros(n_groups, dtype=float)
    for idx, (i, j) in enumerate(perm_indices):
        gi, gj = inverse[i], inverse[j]
        if gi == gj:
            within_sums[gi] += distances[idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        deltas = np.divide(within_sums,
                            pair_counts,
                            out=np.zeros_like(within_sums),
                            where=pair_counts > 0)
    delta = float(np.sum(deltas * weights) / weight_sum)

    total_mean = distances.mean() if distances.size else 0.0
    expected_delta = total_mean
    a_statistic = 1.0 - (delta / expected_delta) if expected_delta > 0 else 0.0

    p_value = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = rng.permutation(inverse)
            within_perm = np.zeros(n_groups, dtype=float)
            for idx, (i, j) in enumerate(perm_indices):
                gi, gj = perm[i], perm[j]
                if gi == gj:
                    within_perm[gi] += distances[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                deltas_perm = np.divide(within_perm,
                                        pair_counts,
                                        out=np.zeros_like(within_perm),
                                        where=pair_counts > 0)
            delta_perm = float(np.sum(deltas_perm * weights) / weight_sum)
            if delta_perm <= delta + 1e-12:
                exceed += 1
        p_value = (exceed + 1) / (permutations + 1)

    return {
        "delta": float(delta),
        "expected_delta": float(expected_delta),
        "a_statistic": float(a_statistic),
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "permutations": permutations,
    }
