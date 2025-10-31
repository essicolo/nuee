"""
Beta dispersion (multivariate homogeneity of group dispersions).

Implements the procedure used in ``vegan::betadisper``: project the
distance matrix into principal coordinates, compute distances of samples
to their group centroids, and perform an ANOVA-style test with optional
permutation p-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from typing import Dict, Optional, Tuple, Union


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _as_numpy_distance(matrix: ArrayLike, method: str = "bray") -> np.ndarray:
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
        raise ValueError("Grouping vector length must match distance matrix.")
    return grouping


def _pcoa_embedding(D: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    min_eval = evals.min()
    if min_eval < -tol:
        c = -min_eval
        D2 = D2 + 2 * c
        B = -0.5 * J @ D2 @ J
        evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    mask = np.abs(evals) > tol
    if not np.any(mask):
        return np.zeros((n, 0), dtype=float), np.empty((0,), dtype=float)
    evals = evals[mask]
    evecs = evecs[:, mask]
    coords = evecs * np.sqrt(np.abs(evals))
    for j in range(coords.shape[1]):
        col = coords[:, j]
        if col.size and np.abs(col).max() > 0 and col[np.abs(col).argmax()] < 0:
            coords[:, j] *= -1
    signs = np.sign(evals)
    return coords, signs


def betadisper(distance_matrix: ArrayLike,
               grouping: ArrayLike,
               permutations: int = 0,
               random_state: Optional[Union[int, np.random.Generator]] = None,
               distance_method: str = "bray",
               **kwargs) -> Dict[str, Union[float, int, np.ndarray, pd.DataFrame]]:
    """
    Multivariate homogeneity of group dispersions.

    Parameters
    ----------
    distance_matrix:
        Square or condensed distance matrix, or original data matrix which will
        be converted via ``vegdist``.
    grouping:
        Group assignments for each observation.
    permutations:
        Number of permutations for the ANOVA-like dispersion test.
    random_state:
        Seed controlling permutation reproducibility.
    distance_method:
        Distance metric when converting from raw data (default ``bray``).

    Returns
    -------
    dict
        - ``distances``: Series of distances to group centroids.
        - ``centroids``: DataFrame of group centroids in PCoA space.
        - ``f_statistic`` / ``p_value``: results of the dispersion test.
        - ``group_means``: within-group mean distances.
        - ``permutations``: number of permutations performed.
    """
    D = _as_numpy_distance(distance_matrix, method=distance_method)
    n = D.shape[0]
    grouping = _prepare_grouping(grouping, n)

    coords, signs = _pcoa_embedding(D)
    if coords.size == 0:
        raise ValueError("Distance matrix does not yield positive eigenvalues for PCoA.")

    label_list = []
    index_map = {}
    inverse = np.empty_like(grouping, dtype=int)
    for idx, label in enumerate(grouping):
        if label not in index_map:
            index_map[label] = len(label_list)
            label_list.append(label)
        inverse[idx] = index_map[label]
    labels = np.asarray(label_list)
    sizes = np.bincount(inverse)
    if np.any(sizes < 2):
        raise ValueError("Each group must contain at least two observations.")

    centroids = np.zeros((len(labels), coords.shape[1]), dtype=float)
    for idx, label in enumerate(labels):
        members = coords[inverse == idx]
        centroids[idx] = members.mean(axis=0)

    diff = coords - centroids[inverse]
    sq_dist = np.sum(signs * diff**2, axis=1)
    sq_dist = np.clip(sq_dist, 0, None)
    distances = np.sqrt(sq_dist)
    group_means = np.array([distances[inverse == idx].mean() for idx in range(len(labels))])

    total_mean = distances.mean()
    ss_between = ((group_means - total_mean) ** 2 * sizes).sum()
    df_between = len(labels) - 1
    ss_within = sum(((distances[inverse == idx] - group_means[idx]) ** 2).sum()
                    for idx in range(len(labels)))
    df_within = n - len(labels)

    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    F = ms_between / ms_within if np.isfinite(ms_between) and np.isfinite(ms_within) and ms_within > 0 else np.nan

    p_value = np.nan
    if permutations and permutations > 0 and np.isfinite(F):
        rng = np.random.default_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = rng.permutation(inverse)
            perm_centroids = np.zeros_like(centroids)
            for idx in range(len(labels)):
                members = coords[perm == idx]
                if members.size == 0:
                    continue
                perm_centroids[idx] = members.mean(axis=0)
            diff_perm = coords - perm_centroids[perm]
            sq_perm = np.sum(signs * diff_perm**2, axis=1)
            sq_perm = np.clip(sq_perm, 0, None)
            dist_perm = np.sqrt(sq_perm)
            perm_means = np.array([dist_perm[perm == idx].mean() for idx in range(len(labels))])
            ssb = ((perm_means - perm_means.mean()) ** 2 * sizes).sum()
            msb = ssb / df_between if df_between > 0 else np.nan
            if np.isfinite(msb) and ms_within > 0:
                F_perm = msb / ms_within
                if np.isfinite(F_perm) and F_perm >= F - 1e-12:
                    exceed += 1
        p_value = (exceed + 1) / (permutations + 1)

    col_names = [f"PCoA{i+1}" for i in range(coords.shape[1])]
    centroid_df = pd.DataFrame(centroids, index=labels, columns=col_names)
    distance_series = pd.Series(distances, index=np.arange(n), name="distance")
    group_series = pd.Series(group_means, index=labels, name="mean_distance")

    return {
        "distances": distance_series,
        "centroids": centroid_df,
        "group_means": group_series,
        "f_statistic": float(F) if np.isfinite(F) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "permutations": permutations,
    }
