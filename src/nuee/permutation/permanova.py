"""
PERMANOVA (Permutational Multivariate Analysis of Variance).

This module provides a basic implementation of distance-based PERMANOVA
that mirrors the behaviour of ``vegan::adonis2`` for sequential (type I)
tests.  The routine accepts a distance matrix together with one or more
predictor terms supplied as columns in a ``DataFrame`` or array.  The
current implementation focuses on the common use case where terms are
entered in the order they should be evaluated; interactions and strata
are not yet supported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import qr
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Sequence, Tuple, Union


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _as_numpy_distance(matrix: ArrayLike,
                       method: str = "bray") -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = squareform(matrix)
    if matrix.ndim != 2:
        raise ValueError("Distance input must be 1-D condensed, 2-D square, or a data matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        from ..dissimilarity.distances import vegdist
        return vegdist(matrix, method=method)
    if matrix.shape[0] < 2:
        raise ValueError("Distance matrix must contain at least two observations.")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric.")
    return matrix


def _prepare_terms(design: ArrayLike) -> List[Tuple[str, np.ndarray]]:
    if isinstance(design, pd.Series):
        design = design.to_frame()
    elif not isinstance(design, pd.DataFrame):
        design = pd.DataFrame(design)
    terms: List[Tuple[str, np.ndarray]] = []
    for column in design.columns:
        values = design[column]
        if values.dtype.kind in ("U", "S", "O") or str(values.dtype).startswith("category"):
            dummies = pd.get_dummies(values, drop_first=True).astype(float)
            if dummies.shape[1] == 0:
                continue
            terms.append((str(column), dummies.values))
        else:
            col = np.asarray(values, dtype=float)
            if np.all(np.isfinite(col)):
                terms.append((str(column), col[:, None]))
    return terms


def _orthonormal_basis(X: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, int]:
    if X.size == 0:
        return np.empty((X.shape[0], 0)), 0
    Q, R = qr(X, mode="reduced")
    if Q.size == 0:
        return Q, 0
    diag = np.abs(np.diag(R))
    keep = diag > tol
    Q = Q[:, keep]
    return Q, Q.shape[1]


def _sequential_components(G: np.ndarray,
                           terms: List[Tuple[str, np.ndarray]],
                           tol: float = 1e-10) -> Tuple[List[Dict[str, Union[str, float, int, np.ndarray]]],
                                                         float, int, float]:
    n = G.shape[0]
    total_ss = float(np.trace(G))
    q_prev = np.empty((n, 0))
    term_results: List[Dict[str, Union[str, float, int, np.ndarray]]] = []
    accumulated_ss = 0.0
    accumulated_df = 0

    for name, block in terms:
        block = np.asarray(block, dtype=float)
        if block.ndim == 1:
            block = block[:, None]
        block = block - block.mean(axis=0, keepdims=True)
        if q_prev.size:
            block = block - q_prev @ (q_prev.T @ block)
        q_j, rank_j = _orthonormal_basis(block, tol=tol)
        if rank_j > 0:
            ss_j = float(np.trace(q_j.T @ G @ q_j))
            q_prev = np.hstack([q_prev, q_j]) if q_prev.size else q_j
        else:
            ss_j = 0.0
        term_results.append({"name": name, "ss": ss_j, "df": rank_j, "basis": q_j})
        accumulated_ss += ss_j
        accumulated_df += rank_j

    residual_df = max(n - 1 - accumulated_df, 0)
    residual_ss = max(total_ss - accumulated_ss, 0.0)
    return term_results, residual_ss, residual_df, total_ss


def permanova(distance_matrix: ArrayLike,
              factors: ArrayLike,
              permutations: int = 999,
              random_state: Optional[Union[int, np.random.Generator]] = None,
              distance_method: str = "bray",
              **kwargs) -> pd.DataFrame:
    """
    Distance-based PERMANOVA (sequential sums of squares).

    Parameters
    ----------
    distance_matrix:
        Square or condensed distance matrix.
    factors:
        DataFrame, Series, or array of predictor variables. Each column is
        treated as a separate term evaluated sequentially.
    permutations:
        Number of permutations for significance testing. Set to ``0`` or ``None``
        to skip permutation p-values.
    random_state:
        Seed or ``Generator`` for reproducible permutations.

    Returns
    -------
    dict
        Dictionary containing a result table (``table``), the total sum of
        squares, and the permutation F-statistics.
    """
    D = _as_numpy_distance(distance_matrix, method=distance_method)
    terms = _prepare_terms(factors)
    if not terms:
        raise ValueError("No valid predictor terms supplied to PERMANOVA.")

    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ (D ** 2) @ H

    components, residual_ss, residual_df, total_ss = _sequential_components(G, terms)
    residual_f = residual_ss / residual_df if residual_df > 0 else np.nan

    observed_F = []
    rows = []
    sum_ss = 0.0
    sum_df = 0
    for comp in components:
        df = comp["df"]
        ss = comp["ss"]
        r2 = ss / total_ss if total_ss > 0 else np.nan
        if df > 0 and residual_df > 0 and residual_f > 0:
            f_stat = (ss / df) / residual_f
        else:
            f_stat = np.nan
        observed_F.append(f_stat)
        rows.append({
            "Df": df,
            "SumOfSqs": ss,
            "R2": r2,
            "F": f_stat,
            "Pr(>F)": np.nan,  # to be filled after permutations
        })
        sum_ss += ss
        sum_df += df

    rows.append({
        "Df": residual_df,
        "SumOfSqs": residual_ss,
        "R2": residual_ss / total_ss if total_ss > 0 else np.nan,
        "F": np.nan,
        "Pr(>F)": np.nan,
    })
    rows.append({
        "Df": n - 1,
        "SumOfSqs": total_ss,
        "R2": 1.0,
        "F": np.nan,
        "Pr(>F)": np.nan,
    })

    if permutations and permutations > 0 and residual_df > 0:
        rng = np.random.default_rng(random_state)
        term_bases = [comp["basis"] for comp in components]
        perm_counts = np.zeros((len(components),), dtype=int)
        for _ in range(permutations):
            perm = rng.permutation(n)
            G_perm = G[perm][:, perm]
            total_perm = float(np.trace(G_perm))
            ss_terms = []
            for basis in term_bases:
                if basis is None or basis.size == 0:
                    ss_terms.append(0.0)
                else:
                    ss_terms.append(float(np.trace(basis.T @ G_perm @ basis)))
            ss_terms = np.array(ss_terms)
            res_ss_perm = max(total_perm - ss_terms.sum(), 0.0)
            res_f_perm = res_ss_perm / residual_df if residual_df > 0 else np.nan
            for idx, (ss_j, df_j) in enumerate(zip(ss_terms, [comp["df"] for comp in components])):
                obs_F = observed_F[idx]
                if df_j > 0 and np.isfinite(res_f_perm) and res_f_perm > 0 and np.isfinite(obs_F):
                    f_perm = (ss_j / df_j) / res_f_perm
                    if np.isfinite(f_perm) and f_perm >= obs_F - 1e-12:
                        perm_counts[idx] += 1
        for idx, row in enumerate(rows[:len(components)]):
            if observed_F[idx] is not None and np.isfinite(observed_F[idx]) and permutations > 0:
                row["Pr(>F)"] = (perm_counts[idx] + 1) / (permutations + 1)

    table = pd.DataFrame(rows,
                         index=[comp["name"] for comp in components] + ["Residual", "Total"],
                         columns=["Df", "SumOfSqs", "R2", "F", "Pr(>F)"])
    table.attrs.update({
        "total_ss": total_ss,
        "residual_ss": residual_ss,
        "permutations": permutations,
        "distance_method": distance_method,
    })
    return table


def adonis2(distance_matrix: ArrayLike,
            factors: ArrayLike,
            permutations: int = 999,
            random_state: Optional[Union[int, np.random.Generator]] = None,
            distance_method: str = "bray",
            **kwargs) -> pd.DataFrame:
    """
    Convenience wrapper for distance-based PERMANOVA.

    Parameters
    ----------
    distance_matrix:
        Square or condensed distance matrix.
    factors:
        Predictor variables supplied as a ``DataFrame``/``Series``/array.
    permutations:
        Number of permutations for the significance test.
    random_state:
        Seed or ``Generator`` controlling permutation reproducibility.

    Returns
    -------
    dict
        Same payload as :func:`permanova`.
    """
    return permanova(distance_matrix,
                     factors,
                     permutations=permutations,
                     random_state=random_state,
                     distance_method=distance_method,
                     **kwargs)
