"""
Imputation methods for compositional data.

Provides:
- ``replace_zeros``: multiplicative zero replacement with optional
  per-component detection limits (Martín-Fernández et al. 2003).
- ``impute_missing``: log-ratio EM (lrEM) algorithm for missing data
  in compositions (Palarea-Albaladejo & Martín-Fernández 2008).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Optional


def _get_composition_funcs():
    """Late import to avoid circular dependency with __init__."""
    from . import closure, ilr, ilr_inv, _default_basis
    return closure, ilr, ilr_inv, _default_basis


def _sweep_matrix(A, ind):
    """Beaton sweep operator matching R's zCompositions implementation."""
    A = A.copy()
    D = A.shape[0]
    S = A.copy()
    for j in ind:
        S[j, j] = -1.0 / A[j, j]
        for i in range(D):
            if i != j:
                S[i, j] = -A[i, j] * S[j, j]
                S[j, i] = S[i, j]
        for i in range(D):
            if i != j:
                for k in range(D):
                    if k != j:
                        S[i, k] = A[i, k] - S[i, j] * A[j, k]
                        S[k, i] = S[i, k]
        A = S.copy()
    return A


def _lm_sweep(M, C, varobs):
    """Regression via sweep on augmented mean-covariance matrix (matches R).

    Parameters
    ----------
    M : ndarray, shape (D,)
        Mean vector.
    C : ndarray, shape (D, D)
        Covariance matrix.
    varobs : ndarray of int
        0-indexed indices of observed variables.

    Returns
    -------
    B : ndarray, shape (len(varobs)+1, len(dep))
        Regression coefficients. B[0] is the intercept row, B[1:] are slopes.
    CR : ndarray, shape (len(dep), len(dep))
        Conditional (residual) covariance of missing given observed.
    """
    D = len(M)
    q = len(varobs)
    i_flag = np.ones(D, dtype=int)
    for v in varobs:
        i_flag[v] = 0
    dep = np.where(i_flag != 0)[0]

    # Build augmented matrix (D+1) x (D+1)
    A = np.zeros((D + 1, D + 1))
    A[0, 0] = -1.0
    A[0, 1:D + 1] = M
    A[1:D + 1, 0] = M
    A[1:D + 1, 1:D + 1] = C

    # Reorder: [intercept, observed+1, missing+1]
    reor = np.concatenate([[0], varobs + 1, dep + 1]).astype(int)
    A = A[np.ix_(reor, reor)]

    # Sweep on intercept + observed variable indices
    A = _sweep_matrix(A, list(range(q + 1)))

    B = A[0:q + 1, q + 1:D + 1]
    CR = A[q + 1:D + 1, q + 1:D + 1]
    return B, CR


def replace_zeros(
    X: Union[np.ndarray, pd.DataFrame],
    detection_limits: Optional[np.ndarray] = None,
    delta: Optional[float] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Multiplicative zero replacement for compositional data.

    Replaces zeros with a small value proportional to the detection limit
    (or column minimum of non-zero values) and adjusts non-zero entries so
    that each row sum is preserved exactly.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, D)
        Compositional data matrix.  Zeros mark below-detection-limit values.
    detection_limits : array-like of shape (D,), optional
        Per-component detection limits.  When *None*, the column-wise minimum
        of strictly positive values is used as a proxy.
    delta : float, optional
        Fraction of the detection limit used as the replacement value.
        Default is 0.65 (Martín-Fernández et al. 2003).

    Returns
    -------
    numpy.ndarray or DataFrame
        Data with zeros replaced.  Row sums match the input exactly.

    References
    ----------
    .. [1] Martín-Fernández, J. A., Barceló-Vidal, C. & Pawlowsky-Glahn, V.
           (2003). Dealing with zeros and missing values in compositional data
           sets using nonparametric imputation. *Mathematical Geology* 35(3).
    """
    is_df = isinstance(X, pd.DataFrame)
    if is_df:
        index, columns = X.index, X.columns
        mat = X.values.astype(float)
    else:
        mat = np.asarray(X, dtype=float)

    if mat.ndim != 2:
        raise ValueError("X must be a 2-D array or DataFrame")
    if np.any(mat < 0):
        raise ValueError("X must not contain negative values")

    if delta is None:
        delta = 0.65

    n, D = mat.shape
    is_zero = mat == 0

    # Determine detection limits
    if detection_limits is not None:
        dl = np.asarray(detection_limits, dtype=float).ravel()
        if dl.shape[0] != D:
            raise ValueError(
                f"detection_limits has {dl.shape[0]} elements, expected {D}"
            )
    else:
        # Use column minimum of non-zero values as proxy
        dl = np.full(D, np.nan)
        for j in range(D):
            col_pos = mat[:, j][mat[:, j] > 0]
            if col_pos.size > 0:
                dl[j] = col_pos.min()
            else:
                dl[j] = 1.0  # fallback for all-zero columns
    if np.any(np.isnan(dl)):
        raise ValueError("Could not determine detection limits for all components")

    replacement_values = delta * dl  # shape (D,)

    result = mat.copy()
    row_sums = mat.sum(axis=1)

    for i in range(n):
        zero_mask = is_zero[i]
        if not np.any(zero_mask):
            continue
        if np.all(zero_mask):
            raise ValueError(f"Row {i} is entirely zeros; cannot replace")

        r_sum = row_sums[i]
        if r_sum == 0:
            raise ValueError(f"Row {i} sums to zero; cannot replace")

        # Sum of replacement values in this row
        repl_total = replacement_values[zero_mask].sum()
        if repl_total >= r_sum:
            raise ValueError(
                f"Row {i}: sum of replacement values ({repl_total:.4g}) "
                f"exceeds the row sum ({r_sum:.4g}).  "
                "Consider smaller delta or detection limits."
            )

        # Replace zeros
        result[i, zero_mask] = replacement_values[zero_mask]
        # Adjust non-zero entries to preserve row sum
        adjust = (r_sum - repl_total) / r_sum
        result[i, ~zero_mask] = mat[i, ~zero_mask] * adjust

    if is_df:
        return pd.DataFrame(result, index=index, columns=columns)
    return result


def impute_missing(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = "lrEM",
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Impute missing values in compositional data using the lrEM algorithm.

    Uses the ALR (additive log-ratio) EM algorithm of Palarea-Albaladejo &
    Martín-Fernández (2008), matching the approach in R's ``zCompositions``
    package.  Observed values are preserved exactly in the output.

    At least one column must be fully observed (no ``NaN`` values) to serve
    as the ALR denominator.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, D)
        Compositional data matrix.  ``NaN`` marks missing components.
        Observed (non-NaN) values must be strictly positive.
    method : {"lrEM", "lrDA"}, default "lrEM"
        ``"lrEM"`` returns the conditional expectation (deterministic).
        ``"lrDA"`` adds noise from the conditional covariance for multiple
        imputation / data augmentation.
    max_iter : int, default 100
        Maximum number of EM iterations.
    tol : float, default 1e-4
        Convergence tolerance on the relative change of the log-likelihood.
    random_state : int, optional
        Seed for the random number generator (only used when *method="lrDA"*).

    Returns
    -------
    numpy.ndarray or DataFrame
        Completed data.  Observed values are unchanged; imputed values are
        scaled consistently with the original observed components.

    References
    ----------
    .. [1] Palarea-Albaladejo, J. & Martín-Fernández, J. A. (2008).
           A modified EM alr-algorithm for replacing rounded zeros in
           compositional data sets. *Computers & Geosciences* 34(8).
    """
    if method not in ("lrEM", "lrDA"):
        raise ValueError(f"method must be 'lrEM' or 'lrDA', got '{method}'")

    is_df = isinstance(X, pd.DataFrame)
    if is_df:
        index, columns = X.index, X.columns
        mat = X.values.astype(float)
    else:
        mat = np.asarray(X, dtype=float).copy()

    if mat.ndim != 2:
        raise ValueError("X must be a 2-D array or DataFrame")

    n, D = mat.shape
    if D < 2:
        raise ValueError("Need at least 2 components for compositional imputation")

    rng = np.random.default_rng(random_state)

    # Identify missingness
    nan_mask = np.isnan(mat)
    has_missing = nan_mask.any(axis=1)
    all_missing = nan_mask.all(axis=1)

    if np.any(all_missing):
        raise ValueError("Some rows are entirely NaN; cannot impute")

    # Check observed values are positive
    observed_vals = mat[~nan_mask]
    if np.any(observed_vals < 0):
        raise ValueError("Observed (non-NaN) values must be non-negative")
    if np.any(observed_vals[~np.isnan(observed_vals)] == 0):
        raise ValueError(
            "Observed values contain zeros. Use replace_zeros() first, "
            "then impute_missing() for NaN values."
        )

    # Find the first complete column (no NAs) as ALR denominator (matches R)
    col_na_count = nan_mask.sum(axis=0)
    complete_cols = np.where(col_na_count == 0)[0]
    if complete_cols.size == 0:
        raise ValueError(
            "lrEM requires at least one complete column (no NaN values) "
            "to use as the ALR denominator."
        )
    denom_idx = int(complete_cols[0])  # first complete column (like R)

    # Non-denominator column indices (these become the ALR coordinates)
    alr_cols = [j for j in range(D) if j != denom_idx]
    n_alr = len(alr_cols)  # D - 1

    # Compute ALR directly from original data (NAs become NAs in ALR space).
    # ALR coord k = log(X[:, alr_cols[k]] / X[:, denom_idx]).
    X_alr = np.full((n, n_alr), np.nan)
    for k, j in enumerate(alr_cols):
        obs_both = ~nan_mask[:, j]  # denom col is always observed
        X_alr[obs_both, k] = np.log(mat[obs_both, j] / mat[obs_both, denom_idx])

    # Group rows by missingness pattern in ALR space
    alr_nan = np.isnan(X_alr)
    miss_rows = np.where(alr_nan.any(axis=1))[0]
    patterns: dict[tuple, list[int]] = {}
    for i in miss_rows:
        key = tuple(alr_nan[i])
        patterns.setdefault(key, []).append(i)

    pattern_alr_info: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for pat in patterns:
        pat_arr = np.array(pat)
        obs_alr = np.where(~pat_arr)[0]
        miss_alr = np.where(pat_arr)[0]
        pattern_alr_info[pat] = (obs_alr, miss_alr)

    # Initial M: column-wise means ignoring NAs (matches R colMeans(na.rm=T))
    M = np.nanmean(X_alr, axis=0)

    # Initial C: covariance from complete rows (matches R cov(use="complete.obs"))
    complete_rows = np.where(~alr_nan.any(axis=1))[0]
    if complete_rows.size > n_alr:
        Z_complete = X_alr[complete_rows]
        mu_c = Z_complete.mean(axis=0)
        cent_c = Z_complete - mu_c
        C = (cent_c.T @ cent_c) / max(len(complete_rows) - 1, 1)
    else:
        # Fallback: use all available data with pairwise complete
        C = np.zeros((n_alr, n_alr))
        for a in range(n_alr):
            for b in range(a, n_alr):
                valid = ~(alr_nan[:, a] | alr_nan[:, b])
                if valid.sum() > 1:
                    va = X_alr[valid, a]
                    vb = X_alr[valid, b]
                    C[a, b] = C[b, a] = np.cov(va, vb)[0, 1]

    for iteration in range(max_iter):
        M_old = M.copy()
        C_old = C.copy()

        # E-step: fill missing ALR coords via regression (sweep operator)
        Y = X_alr.copy()  # start from original ALR (with NAs)
        v = np.zeros((n_alr, n_alr))  # variance correction accumulator

        for pat, row_indices in patterns.items():
            obs_alr, miss_alr = pattern_alr_info[pat]

            if miss_alr.size == 0:
                continue
            if obs_alr.size == 0:
                # All ALR coords missing — use marginal mean
                for i in row_indices:
                    Y[i, miss_alr] = M[miss_alr]
                    if method == "lrDA":
                        cov_mm = C[np.ix_(miss_alr, miss_alr)]
                        Y[i, miss_alr] += rng.multivariate_normal(
                            np.zeros(miss_alr.size), cov_mm
                        )
                cond_var = C[np.ix_(miss_alr, miss_alr)]
                block = np.zeros((n_alr, n_alr))
                block[np.ix_(miss_alr, miss_alr)] = cond_var
                v += block * len(row_indices)
                continue

            # Regression via sweep on augmented mean-covariance matrix
            B, CR = _lm_sweep(M, C, obs_alr)
            CR = np.atleast_2d(CR)
            CR = (CR + CR.T) / 2

            if method == "lrDA":
                cond_cov = CR.copy()
                eigvals = np.linalg.eigvalsh(cond_cov)
                if np.any(eigvals < 0):
                    cond_cov += np.eye(miss_alr.size) * (
                        abs(eigvals.min()) + 1e-10
                    )

            for i in row_indices:
                z_obs = X_alr[i, obs_alr]
                Y[i, miss_alr] = B[0, :] + z_obs @ B[1:, :]

                if method == "lrDA":
                    Y[i, miss_alr] += rng.multivariate_normal(
                        np.zeros(miss_alr.size), cond_cov
                    )

            # Accumulate variance correction
            block = np.zeros((n_alr, n_alr))
            block[np.ix_(miss_alr, miss_alr)] = CR
            v += block * len(row_indices)

        # M-step
        M = Y.mean(axis=0)
        dif = Y - M
        PC = dif.T @ dif
        C = (PC + v) / max(n - 1, 1)

        # Convergence: check max change in M and C (matches R)
        M_diff = np.max(np.abs(M - M_old))
        C_diff = np.max(np.abs(C - C_old))
        if max(M_diff, C_diff) < tol:
            break

    # ALR inverse: Y -> compositions
    exp_Y = np.exp(Y)
    total = 1.0 + exp_Y.sum(axis=1, keepdims=True)
    comp = np.empty((n, D))
    for k, j in enumerate(alr_cols):
        comp[:, j] = exp_Y[:, k] / total.ravel()
    comp[:, denom_idx] = 1.0 / total.ravel()

    # Restore original scale: for rows with missing values, scale so that
    # observed components match their original values.  Use the denominator
    # column (always observed) as the anchor: scale = X_orig[denom] / comp[denom].
    result = mat.copy()  # start from original data
    for i in range(n):
        if has_missing[i]:
            miss_cols = np.where(nan_mask[i])[0]
            scale = mat[i, denom_idx] / comp[i, denom_idx]
            for j in miss_cols:
                result[i, j] = comp[i, j] * scale

    if is_df:
        return pd.DataFrame(result, index=index, columns=columns)
    return result
