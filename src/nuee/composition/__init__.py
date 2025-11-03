"""
Compositional data analysis utilities tailored for nuee.

This module provides NumPy-only adaptations of scikit-bio's composition
utilities so they can be used in environments where SciPy is unavailable
(for example Pyodide).  The implementations are derived from the scikit-bio
project (Modified BSD License) with minimal adjustments for integration in
``nuee``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "closure",
    "multiplicative_replacement",
    "power",
    "clr",
    "clr_inv",
    "inner",
    "ilr",
    "ilr_inv",
    "alr",
    "alr_inv",
    "sbp_basis",
    "center",
    "centralize",
]


def closure(mat: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
    """
    Perform closure so that each composition sums to 1.

    Parameters
    ----------
    mat
        A matrix where rows are compositions and columns are components.
    out
        Optional array where the result is stored.

    Returns
    -------
    numpy.ndarray
        Matrix of proportions with non-negative entries that sum to 1 per row.
    """
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    if out is not None:
        out = np.atleast_2d(out)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    norm = mat.sum(axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Input matrix cannot have rows with all zeros")
    result = np.divide(mat, norm, out=out)
    if was_1d:
        return result[0]
    return result


def multiplicative_replacement(mat: np.ndarray, delta: float | None = None) -> np.ndarray:
    """Replace structural zeros with a small non-zero value."""
    mat = np.asarray(mat, dtype=float)
    z_mat = mat == 0
    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1.0 / num_feats) ** 2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError(
            "The multiplicative replacement created negative proportions. "
            "Consider using a smaller `delta`."
        )
    mat = np.where(z_mat, delta, zcnts * mat)
    return closure(mat)


def power(x: np.ndarray, a: float) -> np.ndarray:
    """Raise each component to a power and renormalise via closure."""
    y = np.power(x, a)
    return closure(y, out=y)


def clr(mat: np.ndarray, ignore_zero: bool = False) -> np.ndarray:
    """Compute the centred log-ratio transformation."""
    mat = np.asarray(mat, dtype=float)
    if ignore_zero:
        mat = np.where(mat == 0, 1, mat)
    if np.any(mat < 0):
        raise ValueError("clr only accepts non-negative values")
    if np.any(np.sum(mat, axis=-1) == 0):
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = closure(mat)
    gm = _gmean(mat, axis=-1)
    return np.log(np.divide(mat, gm[..., np.newaxis]))


def clr_inv(mat: np.ndarray) -> np.ndarray:
    """Inverse centred log-ratio transformation."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    mat = np.exp(mat)
    result = closure(mat)
    if was_1d:
        return result[0]
    return result


def inner(mat: np.ndarray) -> np.ndarray:
    """Compute the inner product matrix in the Aitchison simplex."""
    mat = np.asarray(mat, dtype=float)
    mat = np.atleast_2d(mat)
    _check_positive(mat)
    mat = closure(mat)
    mat = np.log(mat)
    centered = mat - mat.mean(axis=1, keepdims=True)
    return np.dot(centered, centered.T) / mat.shape[1]


def ilr(mat: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Perform the isometric log-ratio transformation."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    if basis is None:
        basis = _default_basis(mat.shape[-1])
    _check_positive(mat)
    mat = closure(mat)
    mat = np.log(mat)
    transformed = np.dot(mat, basis.T)
    if was_1d:
        return transformed[0]
    return transformed


def ilr_inv(mat: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Inverse isometric log-ratio transformation."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    if basis is None:
        basis = _default_basis(mat.shape[-1] + 1)
    mat = np.dot(mat, basis)
    mat = np.exp(mat)
    result = closure(mat)
    if was_1d:
        return result[0]
    return result


def alr(mat: np.ndarray, denominator_idx: int = -1) -> np.ndarray:
    """Perform the additive log-ratio transformation."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    _check_positive(mat)
    mat = closure(mat)
    denom = mat[..., denominator_idx][..., np.newaxis]
    numerators = np.delete(mat, denominator_idx, axis=-1)
    transformed = np.log(np.divide(numerators, denom))
    if was_1d:
        return transformed[0]
    return transformed


def alr_inv(mat: np.ndarray, denominator_idx: int = -1) -> np.ndarray:
    """Inverse additive log-ratio transformation."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    if mat.ndim == 2:
        mat_exp = np.insert(np.exp(mat), denominator_idx, 1, axis=1)
    elif mat.ndim == 1:
        mat_exp = np.insert(np.exp(mat), denominator_idx, 1)
    else:
        raise ValueError("mat must be either 1D or 2D")
    result = closure(mat_exp)
    if was_1d:
        return result
    return result


def sbp_basis(sbp: np.ndarray) -> np.ndarray:
    """Construct an orthonormal basis from a sequential binary partition."""
    sbp = np.asarray(sbp, dtype=int)
    n_pos = (sbp == 1).sum(axis=1)
    n_neg = (sbp == -1).sum(axis=1)
    psi = np.zeros(sbp.shape, dtype=float)
    for i in range(sbp.shape[0]):
        psi[i, :] = sbp[i, :] * np.sqrt(
            (n_neg[i] / n_pos[i]) ** sbp[i, :] / np.sum(np.abs(sbp[i, :]))
        )
    return psi


def center(mat: np.ndarray) -> np.ndarray:
    """Alias for :func:`centralize` kept for API parity."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    cen = _gmean(mat, axis=0)
    result = closure(mat / cen)
    if was_1d:
        return result[0]
    return result


def centralize(mat: np.ndarray) -> np.ndarray:
    """Center compositions by their geometric mean."""
    mat = np.asarray(mat, dtype=float)
    was_1d = mat.ndim == 1
    mat = np.atleast_2d(mat)
    cen = _gmean(mat, axis=0)
    result = closure(mat / cen)
    if was_1d:
        return result[0]
    return result


def _gmean(mat: np.ndarray, axis: int = 0) -> np.ndarray:
    """Geometric mean along a given axis (SciPy-free)."""
    log_mat = np.log(mat)
    return np.exp(log_mat.mean(axis=axis))


def _default_basis(size: int) -> np.ndarray:
    """Return the Gram-Schmidt-based orthonormal ILR basis."""
    if size <= 1:
        raise ValueError("Basis size must be greater than 1")
    return _gram_schmidt_basis(size)


def _gram_schmidt_basis(n: int) -> np.ndarray:
    """Build CLR-transformed basis derived from Gram-Schmidt orthogonalisation."""
    basis = np.zeros((n, n - 1), dtype=float)
    for j in range(n - 1):
        i = j + 1
        vector = np.array([(1 / i)] * i + [-1] + [0] * (n - i - 1), dtype=float)
        basis[:, j] = vector * np.sqrt(i / (i + 1))
    return basis.T


def _check_positive(mat: np.ndarray) -> None:
    """Ensure all entries are strictly positive."""
    if np.any(mat <= 0):
        raise ValueError("Input matrix cannot contain zeros or negative values")
