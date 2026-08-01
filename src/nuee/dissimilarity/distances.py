"""
Distance and dissimilarity measures for community ecology.

This module implements various distance measures commonly used in
community ecology, following the conventions of the R nuee package.
"""

import numpy as np
from scipy.special import gammaln
import pandas as pd
from typing import Union, Optional, Callable, Dict, Any
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import warnings


def vegdist(x: Union[np.ndarray, pd.DataFrame],
           method: str = "bray",
           binary: bool = False,
           diag: bool = False,
           upper: bool = False,
           na_rm: bool = False) -> np.ndarray:
    """
    Calculate ecological distance matrices.

    This function calculates various dissimilarity indices commonly used
    in community ecology. It is designed to be compatible with the R vegan
    package's vegdist function.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Community data matrix with samples in rows and species in columns.
    method : str, default='bray'
        Distance measure: 'bray', 'jaccard', 'euclidean', 'manhattan',
        'canberra', 'gower', 'altGower', 'morisita', 'horn', 'mountford',
        'raup', 'binomial', 'chao', 'cao', 'kulczynski', 'mahalanobis'
    binary : bool, default=False
        Convert data to presence/absence before calculating distances.
    diag : bool, default=False
        Include diagonal elements in output.
    upper : bool, default=False
        Include upper triangle in output.
    na_rm : bool, default=False
        Remove samples with missing values.

    Returns
    -------
    np.ndarray
        Square symmetric distance matrix of shape (n_samples, n_samples).

    Examples
    --------
    >>> import nuee
    >>> species = nuee.datasets.varespec()
    >>> dist = nuee.vegdist(species, method="bray")
    >>> print(f"Shape: {dist.shape}")
    Shape: (24, 24)

    See Also
    --------
    adonis2 : PERMANOVA using distance matrices
    betadisper : Test for homogeneity of dispersions
    mantel : Mantel test for matrix correlation

    References
    ----------
    .. [1] Legendre, P. and Legendre, L. (2012). Numerical Ecology. Elsevier.
    """
    # Convert input to numpy array
    if isinstance(x, pd.DataFrame):
        x_array = x.values
    else:
        x_array = np.asarray(x, dtype=float)
    
    # Handle missing values
    if na_rm:
        x_array = x_array[~np.isnan(x_array).any(axis=1)]
    elif np.any(np.isnan(x_array)):
        warnings.warn("Missing values found in data. Consider using na_rm=True")
    
    # Convert to binary if requested
    if binary:
        x_array = (x_array > 0).astype(float)
    
    # Calculate distances based on method
    if method.lower() in ["bray", "braycurtis"]:
        dist_matrix = _bray_curtis(x_array)
    elif method.lower() == "jaccard":
        dist_matrix = _jaccard(x_array)
    elif method.lower() == "gower":
        dist_matrix = _gower(x_array)
    elif method.lower() == "altgower":
        dist_matrix = _alt_gower(x_array)
    elif method.lower() == "morisita":
        dist_matrix = _morisita(x_array)
    elif method.lower() == "horn":
        dist_matrix = _horn(x_array)
    elif method.lower() == "mountford":
        dist_matrix = _mountford(x_array)
    elif method.lower() == "raup":
        dist_matrix = _raup_crick(x_array)
    elif method.lower() == "binomial":
        dist_matrix = _binomial(x_array)
    elif method.lower() == "chao":
        dist_matrix = _chao(x_array)
    elif method.lower() == "cao":
        dist_matrix = _cao(x_array)
    elif method.lower() == "kulczynski":
        dist_matrix = _kulczynski(x_array)
    elif method.lower() == "mahalanobis":
        dist_matrix = _mahalanobis(x_array)
    elif method.lower() in ["manhattan", "cityblock"]:
        dist_matrix = squareform(pdist(x_array, metric="cityblock"))
    elif method.lower() == "euclidean":
        dist_matrix = squareform(pdist(x_array, metric="euclidean"))
    elif method.lower() == "canberra":
        # scipy's canberra is the unnormalised sum, which grows with the size
        # of the species pool; the ecological index averages over the terms
        # that contribute.
        dist_matrix = _canberra(x_array)
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    # Ensure symmetric matrix
    if not np.allclose(dist_matrix, dist_matrix.T):
        # If not symmetric, make it symmetric
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    # Handle diagonal 
    if not diag:
        np.fill_diagonal(dist_matrix, 0)
    
    return dist_matrix


def _bray_curtis(x: np.ndarray) -> np.ndarray:
    """Calculate Bray-Curtis dissimilarity."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            numerator = np.sum(np.abs(x[i] - x[j]))
            denominator = np.sum(x[i] + x[j])
            
            if denominator > 0:
                dist_matrix[i, j] = numerator / denominator
            else:
                dist_matrix[i, j] = 0
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix

def _pairwise(x: np.ndarray, kernel) -> np.ndarray:
    """Fill a symmetric distance matrix from a function of two rows."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            value = float(kernel(x[i], x[j]))
            dist_matrix[i, j] = value
            dist_matrix[j, i] = value
    return dist_matrix


def _jaccard(x: np.ndarray) -> np.ndarray:
    """Quantitative Jaccard distance, ``2B / (1 + B)`` for Bray-Curtis ``B``.

    Note this is *not* the binary Jaccard index: abundances are used. Pass
    ``binary=True`` to :func:`vegdist` for the presence/absence form, which is
    what vegan does too.
    """
    def kernel(a, b):
        total = a.sum() + b.sum()
        if total == 0:
            return 0.0
        bray = np.abs(a - b).sum() / total
        return 2 * bray / (1 + bray)

    return _pairwise(x, kernel)


def _gower(x: np.ndarray) -> np.ndarray:
    """Gower distance: mean of the range-standardised absolute differences."""
    column_range = x.max(axis=0) - x.min(axis=0)
    usable = column_range > 0

    def kernel(a, b):
        if not usable.any():
            return 0.0
        return float(np.sum(np.abs(a[usable] - b[usable]) / column_range[usable])
                     / usable.sum())

    return _pairwise(x, kernel)


def _alt_gower(x: np.ndarray) -> np.ndarray:
    """Alternative Gower distance: mean absolute difference over the species
    that are absent from neither site."""
    def kernel(a, b):
        non_double_zero = (a + b) != 0
        if not non_double_zero.any():
            return 0.0
        return float(np.sum(np.abs(a - b)) / non_double_zero.sum())

    return _pairwise(x, kernel)


def _canberra(x: np.ndarray) -> np.ndarray:
    """Canberra distance, averaged over the terms that contribute.

    Dividing by the number of non-double-zero species is what keeps the index
    in [0, 1]; without it the value grows with the size of the species pool.
    """
    def kernel(a, b):
        total = a + b
        contributing = total != 0
        if not contributing.any():
            return 0.0
        return float(np.sum(np.abs(a[contributing] - b[contributing])
                            / total[contributing]) / contributing.sum())

    return _pairwise(x, kernel)


def _morisita(x: np.ndarray) -> np.ndarray:
    """Morisita index of dissimilarity. Intended for integer counts."""
    def kernel(a, b):
        n_a, n_b = a.sum(), b.sum()
        if n_a <= 1 or n_b <= 1:
            return 0.0
        lambda_a = np.sum(a * (a - 1)) / (n_a * (n_a - 1))
        lambda_b = np.sum(b * (b - 1)) / (n_b * (n_b - 1))
        denominator = (lambda_a + lambda_b) * n_a * n_b
        if denominator == 0:
            return 0.0
        # Rounding can push a perfectly similar pair just below zero.
        return max(0.0, float(1 - 2 * np.sum(a * b) / denominator))

    return _pairwise(x, kernel)


def _horn(x: np.ndarray) -> np.ndarray:
    """Horn-Morisita index of dissimilarity."""
    def kernel(a, b):
        n_a, n_b = a.sum(), b.sum()
        if n_a == 0 or n_b == 0:
            return 0.0
        lambda_a = np.sum(a ** 2) / (n_a ** 2)
        lambda_b = np.sum(b ** 2) / (n_b ** 2)
        denominator = (lambda_a + lambda_b) * n_a * n_b
        if denominator == 0:
            return 0.0
        return max(0.0, float(1 - 2 * np.sum(a * b) / denominator))

    return _pairwise(x, kernel)


def _mountford(x: np.ndarray) -> np.ndarray:
    """Mountford index, ``M = 1/alpha`` of Fisher's log-series.

    ``M`` is the positive root of
    ``exp(a*M) + exp(b*M) = 1 + exp((a + b - j)*M)``, on presence/absence
    data. vegan reports ``log(2) - M``, which puts the index in ``[0,
    log(2)]`` with 0 for a subset and ``log(2)`` for disjoint sites; this
    follows vegan so the two agree.
    """
    log2 = np.log(2.0)

    def kernel(a, b):
        present_a, present_b = a > 0, b > 0
        n_a = int(present_a.sum())
        n_b = int(present_b.sum())
        shared = int(np.sum(present_a & present_b))

        if shared == 0:
            return log2
        if shared == n_a or shared == n_b:
            # One community is a subset of the other.
            return 0.0

        exponent = n_a + n_b - shared

        def equation(m):
            return np.exp(n_a * m) + np.exp(n_b * m) - np.exp(exponent * m) - 1.0

        # Bracket the positive root: the equation rises from zero then falls.
        upper = 1e-9
        while equation(upper) > 0 and upper < 50.0:
            upper *= 2.0
        if equation(upper) > 0:
            return 0.0

        lower = 1e-12
        for _ in range(200):
            middle = 0.5 * (lower + upper)
            if equation(middle) > 0:
                lower = middle
            else:
                upper = middle
            if upper - lower < 1e-14:
                break

        return float(log2 - 0.5 * (lower + upper))

    return _pairwise(x, kernel)


def _raup_crick(x: np.ndarray) -> np.ndarray:
    """Raup-Crick dissimilarity: the probability of finding fewer shared
    species than observed, if both sites drew from the same pool.

    This is a genuine probabilistic index; it used to return the binary
    Jaccard distance instead.
    """
    n_species = x.shape[1]

    def kernel(a, b):
        present_a, present_b = a > 0, b > 0
        n_a = int(present_a.sum())
        n_b = int(present_b.sum())
        shared = int(np.sum(present_a & present_b))

        probability = 0.0
        for count in range(shared):
            if n_b - count < 0 or n_species - n_a - n_b + count < 0:
                continue
            log_probability = (
                gammaln(n_a + 1) - gammaln(count + 1) - gammaln(n_a - count + 1)
                + gammaln(n_species - n_a + 1) - gammaln(n_b - count + 1)
                - gammaln(n_species - n_a - n_b + count + 1)
                - (gammaln(n_species + 1) - gammaln(n_b + 1)
                   - gammaln(n_species - n_b + 1))
            )
            probability += np.exp(log_probability)

        return float(1.0 - probability)

    return _pairwise(x, kernel)


def _binomial(x: np.ndarray) -> np.ndarray:
    """Binomial index, a log-likelihood contrast of the two sites."""
    def kernel(a, b):
        total = a + b
        contributing = total > 0
        if not contributing.any():
            return 0.0
        a_c, b_c, total_c = a[contributing], b[contributing], total[contributing]
        with np.errstate(divide="ignore", invalid="ignore"):
            term_a = np.where(a_c > 0, a_c * np.log(a_c / total_c), 0.0)
            term_b = np.where(b_c > 0, b_c * np.log(b_c / total_c), 0.0)
        return float(np.sum((term_a + term_b + total_c * np.log(2.0)) / total_c))

    return _pairwise(x, kernel)


def _chao(x: np.ndarray) -> np.ndarray:
    """Chao's abundance-based Jaccard dissimilarity.

    Corrects for the species shared but unseen in one of the samples, using
    the singletons and doubletons among the shared species.
    """
    def kernel(a, b):
        n_a, n_b = a.sum(), b.sum()
        shared = (a > 0) & (b > 0)
        if not shared.any() or n_a == 0 or n_b == 0:
            return 1.0

        singletons_b = np.sum(shared & (b == 1))
        doubletons_b = max(int(np.sum(shared & (b == 2))), 1)
        singletons_a = np.sum(shared & (a == 1))
        doubletons_a = max(int(np.sum(shared & (a == 2))), 1)

        u = a[shared].sum() / n_a
        v = b[shared].sum() / n_b
        u += ((n_b - 1) / n_b) * (singletons_b / (2 * doubletons_b)) \
            * np.sum(a[shared & (b == 1)]) / n_a
        v += ((n_a - 1) / n_a) * (singletons_a / (2 * doubletons_a)) \
            * np.sum(b[shared & (a == 1)]) / n_b
        u, v = min(u, 1.0), min(v, 1.0)

        denominator = u + v - u * v
        return float(1 - u * v / denominator) if denominator else 1.0

    return _pairwise(x, kernel)


def _cao(x: np.ndarray) -> np.ndarray:
    """Cao (CYd) index, for count data with variable sampling intensity.

    Undefined at zero abundance, so abundances are floored at 0.1 following
    Cao et al. (1997). Flooring rather than replacing only the zeros matters
    for non-integer input; on counts the two coincide. Natural logarithms are
    used, as vegan does; the original paper used base 10, which makes values
    about 2.30 times smaller.
    """
    floor = 0.1

    def kernel(a, b):
        present = (a > 0) | (b > 0)
        if not present.any():
            return 0.0
        a_filled = np.where(a < floor, floor, a).astype(float)
        b_filled = np.where(b < floor, floor, b).astype(float)
        total = a_filled + b_filled
        term = (np.log(total / 2.0)
                - (a_filled * np.log(b_filled) + b_filled * np.log(a_filled)) / total)
        return float(np.sum(term[present]) / present.sum())

    return _pairwise(x, kernel)


def _kulczynski(x: np.ndarray) -> np.ndarray:
    """Kulczynski dissimilarity."""
    def kernel(a, b):
        shared = np.minimum(a, b).sum()
        n_a, n_b = a.sum(), b.sum()
        if n_a == 0 or n_b == 0:
            return 1.0
        return float(1 - 0.5 * (shared / n_a + shared / n_b))

    return _pairwise(x, kernel)


def _mahalanobis(x: np.ndarray) -> np.ndarray:
    """Calculate Mahalanobis distance."""
    try:
        # Calculate covariance matrix
        cov_matrix = np.cov(x.T)
        
        # Calculate pseudo-inverse for singular matrices
        cov_inv = np.linalg.pinv(cov_matrix)
        
        # Calculate Mahalanobis distance
        dist_matrix = squareform(pdist(x, metric='mahalanobis', VI=cov_inv))
        
        return dist_matrix
    except:
        # Fallback to Euclidean if Mahalanobis fails
        warnings.warn("Mahalanobis distance calculation failed, using Euclidean")
        return squareform(pdist(x, metric='euclidean'))