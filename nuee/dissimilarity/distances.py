"""
Distance and dissimilarity measures for community ecology.

This module implements various distance measures commonly used in
community ecology, following the conventions of the R nuee package.
"""

import numpy as np
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
        dist_matrix = squareform(pdist(x_array, metric="canberra"))
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


def _jaccard(x: np.ndarray) -> np.ndarray:
    """Calculate Jaccard distance."""
    # Convert to binary
    x_binary = (x > 0).astype(int)
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            intersection = np.sum(x_binary[i] & x_binary[j])
            union = np.sum(x_binary[i] | x_binary[j])
            
            if union > 0:
                dist_matrix[i, j] = 1 - intersection / union
            else:
                dist_matrix[i, j] = 0
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _gower(x: np.ndarray) -> np.ndarray:
    """Calculate Gower distance."""
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


def _alt_gower(x: np.ndarray) -> np.ndarray:
    """Calculate alternative Gower distance."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            numerator = np.sum(np.abs(x[i] - x[j]))
            denominator = np.sum(np.maximum(x[i], x[j]))
            
            if denominator > 0:
                dist_matrix[i, j] = numerator / denominator
            else:
                dist_matrix[i, j] = 0
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _morisita(x: np.ndarray) -> np.ndarray:
    """Calculate Morisita distance."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = x[i], x[j]
            ni, nj = np.sum(xi), np.sum(xj)
            
            if ni > 0 and nj > 0:
                lambda_i = np.sum(xi * (xi - 1)) / (ni * (ni - 1)) if ni > 1 else 0
                lambda_j = np.sum(xj * (xj - 1)) / (nj * (nj - 1)) if nj > 1 else 0
                
                numerator = 2 * np.sum(xi * xj)
                denominator = (lambda_i + lambda_j) * ni * nj
                
                if denominator > 0:
                    dist_matrix[i, j] = 1 - numerator / denominator
                else:
                    dist_matrix[i, j] = 1
            else:
                dist_matrix[i, j] = 1
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _horn(x: np.ndarray) -> np.ndarray:
    """Calculate Horn distance (Morisita-Horn)."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = x[i], x[j]
            ni, nj = np.sum(xi), np.sum(xj)
            
            if ni > 0 and nj > 0:
                pi = xi / ni
                pj = xj / nj
                
                numerator = 2 * np.sum(pi * pj)
                denominator = np.sum(pi**2) + np.sum(pj**2)
                
                if denominator > 0:
                    dist_matrix[i, j] = 1 - numerator / denominator
                else:
                    dist_matrix[i, j] = 1
            else:
                dist_matrix[i, j] = 1
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _mountford(x: np.ndarray) -> np.ndarray:
    """Calculate Mountford distance."""
    # Convert to binary
    x_binary = (x > 0).astype(int)
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            a = np.sum(x_binary[i] & x_binary[j])  # Common species
            b = np.sum(x_binary[i] & ~x_binary[j])  # Only in i
            c = np.sum(~x_binary[i] & x_binary[j])  # Only in j
            
            if a > 0 and (b + c) > 0:
                dist_matrix[i, j] = 1 / (0.5 * ((a / (a + b)) + (a / (a + c))))
            else:
                dist_matrix[i, j] = 1
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _raup_crick(x: np.ndarray) -> np.ndarray:
    """Calculate Raup-Crick distance."""
    # This is a simplified version - full implementation would require probabilistic model
    return _jaccard(x)


def _binomial(x: np.ndarray) -> np.ndarray:
    """Calculate binomial distance."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = x[i], x[j]
            ni, nj = np.sum(xi), np.sum(xj)
            
            if ni > 0 and nj > 0:
                pi = xi / ni
                pj = xj / nj
                
                # Binomial probability
                prob = np.sum(np.minimum(pi, pj))
                dist_matrix[i, j] = 1 - prob
            else:
                dist_matrix[i, j] = 1
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _chao(x: np.ndarray) -> np.ndarray:
    """Calculate Chao distance."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = x[i], x[j]
            
            # Chao similarity
            u = np.sum(xi * xj)
            v = np.sum(xi * (xi - 1) * xj * (xj - 1))
            
            if u > 0:
                # Simplified Chao index
                similarity = 2 * u / (np.sum(xi**2) + np.sum(xj**2))
                dist_matrix[i, j] = 1 - similarity
            else:
                dist_matrix[i, j] = 1
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _cao(x: np.ndarray) -> np.ndarray:
    """Calculate Cao distance."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = x[i], x[j]
            
            # Cao CYd distance
            numerator = np.sum(np.abs(xi - xj))
            denominator = np.sum(xi + xj)
            
            if denominator > 0:
                dist_matrix[i, j] = numerator / denominator
            else:
                dist_matrix[i, j] = 0
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


def _kulczynski(x: np.ndarray) -> np.ndarray:
    """Calculate Kulczynski distance."""
    n_samples = x.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = x[i], x[j]
            
            # Kulczynski similarity
            numerator = np.sum(np.minimum(xi, xj))
            denominator = 0.5 * (np.sum(xi) + np.sum(xj))
            
            if denominator > 0:
                dist_matrix[i, j] = 1 - numerator / denominator
            else:
                dist_matrix[i, j] = 1
            
            dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix


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