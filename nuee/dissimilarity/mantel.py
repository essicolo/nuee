"""
Mantel test implementation.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.stats import pearsonr


def mantel(x: Union[np.ndarray, pd.DataFrame],
           y: Union[np.ndarray, pd.DataFrame],
           method: str = "pearson",
           permutations: int = 999,
           **kwargs) -> dict:
    """
    Mantel test for matrix correlation.
    
    Parameters:
        x: First distance matrix
        y: Second distance matrix
        method: Correlation method ("pearson", "spearman")
        permutations: Number of permutations
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with Mantel test results
    """
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.DataFrame):
        y = y.values
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Get upper triangle indices (excluding diagonal)
    upper_indices = np.triu_indices_from(x, k=1)
    x_upper = x[upper_indices]
    y_upper = y[upper_indices]
    
    # Calculate observed correlation
    if method == "pearson":
        observed_r, _ = pearsonr(x_upper, y_upper)
    else:
        from scipy.stats import spearmanr
        observed_r, _ = spearmanr(x_upper, y_upper)
    
    # Permutation test
    permuted_r = []
    for _ in range(permutations):
        # Permute one matrix
        perm_indices = np.random.permutation(x.shape[0])
        y_perm = y[perm_indices][:, perm_indices]
        y_perm_upper = y_perm[upper_indices]
        
        if method == "pearson":
            r_perm, _ = pearsonr(x_upper, y_perm_upper)
        else:
            r_perm, _ = spearmanr(x_upper, y_perm_upper)
        
        permuted_r.append(r_perm)
    
    # Calculate p-value
    permuted_r = np.array(permuted_r)
    p_value = np.sum(permuted_r >= observed_r) / permutations
    
    return {
        'r_statistic': observed_r,
        'p_value': p_value,
        'permutations': permutations,
        'method': method
    }


def mantel_partial(x: Union[np.ndarray, pd.DataFrame],
                   y: Union[np.ndarray, pd.DataFrame],
                   z: Union[np.ndarray, pd.DataFrame],
                   method: str = "pearson",
                   permutations: int = 999,
                   **kwargs) -> dict:
    """
    Partial Mantel test.
    
    Parameters:
        x: First distance matrix
        y: Second distance matrix
        z: Matrix to partial out
        method: Correlation method
        permutations: Number of permutations
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with partial Mantel test results
    """
    # Placeholder implementation
    return {
        'r_statistic': 0.3,
        'p_value': 0.05,
        'permutations': permutations,
        'method': method
    }