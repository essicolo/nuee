"""
Protest (Procrustes rotation) for comparing configurations.
"""

import numpy as np
import pandas as pd
from typing import Union
from ..ordination.procrustes import procrustes


def protest(x: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame],
            permutations: int = 999,
            **kwargs) -> dict:
    """
    Protest analysis using Procrustes rotation.
    
    Parameters:
        x: First configuration matrix
        y: Second configuration matrix
        permutations: Number of permutations
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with Protest results
    """
    # Perform Procrustes analysis
    procrustes_result = procrustes(x, y)
    
    # Permutation test
    observed_correlation = procrustes_result['correlation']
    permuted_correlations = []
    
    if isinstance(y, pd.DataFrame):
        y_values = y.values
    else:
        y_values = np.asarray(y)
    
    for _ in range(permutations):
        # Permute rows of y
        perm_indices = np.random.permutation(y_values.shape[0])
        y_perm = y_values[perm_indices]
        
        perm_result = procrustes(x, y_perm)
        permuted_correlations.append(perm_result['correlation'])
    
    # Calculate p-value
    permuted_correlations = np.array(permuted_correlations)
    p_value = np.sum(permuted_correlations >= observed_correlation) / permutations
    
    return {
        'correlation': observed_correlation,
        'p_value': p_value,
        'permutations': permutations,
        'rotation': procrustes_result['rotation'],
        'ss': procrustes_result['ss']
    }