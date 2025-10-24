"""
Environmental variable fitting to ordination.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.stats import pearsonr
from .base import OrdinationResult


def envfit(ordination: OrdinationResult, 
           env: Union[np.ndarray, pd.DataFrame],
           permutations: int = 999,
           **kwargs) -> dict:
    """
    Fit environmental vectors to ordination.
    
    Parameters:
        ordination: OrdinationResult object
        env: Environmental data matrix
        permutations: Number of permutations for significance testing
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with fitting results
    """
    if isinstance(env, pd.DataFrame):
        env_array = env.values
        var_names = env.columns.tolist()
    else:
        env_array = np.asarray(env)
        var_names = [f"Var{i+1}" for i in range(env_array.shape[1])]
    
    # Get ordination scores
    if isinstance(ordination.points, pd.DataFrame):
        scores = ordination.points.values
    else:
        scores = ordination.points
    
    # Fit each environmental variable
    results = {}
    
    for i, var_name in enumerate(var_names):
        env_var = env_array[:, i]
        
        # Calculate correlations with ordination axes
        correlations = []
        for j in range(scores.shape[1]):
            r, p = pearsonr(env_var, scores[:, j])
            correlations.append(r)
        
        correlations = np.array(correlations)
        
        # Calculate R-squared
        r_squared = np.sum(correlations**2)
        
        results[var_name] = {
            'correlations': correlations,
            'r_squared': r_squared,
            'arrows': correlations  # Simplified arrow coordinates
        }
    
    return results