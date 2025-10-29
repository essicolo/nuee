"""
Procrustes analysis for comparing ordinations.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.linalg import orthogonal_procrustes


def procrustes(X: Union[np.ndarray, pd.DataFrame], 
               Y: Union[np.ndarray, pd.DataFrame],
               scale: bool = True) -> dict:
    """
    Procrustes analysis to compare two ordinations.
    
    Parameters:
        X: First ordination matrix
        Y: Second ordination matrix  
        scale: Whether to scale configurations
        
    Returns:
        Dictionary with Procrustes results
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
        
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    
    # Center the configurations
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Scale if requested
    if scale:
        X_centered = X_centered / np.linalg.norm(X_centered, 'fro')
        Y_centered = Y_centered / np.linalg.norm(Y_centered, 'fro')
    
    # Find optimal rotation matrix
    R, _ = orthogonal_procrustes(X_centered, Y_centered)
    
    # Transform X to match Y
    X_transformed = X_centered @ R
    
    # Calculate sum of squares
    ss_total = np.sum(Y_centered**2)
    ss_residual = np.sum((Y_centered - X_transformed)**2)
    
    # Procrustes correlation
    correlation = np.sqrt(1 - ss_residual / ss_total)
    
    return {
        'rotation': R,
        'transformed': X_transformed,
        'correlation': correlation,
        'ss': ss_residual,
        'scale': scale
    }