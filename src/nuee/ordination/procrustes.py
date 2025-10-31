"""
Procrustes analysis for comparing ordinations.
"""

import numpy as np
import pandas as pd
import warnings
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

    x_mean = np.mean(X, axis=0)
    y_mean = np.mean(Y, axis=0)

    X_centered = X - x_mean
    Y_centered = Y - y_mean

    if scale:
        norm_x = np.linalg.norm(X_centered, ord="fro")
        norm_y = np.linalg.norm(Y_centered, ord="fro")
        if norm_x > 0:
            X_centered = X_centered / norm_x
        if norm_y > 0:
            Y_centered = Y_centered / norm_y
    else:
        norm_x = norm_y = 1.0

    R, _ = orthogonal_procrustes(X_centered, Y_centered)
    X_rotated = X_centered @ R

    if scale:
        denom = np.trace(X_rotated.T @ X_rotated)
        if denom > 0:
            dilation = np.trace(X_rotated.T @ Y_centered) / denom
        else:
            dilation = 1.0
    else:
        dilation = 1.0

    X_transformed = dilation * X_rotated

    ss_total = np.sum(Y_centered ** 2)
    ss_residual = np.sum((Y_centered - X_transformed) ** 2)
    correlation = np.sqrt(max(0.0, 1 - ss_residual / ss_total)) if ss_total > 0 else 0.0

    translation = y_mean - dilation * (x_mean / norm_x if scale and norm_x > 0 else x_mean) @ R

    warnings.warn(
        "nuee.procrustes currently depends on the metaMDS configuration and may "
        "report residual sums-of-squares that differ slightly from vegan::procrustes.",
        UserWarning,
        stacklevel=2,
    )

    return {
        'rotation': R,
        'transformed': X_transformed,
        'correlation': correlation,
        'ss': ss_residual,
        'scale': scale,
        'dilation': dilation,
        'translation': translation
    }
