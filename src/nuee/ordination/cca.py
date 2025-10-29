"""
Canonical Correspondence Analysis (CCA) implementation.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd

from .base import ConstrainedOrdinationResult, OrdinationMethod


class CCA(OrdinationMethod):
    """Canonical Correspondence Analysis."""
    
    def __init__(self, formula: Optional[str] = None):
        self.formula = formula
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            Y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            **kwargs) -> ConstrainedOrdinationResult:
        """Fit CCA to the data."""
        X = self._validate_data(X)
        
        if Y is None:
            raise ValueError("Environmental matrix Y is required for CCA")
        
        Y = self._validate_data(Y)
        
        # Chi-square transformation for species data
        X_chi = self._chi_square_transform(X)
        
        # Weighted regression
        result = self._weighted_regression(X_chi, Y)
        
        call_info = {
            'method': 'CCA',
            'formula': self.formula
        }
        
        return result


    def _chi_square_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply chi-square transformation to species data."""
        # Row and column totals
        row_totals = np.sum(X, axis=1)
        col_totals = np.sum(X, axis=0)
        grand_total = np.sum(X)
        
        # Avoid division by zero
        row_totals[row_totals == 0] = 1
        col_totals[col_totals == 0] = 1
        
        # Chi-square transformation
        expected = np.outer(row_totals, col_totals) / grand_total
        expected[expected == 0] = 1
        
        X_chi = (X - expected) / np.sqrt(expected)
        
        return X_chi
    
    def _weighted_regression(self, X: np.ndarray, Y: np.ndarray) -> ConstrainedOrdinationResult:
        """Perform weighted regression for CCA."""
        # This is a simplified implementation
        # Full CCA requires more complex weighted analysis
        
        # For now, use a simplified approach similar to RDA
        # but with chi-square transformed data
        
        # Center environmental variables
        Y_centered = self._center_data(Y)
        
        # Perform SVD on cross-covariance matrix
        U, s, Vt = svd(X.T @ Y_centered, full_matrices=False)
        
        # Sample scores
        points = X @ U[:, :min(X.shape[1], Y.shape[1])]
        
        # Species scores
        species_scores = U[:, :min(X.shape[1], Y.shape[1])]
        
        # Eigenvalues
        eigenvalues = s[:min(X.shape[1], Y.shape[1])]
        
        # Biplot scores
        biplot_scores = Vt[:min(X.shape[1], Y.shape[1]), :].T
        
        return ConstrainedOrdinationResult(
            points=points,
            species=species_scores,
            constrained_eig=eigenvalues,
            biplot=biplot_scores,
            tot_chi=np.sum(eigenvalues)
        )


def cca(X: Union[np.ndarray, pd.DataFrame], 
        Y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        formula: Optional[str] = None,
        **kwargs) -> ConstrainedOrdinationResult:
    """
    Canonical Correspondence Analysis.
    
    Parameters:
        X: Species data matrix
        Y: Environmental data matrix
        formula: Formula string (optional)
        **kwargs: Additional parameters
        
    Returns:
        ConstrainedOrdinationResult with CCA results
    """
    cca_obj = CCA(formula=formula)
    return cca_obj.fit(X, Y, **kwargs)