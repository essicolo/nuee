"""
Principal Component Analysis (PCA) implementation.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from .base import OrdinationResult, OrdinationMethod


class PCA(OrdinationMethod):
    """Principal Component Analysis for ordination."""
    
    def __init__(self, n_components: Optional[int] = None, 
                 scale: bool = True, center: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.center = center
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> OrdinationResult:
        """Fit PCA to the data."""
        X = self._validate_data(X)
        
        # Standardize if requested
        if self.scale and self.center:
            X = self._standardize_data(X)
        elif self.center:
            X = self._center_data(X)
        
        # Fit PCA
        pca = SklearnPCA(n_components=self.n_components)
        points = pca.fit_transform(X)
        
        # Species scores (loadings)
        species_scores = pca.components_.T
        
        call_info = {
            'method': 'PCA',
            'scale': self.scale,
            'center': self.center,
            'n_components': self.n_components
        }
        
        return OrdinationResult(
            points=points,
            species=species_scores,
            eigenvalues=pca.explained_variance_,
            call=call_info
        )


def pca(X: Union[np.ndarray, pd.DataFrame], 
        n_components: Optional[int] = None,
        scale: bool = True, **kwargs) -> OrdinationResult:
    """
    Principal Component Analysis.
    
    Parameters:
        X: Data matrix (samples x variables)
        n_components: Number of components to keep
        scale: Whether to scale variables
        **kwargs: Additional parameters
        
    Returns:
        OrdinationResult with PCA results
    """
    pca_obj = PCA(n_components=n_components, scale=scale)
    return pca_obj.fit(X, **kwargs)