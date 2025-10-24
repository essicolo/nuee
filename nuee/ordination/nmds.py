"""
Non-metric Multidimensional Scaling (NMDS) implementation.

This module provides the metaMDS function, which is a wrapper around
scikit-learn's MDS with additional functionality for community ecology.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Callable, Dict, Any
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
import warnings

from .base import OrdinationResult, OrdinationMethod
from ..dissimilarity import vegdist


class NMDS(OrdinationMethod):
    """
    Non-metric Multidimensional Scaling for community ecology.
    
    This implementation follows the approach used in nuee's metaMDS function,
    including multiple random starts and stress evaluation.
    """
    
    def __init__(self, n_components: int = 2, max_iter: int = 200, 
                 n_init: int = 20, eps: float = 1e-3, random_state: Optional[int] = None,
                 dissimilarity: str = "bray", n_jobs: Optional[int] = None):
        """
        Initialize NMDS.
        
        Parameters:
            n_components: Number of dimensions for the embedding
            max_iter: Maximum number of iterations
            n_init: Number of random initializations
            eps: Convergence tolerance
            random_state: Random seed for reproducibility
            dissimilarity: Distance metric to use
            n_jobs: Number of parallel jobs
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.eps = eps
        self.random_state = random_state
        self.dissimilarity = dissimilarity
        self.n_jobs = n_jobs
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            distance: Optional[np.ndarray] = None,
            **kwargs) -> OrdinationResult:
        """
        Fit NMDS to the data.
        
        Parameters:
            X: Community data matrix (samples x species) or distance matrix
            distance: Precomputed distance matrix (optional)
            **kwargs: Additional parameters
            
        Returns:
            OrdinationResult with NMDS coordinates and stress
        """
        X = self._validate_data(X)
        
        # Calculate or use provided distance matrix
        if distance is not None:
            if isinstance(distance, pd.DataFrame):
                distance = distance.values
            dist_matrix = np.asarray(distance)
        else:
            # Calculate distance matrix using vegdist
            dist_matrix = vegdist(X, method=self.dissimilarity)
            
        # Ensure distance matrix is symmetric
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            # If it's a condensed distance matrix, convert to square
            if dist_matrix.shape[0] == X.shape[0] * (X.shape[0] - 1) // 2:
                dist_matrix = squareform(dist_matrix)
            else:
                raise ValueError("Distance matrix dimensions are incompatible")
                
        # Perform NMDS using sklearn's MDS
        mds = MDS(n_components=self.n_components, 
                  max_iter=self.max_iter,
                  n_init=self.n_init,
                  eps=self.eps,
                  random_state=self.random_state,
                  dissimilarity='precomputed',
                  n_jobs=self.n_jobs)
        
        points = mds.fit_transform(dist_matrix)
        stress = mds.stress_
        
        # Normalize stress (following nuee convention)
        stress = stress / np.sum(dist_matrix**2) * 100
        
        call_info = {
            'method': 'NMDS',
            'dissimilarity': self.dissimilarity,
            'n_components': self.n_components,
            'n_init': self.n_init,
            'max_iter': self.max_iter
        }
        
        return OrdinationResult(
            points=points,
            stress=stress,
            converged=mds.n_iter_ < self.max_iter,
            call=call_info
        )


def metaMDS(X: Union[np.ndarray, pd.DataFrame], 
            k: int = 2, 
            distance: str = "bray",
            trymax: int = 20,
            maxit: int = 200,
            trace: bool = False,
            autotransform: bool = True,
            wascores: bool = True,
            expand: bool = True,
            **kwargs) -> OrdinationResult:
    """
    Non-metric Multidimensional Scaling with automatic transformation.
    
    This function is designed to be similar to nuee's metaMDS function,
    providing a high-level interface for NMDS with sensible defaults.
    
    Parameters:
        X: Community data matrix (samples x species)
        k: Number of dimensions
        distance: Distance metric to use
        trymax: Maximum number of random starts
        maxit: Maximum number of iterations per start
        trace: Whether to print progress information
        autotransform: Whether to apply automatic data transformation
        wascores: Whether to calculate species scores
        expand: Whether to expand the result to include more information
        **kwargs: Additional parameters passed to NMDS
        
    Returns:
        OrdinationResult with NMDS coordinates and additional information
    """
    # Validate input
    if isinstance(X, pd.DataFrame):
        row_names = X.index.tolist()
        col_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.asarray(X)
        row_names = [f"Site{i+1}" for i in range(X_array.shape[0])]
        col_names = [f"Species{i+1}" for i in range(X_array.shape[1])]
    
    # Data transformation
    if autotransform:
        # Apply Wisconsin double standardization if requested
        if 'wisconsin' in kwargs.get('transform', []):
            X_array = _wisconsin_transform(X_array)
        
        # Apply square root transformation for abundance data
        if np.any(X_array > 1):
            if trace:
                print("Applying square root transformation")
            X_array = np.sqrt(X_array)
    
    # Perform NMDS
    nmds = NMDS(n_components=k, 
                max_iter=maxit,
                n_init=trymax,
                dissimilarity=distance,
                **kwargs)
    
    result = nmds.fit(X_array)
    
    # Calculate species scores if requested
    if wascores and X_array.shape[1] > 0:
        species_scores = _calculate_species_scores(X_array, result.points)
        result.species = species_scores
    
    # Add row and column names
    if hasattr(result, 'points'):
        result.points = pd.DataFrame(result.points, 
                                   index=row_names,
                                   columns=[f"NMDS{i+1}" for i in range(k)])
    
    if hasattr(result, 'species') and result.species is not None:
        result.species = pd.DataFrame(result.species,
                                    index=col_names,
                                    columns=[f"NMDS{i+1}" for i in range(k)])
    
    if trace:
        print(f"NMDS stress: {result.stress:.4f}")
        if result.converged:
            print("NMDS converged")
        else:
            print("NMDS did not converge")
    
    return result


def _wisconsin_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply Wisconsin double standardization.
    
    First standardizes by species maxima, then by site totals.
    """
    # Standardize by species maxima
    species_max = np.max(X, axis=0)
    species_max[species_max == 0] = 1  # Avoid division by zero
    X_std = X / species_max
    
    # Standardize by site totals
    site_totals = np.sum(X_std, axis=1)
    site_totals[site_totals == 0] = 1  # Avoid division by zero
    X_std = X_std / site_totals[:, np.newaxis]
    
    return X_std


def _calculate_species_scores(X: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Calculate weighted average species scores.
    
    Species scores are calculated as weighted averages of site scores,
    with species abundances as weights.
    """
    # Normalize abundances to weights
    weights = X / np.sum(X, axis=0)[np.newaxis, :]
    weights = np.nan_to_num(weights)  # Handle division by zero
    
    # Calculate weighted average coordinates
    species_scores = np.dot(weights.T, points)
    
    return species_scores