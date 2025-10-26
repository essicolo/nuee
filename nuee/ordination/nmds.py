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

    This implementation follows the approach used in vegan's metaMDS function,
    including multiple random starts and stress evaluation. NMDS is a rank-based
    ordination method that attempts to represent ecological distances in a
    reduced dimensional space.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions for the embedding (ordination axes).
    max_iter : int, default=200
        Maximum number of iterations for the optimization algorithm.
    n_init : int, default=20
        Number of random initializations. The solution with minimum stress
        is returned.
    eps : float, default=1e-3
        Convergence tolerance for the stress value.
    random_state : int, optional
        Random seed for reproducibility. If None, the random state is not fixed.
    dissimilarity : str, default="bray"
        Distance metric to use. See :func:`nuee.vegdist` for available options.
    n_jobs : int, optional
        Number of parallel jobs for computation. If None, uses a single core.

    Attributes
    ----------
    n_components : int
        Number of dimensions.
    max_iter : int
        Maximum iterations.
    n_init : int
        Number of random starts.
    eps : float
        Convergence tolerance.

    See Also
    --------
    metaMDS : High-level interface with automatic transformations
    MDS : Scikit-learn's MDS implementation

    Notes
    -----
    NMDS is particularly useful when:
    - Relationships between samples are non-linear
    - You want to use a specific distance metric
    - You have presence/absence or abundance data

    The stress value indicates goodness-of-fit:
    - Values < 0.05 indicate excellent fit
    - Values 0.05-0.10 indicate good fit
    - Values > 0.20 indicate poor fit

    References
    ----------
    .. [1] Kruskal, J.B. (1964). Nonmetric multidimensional scaling.
           Psychometrika 29, 115-129.

    Examples
    --------
    >>> from nuee.ordination import NMDS
    >>> import nuee
    >>> species = nuee.datasets.varespec()
    >>> nmds = NMDS(n_components=2, n_init=20)
    >>> result = nmds.fit(species)
    >>> print(f"Stress: {result.stress:.3f}")
    """

    def __init__(self, n_components: int = 2, max_iter: int = 200,
                 n_init: int = 20, eps: float = 1e-3, random_state: Optional[int] = None,
                 dissimilarity: str = "bray", n_jobs: Optional[int] = None):
        """Initialize NMDS with specified parameters."""
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

    This function provides a high-level interface for NMDS ordination, following
    the conventions of the R vegan package's metaMDS function. It automatically
    handles data transformation and uses multiple random starts to find the best
    ordination solution.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Community data matrix with samples in rows and species in columns.
        Values should be non-negative abundances or counts.
    k : int, default=2
        Number of dimensions for the ordination. Common choices are 2 or 3.
    distance : str, default="bray"
        Distance metric to use for calculating dissimilarities.
        See :func:`nuee.vegdist` for available options.
    trymax : int, default=20
        Maximum number of random starts to find the best solution.
        Higher values increase computation time but may find better solutions.
    maxit : int, default=200
        Maximum number of iterations for each random start.
    trace : bool, default=False
        If True, print progress information including stress values.
    autotransform : bool, default=True
        If True, automatically apply square root transformation to abundance data
        (values > 1) to reduce the influence of dominant species.
    wascores : bool, default=True
        If True, calculate weighted average species scores based on site scores
        and species abundances.
    expand : bool, default=True
        If True, expand the result to include additional information.
    **kwargs : dict
        Additional parameters passed to the NMDS class.

    Returns
    -------
    OrdinationResult
        Result object containing:
        - points : pd.DataFrame
            Site (sample) scores in the ordination space
        - species : pd.DataFrame, optional
            Species scores (if wascores=True)
        - stress : float
            Final stress value (lower is better)
        - converged : bool
            Whether the solution converged

    Notes
    -----
    NMDS is a rank-based ordination method that attempts to preserve the rank
    order of dissimilarities between samples. Unlike metric methods like PCA,
    NMDS makes no assumptions about the linearity of relationships.

    Stress values provide a measure of fit:
    - < 0.05: excellent
    - 0.05 - 0.10: good
    - 0.10 - 0.20: acceptable
    - > 0.20: poor (consider increasing k or using a different method)

    The function uses multiple random starts (trymax) because NMDS can get stuck
    in local optima. The solution with the lowest stress is returned.

    Examples
    --------
    Basic NMDS ordination:

    >>> import nuee
    >>> species = nuee.datasets.varespec()
    >>> nmds = nuee.metaMDS(species, k=2, distance="bray")
    >>> print(f"Stress: {nmds.stress:.3f}")

    With custom parameters:

    >>> nmds = nuee.metaMDS(species, k=3, distance="bray", trymax=50, trace=True)

    Visualize the ordination:

    >>> import matplotlib.pyplot as plt
    >>> fig = nuee.plot_ordination(nmds, display="sites")
    >>> plt.show()

    See Also
    --------
    NMDS : Lower-level NMDS class
    rda : Redundancy Analysis (constrained ordination)
    cca : Canonical Correspondence Analysis
    pca : Principal Component Analysis

    References
    ----------
    .. [1] Kruskal, J.B. (1964). Nonmetric multidimensional scaling: a numerical
           method. Psychometrika 29, 115-129.
    .. [2] Minchin, P.R. (1987). An evaluation of relative robustness of techniques
           for ecological ordinations. Vegetatio 69, 89-107.
    .. [3] Oksanen, J., et al. (2020). vegan: Community Ecology Package.
           https://CRAN.R-project.org/package=vegan
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