"""
Base classes and utilities for ordination methods.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, Tuple


class OrdinationResult:
    """
    Base class for ordination results.
    
    Attributes:
        points: Sample coordinates in ordination space
        species: Species coordinates (if applicable)
        eigenvalues: Eigenvalues for ordination axes
        stress: Stress value (for NMDS)
        converged: Whether the ordination converged
        nobj: Number of objects (samples)
        ndim: Number of dimensions
        call: Dictionary containing method call parameters
    """
    
    def __init__(self, points: np.ndarray, species: Optional[np.ndarray] = None,
                 eigenvalues: Optional[np.ndarray] = None, stress: Optional[float] = None,
                 converged: bool = True, call: Optional[Dict[str, Any]] = None):
        self.points = points
        self.species = species
        self.eigenvalues = eigenvalues
        self.stress = stress
        self.converged = converged
        self.nobj = points.shape[0]
        self.ndim = points.shape[1]
        self.call = call or {}
        
    def __repr__(self):
        return f"OrdinationResult(nobj={self.nobj}, ndim={self.ndim})"
    
    def plot(self, axes=(0, 1), display="sites", type="points", **kwargs):
        """
        Plot the ordination result.
        
        Parameters:
            axes: Which axes to plot (tuple of axis indices)
            display: What to display ("sites", "species", "both")
            type: Plot type ("points", "text", "none")
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        from ..plotting.ordination_plots import plot_ordination
        return plot_ordination(self, axes=axes, display=display, type=type, **kwargs)
    
    def biplot(self, **kwargs):
        """
        Create a biplot if applicable.
        
        Parameters:
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        if hasattr(self, 'biplot') and self.biplot is not None:
            from ..plotting.ordination_plots import biplot
            return biplot(self, **kwargs)
        else:
            return self.plot(display="both", **kwargs)


class ConstrainedOrdinationResult(OrdinationResult):
    """
    Result class for constrained ordination methods (RDA, CCA).
    
    Additional attributes:
        constrained_eig: Eigenvalues for constrained axes
        unconstrained_eig: Eigenvalues for unconstrained axes  
        biplot: Biplot scores for environmental variables
        centroids: Centroids for factor variables
        envfit_result: Environmental fitting results
        tot_chi: Total inertia
        partial_chi: Partial inertia (if partial ordination)
    """
    
    def __init__(self, points: np.ndarray, species: Optional[np.ndarray] = None,
                 constrained_eig: Optional[np.ndarray] = None,
                 unconstrained_eig: Optional[np.ndarray] = None,
                 biplot: Optional[np.ndarray] = None,
                 centroids: Optional[np.ndarray] = None,
                 tot_chi: Optional[float] = None,
                 partial_chi: Optional[float] = None,
                 call: Optional[Dict[str, Any]] = None):
        
        # Handle eigenvalues concatenation safely
        eig_parts = []
        if constrained_eig is not None and len(constrained_eig) > 0:
            eig_parts.append(constrained_eig)
        if unconstrained_eig is not None and len(unconstrained_eig) > 0:
            eig_parts.append(unconstrained_eig)
        
        all_eig = np.concatenate(eig_parts) if eig_parts else np.array([])
        super().__init__(points, species, all_eig, call=call)
        
        self.constrained_eig = constrained_eig
        self.unconstrained_eig = unconstrained_eig
        self.biplot = biplot
        self.centroids = centroids
        self.tot_chi = tot_chi
        self.partial_chi = partial_chi
    
    def biplot(self, **kwargs):
        """
        Create a biplot for constrained ordination with environmental arrows.
        
        Parameters:
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        from ..plotting.ordination_plots import biplot
        return biplot(self, **kwargs)
        
    @property
    def rank(self) -> int:
        """Number of constrained axes."""
        return len(self.constrained_eig) if self.constrained_eig is not None else 0
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Proportion of variance explained by each axis."""
        if self.eigenvalues is not None and self.tot_chi is not None:
            return self.eigenvalues / self.tot_chi
        return np.array([])


class OrdinationMethod(ABC):
    """
    Abstract base class for ordination methods.
    """
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> OrdinationResult:
        """
        Fit the ordination method to the data.
        
        Parameters:
            X: Community data matrix (samples x species)
            **kwargs: Method-specific parameters
            
        Returns:
            OrdinationResult object
        """
        pass
    
    def _validate_data(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Parameters:
            X: Input data matrix
            
        Returns:
            Validated numpy array
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array")
            
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values")
            
        if np.any(np.isinf(X)):
            raise ValueError("Input data contains infinite values")
            
        return X
    
    def _center_data(self, X: np.ndarray) -> np.ndarray:
        """Center the data by subtracting column means."""
        return X - np.mean(X, axis=0)
    
    def _standardize_data(self, X: np.ndarray) -> np.ndarray:
        """Standardize the data to unit variance."""
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)