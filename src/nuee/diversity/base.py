"""
Base classes for diversity analysis results.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union


class DiversityResult:
    """
    Container for diversity analysis results with automatic plotting.
    """
    
    def __init__(self, values: Union[np.ndarray, pd.Series], 
                 index_name: str = "diversity",
                 sample_names: Optional[list] = None):
        self.values = values
        self.index_name = index_name
        self.sample_names = sample_names
        
        if isinstance(values, pd.Series):
            self.sample_names = values.index.tolist()
            self.values = values.values
    
    def __repr__(self):
        return f"DiversityResult({self.index_name}, n={len(self.values)})"
    
    def __getitem__(self, key):
        """Allow indexing like an array."""
        return self.values[key]
    
    def __len__(self):
        """Return length."""
        return len(self.values)
    
    def mean(self):
        """Calculate mean diversity."""
        return np.mean(self.values)
    
    def std(self):
        """Calculate standard deviation."""
        return np.std(self.values)
    
    def summary(self):
        """Return summary statistics."""
        return {
            'count': len(self.values),
            'mean': np.mean(self.values),
            'std': np.std(self.values),
            'min': np.min(self.values),
            'max': np.max(self.values),
            'median': np.median(self.values)
        }
    
    def plot(self, kind="hist", **kwargs):
        """
        Plot the diversity values.
        
        Parameters:
            kind: Plot type ("hist", "box", "bar", "violin")
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        from ..plotting.diversity_plots import plot_diversity_result
        return plot_diversity_result(self, kind=kind, **kwargs)


class RarefactionResult:
    """
    Container for rarefaction analysis results with automatic plotting.
    """
    
    def __init__(self, curves: Dict[str, Dict[str, np.ndarray]]):
        self.curves = curves
        self.sample_names = list(curves.keys())
    
    def __repr__(self):
        return f"RarefactionResult(samples={len(self.sample_names)})"
    
    def plot(self, samples=None, **kwargs):
        """
        Plot rarefaction curves.
        
        Parameters:
            samples: Which samples to plot (default: all)
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        from ..plotting.diversity_plots import plot_rarecurve
        return plot_rarecurve(self.curves, samples=samples, **kwargs)


class AccumulationResult:
    """
    Container for species accumulation results with automatic plotting.
    """
    
    def __init__(self, sites: np.ndarray, richness: np.ndarray, 
                 sd: Optional[np.ndarray] = None, method: str = "random"):
        self.sites = sites
        self.richness = richness
        self.sd = sd
        self.method = method
    
    def __repr__(self):
        return f"AccumulationResult(method={self.method}, sites={len(self.sites)})"
    
    def plot(self, **kwargs):
        """
        Plot species accumulation curve.
        
        Parameters:
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        from ..plotting.diversity_plots import plot_specaccum
        return plot_specaccum({
            'sites': self.sites,
            'richness': self.richness,
            'sd': self.sd,
            'method': self.method
        }, **kwargs)