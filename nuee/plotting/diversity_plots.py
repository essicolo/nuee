"""
Plotting functions for diversity analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Any


def plot_diversity(diversity_data: Union[np.ndarray, pd.Series, pd.DataFrame],
                   figsize: tuple = (8, 6),
                   **kwargs) -> plt.Figure:
    """
    Plot diversity indices.
    
    Parameters:
        diversity_data: Diversity values
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(diversity_data, pd.DataFrame):
        diversity_data.plot(kind='bar', ax=ax, **kwargs)
    elif isinstance(diversity_data, pd.Series):
        diversity_data.plot(kind='bar', ax=ax, **kwargs)
    else:
        ax.bar(range(len(diversity_data)), diversity_data, **kwargs)
    
    ax.set_ylabel('Diversity')
    ax.set_title('Diversity Indices')
    plt.tight_layout()
    return fig


def plot_rarecurve(rarecurve_data: Dict[str, Dict[str, np.ndarray]],
                   figsize: tuple = (10, 6),
                   **kwargs) -> plt.Figure:
    """
    Plot rarefaction curves.
    
    Parameters:
        rarecurve_data: Rarefaction curve data
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for sample_name, curve_data in rarecurve_data.items():
        ax.plot(curve_data['sample_sizes'], curve_data['richness'], 
                label=sample_name, **kwargs)
    
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Species Richness')
    ax.set_title('Rarefaction Curves')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_specaccum(specaccum_data: Dict[str, np.ndarray],
                   figsize: tuple = (8, 6),
                   **kwargs) -> plt.Figure:
    """
    Plot species accumulation curves.
    
    Parameters:
        specaccum_data: Species accumulation data
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sites = specaccum_data['sites']
    richness = specaccum_data['richness']
    
    ax.plot(sites, richness, **kwargs)
    
    # Add confidence intervals if available
    if 'sd' in specaccum_data:
        sd = specaccum_data['sd']
        ax.fill_between(sites, richness - sd, richness + sd, 
                       alpha=0.3, **kwargs)
    
    ax.set_xlabel('Sites')
    ax.set_ylabel('Species Richness')
    ax.set_title('Species Accumulation Curve')
    plt.tight_layout()
    return fig


def plot_diversity_result(diversity_result, kind="hist", figsize=(8, 6), **kwargs):
    """
    Plot diversity result with automatic plot type selection.
    
    Parameters:
        diversity_result: DiversityResult object
        kind: Plot type ("hist", "box", "bar", "violin")
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    values = diversity_result.values
    
    if kind == "hist":
        # Set default values but allow override from kwargs
        hist_params = {'bins': 20, 'alpha': 0.7}
        hist_params.update(kwargs)
        ax.hist(values, **hist_params)
        ax.set_xlabel('Diversity Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{diversity_result.index_name.title()} Distribution')
    elif kind == "box":
        ax.boxplot([values], labels=[diversity_result.index_name], **kwargs)
        ax.set_ylabel('Diversity Value')
        ax.set_title(f'{diversity_result.index_name.title()} Box Plot')
    elif kind == "bar":
        if diversity_result.sample_names:
            ax.bar(diversity_result.sample_names, values, **kwargs)
            ax.set_xlabel('Sample')
            if len(diversity_result.sample_names) > 10:
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.bar(range(len(values)), values, **kwargs)
            ax.set_xlabel('Sample Index')
        ax.set_ylabel('Diversity Value')
        ax.set_title(f'{diversity_result.index_name.title()} by Sample')
    elif kind == "violin":
        ax.violinplot([values], positions=[1], **kwargs)
        ax.set_ylabel('Diversity Value')
        ax.set_title(f'{diversity_result.index_name.title()} Violin Plot')
    else:
        raise ValueError(f"Unknown plot kind: {kind}")
    
    plt.tight_layout()
    return fig