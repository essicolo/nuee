"""
Plotting functions for dissimilarity analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional


def plot_dissimilarity(distance_matrix: Union[np.ndarray, pd.DataFrame],
                       figsize: tuple = (8, 6),
                       **kwargs) -> plt.Figure:
    """
    Plot dissimilarity matrix as heatmap.
    
    Parameters:
        distance_matrix: Distance/dissimilarity matrix
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(distance_matrix, pd.DataFrame):
        sns.heatmap(distance_matrix, ax=ax, **kwargs)
    else:
        sns.heatmap(distance_matrix, ax=ax, **kwargs)
    
    ax.set_title('Dissimilarity Matrix')
    plt.tight_layout()
    return fig


def plot_betadisper(betadisper_result: dict,
                    figsize: tuple = (8, 6),
                    **kwargs) -> plt.Figure:
    """
    Plot beta dispersion results.
    
    Parameters:
        betadisper_result: Beta dispersion results
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Placeholder implementation
    ax.text(0.5, 0.5, 'Beta Dispersion Plot\n(Implementation pending)', 
            ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('Beta Dispersion')
    plt.tight_layout()
    return fig