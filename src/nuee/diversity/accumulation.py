"""
Species accumulation curves.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def specaccum(x: Union[np.ndarray, pd.DataFrame], 
              method: str = "random", 
              permutations: int = 100) -> dict:
    """
    Calculate species accumulation curves.
    
    Parameters:
        x: Community data matrix
        method: Accumulation method ("random", "exact", "rarefaction")
        permutations: Number of permutations for random method
        
    Returns:
        Dictionary with accumulation results
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
    else:
        x_array = np.asarray(x)
    
    n_sites = x_array.shape[0]
    sites = np.arange(1, n_sites + 1)
    
    if method == "random":
        richness_curves = []
        
        for _ in range(permutations):
            # Random permutation of sites
            site_order = np.random.permutation(n_sites)
            
            # Calculate cumulative richness
            cumulative_richness = []
            seen_species = np.zeros(x_array.shape[1], dtype=bool)
            
            for i in range(n_sites):
                site_idx = site_order[i]
                seen_species |= (x_array[site_idx] > 0)
                cumulative_richness.append(np.sum(seen_species))
            
            richness_curves.append(cumulative_richness)
        
        # Calculate mean and standard deviation
        richness_curves = np.array(richness_curves)
        mean_richness = np.mean(richness_curves, axis=0)
        sd_richness = np.std(richness_curves, axis=0)
        
        return {
            'sites': sites,
            'richness': mean_richness,
            'sd': sd_richness,
            'method': method,
            'permutations': permutations
        }
    
    else:
        # Exact method
        richness = []
        seen_species = np.zeros(x_array.shape[1], dtype=bool)
        
        for i in range(n_sites):
            seen_species |= (x_array[i] > 0)
            richness.append(np.sum(seen_species))
        
        return {
            'sites': sites,
            'richness': np.array(richness),
            'method': method
        }


def poolaccum(x: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Calculate pooled species accumulation.
    
    Parameters:
        x: Community data matrix
        
    Returns:
        Dictionary with pooled accumulation results
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
    else:
        x_array = np.asarray(x)
    
    n_sites = x_array.shape[0]
    sites = np.arange(1, n_sites + 1)
    
    # Calculate pooled richness
    pooled_richness = []
    pooled_data = np.zeros(x_array.shape[1])
    
    for i in range(n_sites):
        pooled_data += x_array[i]
        richness = np.sum(pooled_data > 0)
        pooled_richness.append(richness)
    
    return {
        'sites': sites,
        'richness': np.array(pooled_richness),
        'method': 'pooled'
    }