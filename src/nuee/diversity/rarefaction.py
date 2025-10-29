"""
Rarefaction and species accumulation functions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
from scipy.special import comb


def rarefy(x: Union[np.ndarray, pd.DataFrame], 
           sample: int) -> Union[np.ndarray, pd.Series]:
    """
    Rarefy species richness to a standard sample size.
    
    Parameters:
        x: Community data matrix
        sample: Sample size for rarefaction
        
    Returns:
        Rarefied species richness
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index
    else:
        x_array = np.asarray(x)
        sample_names = None
    
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)
    
    rarefied = np.zeros(x_array.shape[0])
    
    for i in range(x_array.shape[0]):
        abundances = x_array[i, x_array[i] > 0]
        total = np.sum(abundances)
        
        if total >= sample:
            # Calculate rarefied richness
            prob_present = 1 - comb(total - abundances, sample) / comb(total, sample)
            rarefied[i] = np.sum(prob_present)
        else:
            rarefied[i] = len(abundances)
    
    if isinstance(x, pd.DataFrame) and sample_names is not None:
        return pd.Series(rarefied, index=sample_names, name=f"rarefied_{sample}")
    
    return rarefied


def rarecurve(x: Union[np.ndarray, pd.DataFrame],
              step: int = 1,
              sample: Optional[int] = None) -> dict:
    """
    Calculate rarefaction curves.
    
    Parameters:
        x: Community data matrix
        step: Step size for rarefaction
        sample: Maximum sample size
        
    Returns:
        Dictionary with rarefaction curves
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index.tolist()
    else:
        x_array = np.asarray(x)
        sample_names = [f"Sample{i+1}" for i in range(x_array.shape[0])]
    
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)
    
    curves = {}
    
    for i in range(x_array.shape[0]):
        row = x_array[i]
        total = int(np.sum(row))
        
        if sample is None:
            max_sample = total
        else:
            max_sample = min(sample, total)
        
        if max_sample < step:
            continue
            
        sample_sizes = np.arange(step, max_sample + 1, step)
        richness = np.zeros(len(sample_sizes))
        
        for j, n in enumerate(sample_sizes):
            richness[j] = rarefy(row.reshape(1, -1), n)[0]
        
        curves[sample_names[i]] = {
            'sample_sizes': sample_sizes,
            'richness': richness
        }
    
    return curves


def estimateR(x: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Estimate species richness using various estimators.
    
    Parameters:
        x: Community data matrix
        
    Returns:
        DataFrame with richness estimates
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index
    else:
        x_array = np.asarray(x)
        sample_names = [f"Sample{i+1}" for i in range(x_array.shape[0])]
    
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)
    
    results = []
    
    for i in range(x_array.shape[0]):
        row = x_array[i]
        
        # Observed richness
        S_obs = np.sum(row > 0)
        
        # Singletons and doubletons
        f1 = np.sum(row == 1)
        f2 = np.sum(row == 2)
        
        # Chao1 estimator
        if f2 > 0:
            chao1 = S_obs + (f1**2) / (2 * f2)
        else:
            chao1 = S_obs + f1 * (f1 - 1) / 2
        
        # ACE estimator (simplified)
        ace = S_obs + f1 / f2 if f2 > 0 else S_obs
        
        results.append({
            'S.obs': S_obs,
            'S.chao1': chao1,
            'S.ACE': ace,
            'f1': f1,
            'f2': f2
        })
    
    return pd.DataFrame(results, index=sample_names)