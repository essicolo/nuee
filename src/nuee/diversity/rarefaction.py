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


#: Abundance at or below which a species counts as "rare" for ACE.
ACE_RARE_THRESHOLD = 10


def _ace_estimator(row: np.ndarray, s_obs: int, f1: int) -> float:
    """Abundance-based Coverage Estimator of richness, as vegan computes it.

    Species with an abundance of at most :data:`ACE_RARE_THRESHOLD` inform the
    sample coverage; the abundant ones are taken as fully observed. The
    coefficient of variation of the rare abundances corrects for uneven
    abundance among them.
    """
    rare = row[(row > 0) & (row <= ACE_RARE_THRESHOLD)]
    n_abundant = int(np.sum(row > ACE_RARE_THRESHOLD))
    n_rare_species = int(rare.size)
    n_rare_individuals = int(rare.sum())

    if n_rare_species == 0 or n_rare_individuals == 0:
        return float(n_abundant)

    coverage = 1.0 - f1 / n_rare_individuals
    if coverage <= 0:
        # Every rare species is a singleton: coverage is zero and the
        # estimator is undefined, so fall back on the observed richness.
        return float(s_obs)

    counts = np.arange(1, ACE_RARE_THRESHOLD + 1)
    frequencies = np.array([np.sum(row == count) for count in counts], dtype=float)

    variation = 0.0
    if n_rare_individuals > 1:
        variation = max(
            n_rare_species / coverage
            * float(np.sum(counts * (counts - 1) * frequencies))
            / (n_rare_individuals * (n_rare_individuals - 1))
            - 1.0,
            0.0,
        )

    return float(n_abundant + n_rare_species / coverage + f1 / coverage * variation)


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
        S_obs = int(np.sum(row > 0))

        # Singletons and doubletons
        f1 = int(np.sum(row == 1))
        f2 = int(np.sum(row == 2))

        # Bias-corrected Chao1, the form vegan reports. The uncorrected
        # f1^2 / (2 * f2) is undefined without doubletons and overestimates
        # when they are few.
        chao1 = S_obs + f1 * (f1 - 1) / 2.0 / (f2 + 1)

        ace = _ace_estimator(row, S_obs, f1)

        results.append({
            'S.obs': S_obs,
            'S.chao1': chao1,
            'S.ACE': ace,
            'f1': f1,
            'f2': f2
        })
    
    return pd.DataFrame(results, index=sample_names)