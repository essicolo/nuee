"""
Core diversity indices and functions.

This module implements the main diversity indices used in community ecology,
following the conventions of the R nuee package.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from scipy.special import gammaln, digamma
from scipy.optimize import brentq


def _uniroot_r_style(func, lower: float, upper: float,
                     tol: float = np.finfo(float).eps ** 0.25,
                     maxiter: int = 1000) -> float:
    """Port of R's uniroot (Brent's method with R tolerances)."""
    a = float(lower)
    b = float(upper)
    fa = func(a)
    fb = func(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Function returned NaN in root finding")
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Root not bracketed for Fisher alpha calculation")

    c = a
    fc = fa
    d = e = b - a
    eps = np.finfo(float).eps

    for _ in range(maxiter):
        if abs(fc) < abs(fb):
            a, b = b, c
            c = a
            fa, fb = fb, fc
            fc = fa

        tol_act = 2 * eps * abs(b) + tol / 2
        m = 0.5 * (c - b)

        if abs(m) <= tol_act or fb == 0.0:
            return b

        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = m
            e = m
        else:
            s = fb / fa
            if a == c:
                p = 2 * m * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2 * p < (min1 if min1 < min2 else min2):
                e = d
                d = p / q
            else:
                d = m
                e = m

        a = b
        fa = fb
        if abs(d) > tol_act:
            b += d
        else:
            b += tol_act if m > 0 else -tol_act
        fb = func(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
            d = b - a
            e = d

    raise RuntimeError("uniroot did not converge")
import warnings

from .base import DiversityResult


def diversity(x: Union[np.ndarray, pd.DataFrame],
              index: str = "shannon",
              groups: Optional[Union[np.ndarray, pd.Series]] = None,
              base: float = np.e) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate diversity indices for community data.

    This function provides a unified interface for calculating various diversity
    indices commonly used in ecology. It can calculate diversity for individual
    samples or for pooled groups.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Community data matrix with samples in rows and species in columns.
        Values should be non-negative abundances or counts. Can also be a
        1D array for a single sample.
    index : {'shannon', 'simpson', 'invsimpson', 'fisher'}, default='shannon'
        Diversity index to calculate:
        - 'shannon': Shannon entropy H' = -sum(p_i * log(p_i))
        - 'simpson': Gini-Simpson index 1 - sum(p_i^2)
        - 'invsimpson': Inverse Simpson 1 / sum(p_i^2)
        - 'fisher': Fisher's alpha
    groups : np.ndarray or pd.Series, optional
        Grouping factor for calculating pooled diversities. If provided,
        samples are pooled within each group before calculating diversity.
    base : float, default=e
        Base of logarithm for Shannon index. Common choices:
        - e (natural log): nats
        - 2: bits
        - 10: dits

    Returns
    -------
    float, np.ndarray, or pd.Series
        Diversity values for each sample (or group if groups is provided).
        If input is a DataFrame, returns a pd.Series with sample/group names.

    Notes
    -----
    Shannon diversity (H') measures both richness and evenness:
    - Higher values indicate more diverse communities
    - Ranges from 0 (single species) to log(S) where S is species richness
    - Most common diversity index in ecology

    Simpson's index measures dominance:
    - We report the Gini-Simpson form (1 - sum(p_i^2)), matching vegan::diversity
    - Larger values indicate greater diversity
    - The inverse Simpson (1 / sum(p_i^2)) is available via ``index='invsimpson'``

    Fisher's alpha assumes a log-series distribution:
    - Useful for abundance data
    - Less sensitive to sample size than richness
    - Can be slow for large datasets

    Examples
    --------
    Calculate Shannon diversity:

    >>> import nuee
    >>> species = nuee.datasets.varespec()
    >>> div = nuee.diversity(species, index="shannon")
    >>> print(f"Mean diversity: {div.mean():.3f}")

    Calculate Simpson diversity:

    >>> div_simp = nuee.diversity(species, index="simpson")

    Calculate diversity for grouped samples:

    >>> import numpy as np
    >>> groups = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    >>> div_grouped = nuee.diversity(species[:6], index="shannon", groups=groups)

    Use different logarithm bases:

    >>> div_bits = nuee.diversity(species, index="shannon", base=2)
    >>> print("Diversity in bits:", div_bits.mean())

    See Also
    --------
    shannon : Shannon diversity (convenience function)
    simpson : Simpson diversity (convenience function)
    fisher_alpha : Fisher's alpha (convenience function)
    renyi : Renyi entropy for multiple scales
    specnumber : Species richness

    References
    ----------
    .. [1] Shannon, C.E. (1948). A mathematical theory of communication.
           Bell System Technical Journal 27, 379-423.
    .. [2] Simpson, E.H. (1949). Measurement of diversity.
           Nature 163, 688.
    .. [3] Fisher, R.A., Corbet, A.S., Williams, C.B. (1943). The relation between
           the number of species and the number of individuals in a random sample
           of an animal population. Journal of Animal Ecology 12, 42-58.
    """
    # Convert input to appropriate format
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index
    else:
        x_array = np.asarray(x)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        sample_names = None
    
    # Validate data
    if np.any(x_array < 0):
        raise ValueError("Community data cannot contain negative values")
    
    # Handle grouping
    if groups is not None:
        return _grouped_diversity(x_array, index, groups, base, sample_names)
    
    # Calculate diversity index
    if index.lower() == "shannon":
        result = _shannon_diversity(x_array, base)
    elif index.lower() == "simpson":
        result = _simpson_diversity(x_array)
    elif index.lower() == "invsimpson":
        result = _inverse_simpson_diversity(x_array)
    elif index.lower() == "fisher":
        result = _fisher_alpha(x_array)
    else:
        raise ValueError(f"Unknown diversity index: {index}")
    
    # Return as pandas Series if input was DataFrame
    if isinstance(x, pd.DataFrame) and sample_names is not None:
        return pd.Series(result, index=sample_names, name=f"{index}_diversity")
    
    return result.squeeze() if result.size == 1 else result


def shannon(x: Union[np.ndarray, pd.DataFrame], 
           base: float = np.e) -> DiversityResult:
    """
    Calculate Shannon diversity index.
    
    H = -sum(p_i * log(p_i))
    
    Parameters:
        x: Community data matrix or vector
        base: Base of logarithm
        
    Returns:
        DiversityResult object with automatic plotting
    """
    result = diversity(x, index="shannon", base=base)
    sample_names = x.index.tolist() if isinstance(x, pd.DataFrame) else None
    return DiversityResult(result, index_name="Shannon", sample_names=sample_names)


def simpson(x: Union[np.ndarray, pd.DataFrame]) -> DiversityResult:
    """
    Calculate Gini-Simpson diversity (1 - sum(p_i^2)).
    
    Parameters:
        x: Community data matrix or vector
        
    Returns:
        DiversityResult object with automatic plotting
    """
    result = diversity(x, index="simpson")
    sample_names = x.index.tolist() if isinstance(x, pd.DataFrame) else None
    return DiversityResult(result, index_name="Simpson", sample_names=sample_names)


def fisher_alpha(x: Union[np.ndarray, pd.DataFrame]) -> DiversityResult:
    """
    Calculate Fisher's alpha diversity index.
    
    Parameters:
        x: Community data matrix or vector
        
    Returns:
        DiversityResult object with automatic plotting
    """
    result = diversity(x, index="fisher")
    sample_names = x.index.tolist() if isinstance(x, pd.DataFrame) else None
    return DiversityResult(result, index_name="Fisher's Alpha", sample_names=sample_names)


def specnumber(x: Union[np.ndarray, pd.DataFrame],
               groups: Optional[Union[np.ndarray, pd.Series]] = None) -> DiversityResult:
    """
    Calculate species richness (number of species).
    
    Parameters:
        x: Community data matrix or vector
        groups: Grouping factor for pooled richness
        
    Returns:
        DiversityResult object with automatic plotting
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index.tolist()
    else:
        x_array = np.asarray(x)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        sample_names = None
    
    # Handle grouping
    if groups is not None:
        result = _grouped_specnumber(x_array, groups, sample_names)
        return DiversityResult(result, index_name="Species Richness", sample_names=result.index.tolist())
    
    # Calculate species richness
    richness = np.sum(x_array > 0, axis=1)
    
    return DiversityResult(richness, index_name="Species Richness", sample_names=sample_names)


def evenness(x: Union[np.ndarray, pd.DataFrame],
             method: str = "pielou") -> DiversityResult:
    """
    Calculate evenness indices.
    
    Parameters:
        x: Community data matrix or vector
        method: Evenness method ("pielou", "simpson", "evar")
        
    Returns:
        DiversityResult object with automatic plotting
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index.tolist()
    else:
        x_array = np.asarray(x)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        sample_names = None
    
    if method.lower() == "pielou":
        # Pielou's evenness: H / log(S)
        H = _shannon_diversity(x_array, np.e)
        S = np.sum(x_array > 0, axis=1)
        evenness_vals = H / np.log(S)
        evenness_vals[S <= 1] = 0  # Handle single-species communities
        
    elif method.lower() == "simpson":
        # Simpson's evenness: (1/D) / S
        D = _simpson_dominance(x_array)
        S = np.sum(x_array > 0, axis=1)
        evenness_vals = (1 / D) / S
        evenness_vals[S <= 1] = 0
        
    elif method.lower() == "evar":
        # Smith and Wilson's evenness
        evenness_vals = _evar_evenness(x_array)
        
    else:
        raise ValueError(f"Unknown evenness method: {method}")
    
    return DiversityResult(evenness_vals, index_name=f"{method.title()} Evenness", sample_names=sample_names)


def renyi(x: Union[np.ndarray, pd.DataFrame],
          scales: Union[float, List[float]] = [0, 1, 2, np.inf],
          hill: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """
    Calculate Renyi entropy or Hill numbers.
    
    Parameters:
        x: Community data matrix or vector
        scales: Scale parameters (alpha values)
        hill: Whether to return Hill numbers instead of Renyi entropy
        
    Returns:
        Renyi entropy or Hill numbers for each scale
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
        sample_names = x.index
    else:
        x_array = np.asarray(x)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        sample_names = None
    
    # Ensure scales is a list
    if not isinstance(scales, (list, tuple, np.ndarray)):
        scales = [scales]
    
    # Calculate relative abundances
    totals = np.sum(x_array, axis=1, keepdims=True)
    totals[totals == 0] = 1  # Avoid division by zero
    p = x_array / totals
    
    # Calculate Renyi entropy for each scale
    results = []
    for alpha in scales:
        if alpha == 0:
            # Species richness
            renyi_vals = np.log(np.sum(p > 0, axis=1))
        elif alpha == 1:
            # Shannon entropy (limit as alpha approaches 1)
            renyi_vals = -np.sum(p * np.log(p + 1e-10), axis=1)
        elif alpha == np.inf:
            # Max entropy
            renyi_vals = -np.log(np.max(p, axis=1))
        else:
            # General case
            renyi_vals = (1 / (1 - alpha)) * np.log(np.sum(p**alpha, axis=1))
        
        if hill:
            # Convert to Hill numbers
            renyi_vals = np.exp(renyi_vals)
            
        results.append(renyi_vals)
    
    results = np.array(results).T
    
    # Return as DataFrame if input was DataFrame
    if isinstance(x, pd.DataFrame) and sample_names is not None:
        scale_names = [f"alpha_{alpha}" for alpha in scales]
        return pd.DataFrame(results, index=sample_names, columns=scale_names)
    
    return results.squeeze() if results.size == 1 else results


# Helper functions
def _shannon_diversity(x: np.ndarray, base: float = np.e) -> np.ndarray:
    """Calculate Shannon diversity index."""
    # Calculate relative abundances
    totals = np.sum(x, axis=1, keepdims=True)
    totals[totals == 0] = 1  # Avoid division by zero
    p = x / totals
    
    # Calculate Shannon entropy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_p = np.log(p + 1e-10) / np.log(base)  # Add small constant to avoid log(0)
        shannon = -np.sum(p * log_p, axis=1)
    
    return shannon


def _simpson_diversity(x: np.ndarray) -> np.ndarray:
    """Calculate Gini-Simpson diversity (1 - sum(p_i^2))."""
    dominance = _simpson_dominance(x)
    return 1.0 - dominance


def _inverse_simpson_diversity(x: np.ndarray) -> np.ndarray:
    """Calculate inverse Simpson diversity (1 / sum(p_i^2))."""
    dominance = _simpson_dominance(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dominance
    inv[~np.isfinite(inv)] = 0.0
    return inv


def _simpson_dominance(x: np.ndarray) -> np.ndarray:
    """Return Simpson dominance sum(p_i^2)."""
    totals = np.sum(x, axis=1, keepdims=True)
    totals[totals == 0] = 1
    p = x / totals
    return np.sum(p**2, axis=1)


def _fisher_alpha(x: np.ndarray) -> np.ndarray:
    """Calculate Fisher's alpha diversity index."""
    n_samples = x.shape[0]
    alpha_values = np.zeros(n_samples)

    for i in range(n_samples):
        raw = x[i, :]
        counts = np.rint(raw).astype(int)
        counts = counts[counts > 0]

        if counts.size == 0:
            alpha_values[i] = 0.0
            continue

        N = int(counts.sum())
        S = int(counts.size)
        if S <= 1 or N <= 1:
            alpha_values[i] = 0.0
            continue

        target = float(S)

        def equation(alpha: float) -> float:
            return alpha * np.log1p(N / alpha) - target

        lower = 0.1
        upper = 50.0
        f_lower = equation(lower)
        f_upper = equation(upper)
        expansion = 0
        while f_lower * f_upper > 0 and expansion < 50:
            upper *= 2.0
            f_upper = equation(upper)
            expansion += 1
        if f_lower * f_upper > 0:
            alpha_values[i] = max(S / np.log(max(N, 2)), 0.0)
            continue

        try:
            alpha_values[i] = _uniroot_r_style(equation, lower, upper)
        except (ValueError, RuntimeError):
            alpha_values[i] = max(S / np.log(max(N, 2)), 0.0)

    return alpha_values


def _evar_evenness(x: np.ndarray) -> np.ndarray:
    """Calculate Smith and Wilson's evenness index."""
    n_samples = x.shape[0]
    evenness_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        sample = x[i, :]
        sample = sample[sample > 0]  # Remove zeros
        
        if len(sample) <= 1:
            evenness_vals[i] = 0
            continue
        
        # Calculate relative abundances
        p = sample / np.sum(sample)
        
        # Calculate expected variance under evenness
        S = len(sample)
        expected_var = (1 - 1/S) / S
        
        # Calculate actual variance
        actual_var = np.var(p, ddof=1)
        
        # Calculate evenness
        evenness_vals[i] = 1 - 2 * actual_var / expected_var
    
    return evenness_vals


def _grouped_diversity(x: np.ndarray, index: str, groups: np.ndarray, 
                      base: float, sample_names: Optional[List[str]]) -> pd.Series:
    """Calculate diversity for grouped data."""
    if isinstance(groups, pd.Series):
        groups = groups.values
    
    unique_groups = np.unique(groups)
    results = {}
    
    for group in unique_groups:
        group_mask = groups == group
        group_data = x[group_mask, :]
        
        # Pool data across samples in group
        pooled_data = np.sum(group_data, axis=0).reshape(1, -1)
        
        # Calculate diversity
        if index.lower() == "shannon":
            div_val = _shannon_diversity(pooled_data, base)[0]
        elif index.lower() == "simpson":
            div_val = _simpson_diversity(pooled_data)[0]
        elif index.lower() == "invsimpson":
            div_val = _inverse_simpson_diversity(pooled_data)[0]
        elif index.lower() == "fisher":
            div_val = _fisher_alpha(pooled_data)[0]
        
        results[group] = div_val
    
    return pd.Series(results, name=f"{index}_diversity")


def _grouped_specnumber(x: np.ndarray, groups: np.ndarray, 
                       sample_names: Optional[List[str]]) -> pd.Series:
    """Calculate species richness for grouped data."""
    if isinstance(groups, pd.Series):
        groups = groups.values
    
    unique_groups = np.unique(groups)
    results = {}
    
    for group in unique_groups:
        group_mask = groups == group
        group_data = x[group_mask, :]
        
        # Pool data across samples in group
        pooled_data = np.sum(group_data, axis=0)
        
        # Calculate species richness
        richness = np.sum(pooled_data > 0)
        results[group] = richness
    
    return pd.Series(results, name="specnumber")
