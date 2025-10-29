"""
Sample datasets for community ecology analysis.

This module provides sample datasets that can be used for testing
and demonstrating nuee functionality.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def varespec() -> pd.DataFrame:
    """
    Lichen species data from Finnish forest.
    
    Returns:
        DataFrame with 24 sites and 44 lichen species
    """
    # Generate synthetic data similar to varespec
    np.random.seed(42)
    n_sites = 24
    n_species = 44
    
    # Create species names
    species_names = [f"Lichen_sp_{i+1}" for i in range(n_species)]
    site_names = [f"Site_{i+1}" for i in range(n_sites)]
    
    # Generate data with some structure
    data = np.random.poisson(3, size=(n_sites, n_species))
    
    # Add some zeros to make it more realistic
    zero_mask = np.random.random((n_sites, n_species)) < 0.4
    data[zero_mask] = 0
    
    return pd.DataFrame(data, index=site_names, columns=species_names)


def varechem() -> pd.DataFrame:
    """
    Environmental data corresponding to varespec.
    
    Returns:
        DataFrame with 24 sites and environmental variables
    """
    np.random.seed(42)
    n_sites = 24
    site_names = [f"Site_{i+1}" for i in range(n_sites)]
    
    # Generate correlated environmental variables
    pH = np.random.normal(4.5, 0.8, n_sites)
    N = np.random.normal(15, 5, n_sites) + pH * 2
    P = np.random.normal(35, 10, n_sites) + pH * 3
    K = np.random.normal(85, 20, n_sites)
    Ca = np.random.normal(400, 150, n_sites) + pH * 50
    Mg = np.random.normal(120, 40, n_sites) + pH * 10
    
    # Ensure positive values
    N = np.maximum(N, 1)
    P = np.maximum(P, 1)
    K = np.maximum(K, 1)
    Ca = np.maximum(Ca, 1)
    Mg = np.maximum(Mg, 1)
    
    return pd.DataFrame({
        'pH': pH,
        'N': N,
        'P': P,
        'K': K,
        'Ca': Ca,
        'Mg': Mg
    }, index=site_names)


def dune() -> pd.DataFrame:
    """
    Dutch dune meadow vegetation data.
    
    Returns:
        DataFrame with 20 sites and 30 species
    """
    np.random.seed(123)
    n_sites = 20
    n_species = 30
    
    species_names = [f"Species_{i+1}" for i in range(n_species)]
    site_names = [f"Dune_{i+1}" for i in range(n_sites)]
    
    # Generate data with abundance scale 1-5
    data = np.random.choice([0, 1, 2, 3, 4, 5], 
                           size=(n_sites, n_species), 
                           p=[0.4, 0.2, 0.2, 0.1, 0.07, 0.03])
    
    return pd.DataFrame(data, index=site_names, columns=species_names)


def dune_env() -> pd.DataFrame:
    """
    Environmental data corresponding to dune.
    
    Returns:
        DataFrame with 20 sites and environmental variables
    """
    np.random.seed(123)
    n_sites = 20
    site_names = [f"Dune_{i+1}" for i in range(n_sites)]
    
    # Generate environmental variables
    moisture = np.random.choice([1, 2, 3, 4, 5], n_sites, p=[0.2, 0.3, 0.3, 0.15, 0.05])
    management = np.random.choice(['BF', 'HF', 'NM', 'SF'], n_sites, p=[0.25, 0.25, 0.25, 0.25])
    use = np.random.choice(['Hayfield', 'Pasture', 'Haypastu', 'Natural'], n_sites, p=[0.3, 0.3, 0.2, 0.2])
    manure = np.random.choice([0, 1, 2, 3, 4], n_sites, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    
    return pd.DataFrame({
        'A1': moisture,
        'Management': management,
        'Use': use,
        'Manure': manure
    }, index=site_names)


def BCI() -> pd.DataFrame:
    """
    Barro Colorado Island tree data.
    
    Returns:
        DataFrame with 50 sites and 225 species
    """
    np.random.seed(456)
    n_sites = 50
    n_species = 225
    
    species_names = [f"Tree_sp_{i+1}" for i in range(n_species)]
    site_names = [f"Plot_{i+1}" for i in range(n_sites)]
    
    # Generate data with many zeros (typical for tree data)
    data = np.random.poisson(1.5, size=(n_sites, n_species))
    
    # Add more zeros to make it realistic
    zero_mask = np.random.random((n_sites, n_species)) < 0.7
    data[zero_mask] = 0
    
    return pd.DataFrame(data, index=site_names, columns=species_names)


def mite() -> pd.DataFrame:
    """
    Oribatid mite data.
    
    Returns:
        DataFrame with 70 sites and 35 species
    """
    np.random.seed(789)
    n_sites = 70
    n_species = 35
    
    species_names = [f"Mite_sp_{i+1}" for i in range(n_species)]
    site_names = [f"Sample_{i+1}" for i in range(n_sites)]
    
    # Generate abundance data
    data = np.random.poisson(2, size=(n_sites, n_species))
    
    # Add zeros
    zero_mask = np.random.random((n_sites, n_species)) < 0.5
    data[zero_mask] = 0
    
    return pd.DataFrame(data, index=site_names, columns=species_names)


def mite_env() -> pd.DataFrame:
    """
    Environmental data corresponding to mite.
    
    Returns:
        DataFrame with 70 sites and environmental variables
    """
    np.random.seed(789)
    n_sites = 70
    site_names = [f"Sample_{i+1}" for i in range(n_sites)]
    
    # Generate environmental variables
    substrate = np.random.choice(['Sphagnum', 'Litter', 'Barepeat', 'Interface'], 
                                n_sites, p=[0.3, 0.3, 0.2, 0.2])
    shrub = np.random.choice(['None', 'Few', 'Many'], n_sites, p=[0.4, 0.4, 0.2])
    topo = np.random.choice(['Blanket', 'Hummock'], n_sites, p=[0.6, 0.4])
    
    water_content = np.random.normal(350, 100, n_sites)
    density = np.random.normal(0.3, 0.1, n_sites)
    
    # Ensure positive values
    water_content = np.maximum(water_content, 50)
    density = np.maximum(density, 0.1)
    
    return pd.DataFrame({
        'SubsDens': density,
        'WatrCont': water_content,
        'Substrate': substrate,
        'Shrub': shrub,
        'Topo': topo
    }, index=site_names)


def load_example_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load example species and environmental data for quick testing.
    
    Returns:
        Tuple of (species_data, environmental_data)
    """
    return varespec(), varechem()