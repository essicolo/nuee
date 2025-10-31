"""
Sample datasets for community ecology analysis.

This module provides sample datasets that can be used for testing
and demonstrating nuee functionality.
"""

from __future__ import annotations

import importlib.resources as resources
from typing import Tuple

import pandas as pd


def _load_csv(name: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file bundled with the package.

    Parameters
    ----------
    name:
        File name inside ``nuee.data``.
    **kwargs:
        Additional arguments passed to ``pandas.read_csv``.
    """
    if "index_col" not in kwargs:
        kwargs["index_col"] = 0
    csv_path = resources.files("nuee.data").joinpath(name)
    with resources.as_file(csv_path) as path:
        return pd.read_csv(path, **kwargs)


def varespec() -> pd.DataFrame:
    """
    Lichen species data from Finnish forest.
    
    Returns:
        DataFrame with 24 sites and 44 lichen species
    """
    return _load_csv("varespec.csv")


def varechem() -> pd.DataFrame:
    """
    Environmental data corresponding to varespec.
    
    Returns:
        DataFrame with 24 sites and environmental variables
    """
    return _load_csv("varechem.csv")


def dune() -> pd.DataFrame:
    """
    Dutch dune meadow vegetation data.
    
    Returns:
        DataFrame with 20 sites and 30 species
    """
    return _load_csv("dune.csv")


def dune_env() -> pd.DataFrame:
    """
    Environmental data corresponding to dune.
    
    Returns:
        DataFrame with 20 sites and environmental variables
    """
    return _load_csv("dune.env.csv")


def BCI() -> pd.DataFrame:
    """
    Barro Colorado Island tree data.
    
    Returns:
        DataFrame with 50 sites and 225 species
    """
    return _load_csv("BCI.csv")


def mite() -> pd.DataFrame:
    """
    Oribatid mite data.
    
    Returns:
        DataFrame with 70 sites and 35 species
    """
    return _load_csv("mite.csv")


def mite_env() -> pd.DataFrame:
    """
    Environmental data corresponding to mite.
    
    Returns:
        DataFrame with 70 sites and environmental variables
    """
    return _load_csv("mite.env.csv")


def load_example_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load example species and environmental data for quick testing.
    
    Returns:
        Tuple of (species_data, environmental_data)
    """
    return varespec(), varechem()
