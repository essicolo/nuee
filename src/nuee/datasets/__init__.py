"""
Sample datasets for testing and examples.

This module provides sample datasets commonly used in community ecology
for testing and demonstrating the functionality of nuee. Every CSV bundled
under ``nuee.data`` has a loader here.
"""

from .sample_data import (
    varespec,
    varechem,
    dune,
    dune_env,
    dune_taxon,
    BCI,
    BCI_env,
    mite,
    mite_env,
    mite_pcnm,
    mite_xy,
    pyrifos,
    sipoo,
    sipoo_map,
)

__all__ = [
    "varespec",
    "varechem",
    "dune",
    "dune_env",
    "dune_taxon",
    "BCI",
    "BCI_env",
    "mite",
    "mite_env",
    "mite_pcnm",
    "mite_xy",
    "pyrifos",
    "sipoo",
    "sipoo_map",
]
