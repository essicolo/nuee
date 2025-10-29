"""
Ordination methods for community ecology.

This module implements various ordination techniques commonly used in 
community ecology analysis, including:
- NMDS (Non-metric Multidimensional Scaling)
- RDA (Redundancy Analysis)
- CCA (Canonical Correspondence Analysis)
- PCA (Principal Component Analysis)
- CA (Correspondence Analysis)
- Environmental variable fitting
- Procrustes analysis
"""

from .nmds import metaMDS
from .rda import rda
from .cca import cca
from .pca import pca
from .envfit import envfit
from .ordistep import ordistep
from .procrustes import procrustes

__all__ = [
    "metaMDS",
    "rda", 
    "cca",
    "pca",
    "envfit",
    "ordistep",
    "procrustes",
]