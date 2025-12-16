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
from .base import OrdinationMethod, OrdinationResult
from .nmds import metaMDS, NMDS
from .rda import rda, RDA
from .cca import cca, CCA
from .pca import pca, PCA
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
    "NMDS",
    "RDA",
    "CCA",
    "PCA",
    "OrdinationMethod",
    "OrdinationResult",
]