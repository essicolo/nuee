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
from .base import OrdinationMethod, OrdinationResult, NMDSResult
from .nmds import metaMDS, NMDS
from .rda import rda, RDA
from .cca import cca, ca, CCA
from .pca import pca, PCA
from .lda import lda, LDA
from .envfit import envfit
from .ordistep import ordistep
from .procrustes import procrustes

__all__ = [
    "metaMDS",
    "rda", 
    "cca",
    "ca",
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