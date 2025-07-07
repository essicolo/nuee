"""
Dissimilarity analysis and related functions.

This module implements dissimilarity measures and multivariate analysis
methods for community ecology, including:
- Distance/dissimilarity measures
- PERMANOVA (adonis2)
- ANOSIM
- MRPP
- Mantel test
- Protest (Procrustes rotation)
"""

from .distances import vegdist
from .anosim import anosim
from .mrpp import mrpp
from .betadisper import betadisper
from .mantel import mantel, mantel_partial
from .protest import protest

# Import PERMANOVA from permutation module
from ..permutation.permanova import adonis2, permanova

__all__ = [
    "vegdist",
    "adonis2",
    "permanova", 
    "anosim",
    "mrpp",
    "betadisper",
    "mantel",
    "mantel_partial",
    "protest",
]