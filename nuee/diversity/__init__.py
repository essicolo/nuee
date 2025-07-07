"""
Diversity analysis for community ecology.

This module implements various diversity indices and related functions
commonly used in community ecology, including:
- Shannon diversity
- Simpson diversity
- Fisher's alpha
- Renyi entropy
- Evenness measures
- Species richness
"""

from .diversity import (
    diversity,
    shannon,
    simpson,
    fisher_alpha,
    renyi,
    specnumber,
    evenness,
)

from .rarefaction import (
    rarefy,
    rarecurve,
    estimateR,
)

from .accumulation import (
    specaccum,
    poolaccum,
)

__all__ = [
    "diversity",
    "shannon", 
    "simpson",
    "fisher_alpha",
    "renyi",
    "specnumber",
    "evenness",
    "rarefy",
    "rarecurve", 
    "estimateR",
    "specaccum",
    "poolaccum",
]