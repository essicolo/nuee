"""
nuee: Python implementation of R nuee package for community ecology.

This package provides tools for descriptive community ecology, including:
- Ordination methods (NMDS, RDA, PCA, CA, CCA)
- Diversity analysis
- Dissimilarity analysis
- Permutation tests
- Environmental variable fitting
- Plotting functions

Classes and functions are organized into modules:
- ordination: Ordination methods and analysis
- diversity: Diversity indices and analysis
- dissimilarity: Dissimilarity measures and analysis
- permutation: Permutation tests and statistics
- plotting: Visualization functions
- datasets: Sample datasets for testing and examples
"""

__version__ = "0.1.0"
__author__ = "nuee Development Team"

from .ordination import (
    metaMDS,
    rda,
    cca,
    pca,
    envfit,
    ordistep,
    procrustes,
)

from .diversity import (
    diversity,
    specnumber,
    fisher_alpha,
    renyi,
    simpson,
    shannon,
    evenness,
)

from .dissimilarity import (
    vegdist,
    adonis2,
    anosim,
    mrpp,
    betadisper,
    mantel,
    protest,
)

from .permutation import (
    permanova,
    permtest,
    permutest,
    anova_cca,
)

from .plotting import (
    plot_ordination,
    plot_diversity,
    plot_dissimilarity,
    biplot,
    ordiplot,
)

from . import datasets

__all__ = [
    # Ordination
    "metaMDS",
    "rda",
    "cca", 
    "pca",
    "envfit",
    "ordistep",
    "procrustes",
    # Diversity
    "diversity",
    "specnumber",
    "fisher_alpha",
    "renyi",
    "simpson",
    "shannon",
    "evenness",
    # Dissimilarity
    "vegdist",
    "adonis2",
    "anosim",
    "mrpp",
    "betadisper",
    "mantel",
    "protest",
    # Permutation
    "permanova",
    "permtest",
    "permutest",
    "anova_cca",
    # Plotting
    "plot_ordination",
    "plot_diversity",
    "plot_dissimilarity",
    "biplot",
    "ordiplot",
    # Datasets
    "datasets",
]