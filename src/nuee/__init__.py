"""
nuee: Community Ecology Analysis in Python
===========================================

``nuee`` is a comprehensive Python implementation of the popular R package ``vegan``
for community ecology analysis. It provides tools for ordination, diversity analysis,
dissimilarity measures, and statistical testing commonly used in ecological research.

Modules
-------
ordination : module
    Ordination methods including NMDS, RDA, CCA, PCA, and environmental fitting
diversity : module
    Diversity indices and rarefaction analysis
dissimilarity : module
    Distance measures and dissimilarity-based tests (PERMANOVA, ANOSIM, etc.)
permutation : module
    Permutation-based statistical tests
plotting : module
    Visualization functions for ecological data
composition : module
    Compositional data analysis utilities (CLR/ILR/ALR transforms, closure, etc.)
datasets : module
    Sample datasets for testing and examples

Examples
--------
Basic NMDS ordination:

>>> import nuee
>>> species_data = nuee.datasets.varespec()
>>> nmds_result = nuee.metaMDS(species_data, k=2, distance="bray")
>>> print(f"NMDS Stress: {nmds_result.stress:.3f}")
NMDS Stress: 0.133

Calculate diversity indices:

>>> shannon_div = nuee.shannon(species_data)
>>> simpson_div = nuee.simpson(species_data)
>>> richness = nuee.specnumber(species_data)

Perform PERMANOVA test:

>>> distances = nuee.vegdist(species_data, method="bray")
>>> env_data = nuee.datasets.varechem()
>>> permanova_result = nuee.adonis2(distances, env_data)

Notes
-----
nuee is inspired by the R package vegan developed by Jari Oksanen and the vegan
development team. It aims to provide similar functionality in a Pythonic interface
while leveraging the scientific Python ecosystem (NumPy, SciPy, pandas, matplotlib).

References
----------
.. [1] Oksanen, J., et al. (2020). vegan: Community Ecology Package.
       R package version 2.5-7. https://CRAN.R-project.org/package=vegan
"""

__version__ = "0.1.2"
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
    rarefy,
    rarecurve,
    estimateR,
    specaccum,
    poolaccum,
)

from .dissimilarity import (
    vegdist,
    adonis2,
    anosim,
    mrpp,
    betadisper,
    mantel,
    mantel_partial,
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
    plot_rarecurve,
    plot_specaccum,
    ordiellipse,
    ordispider,
    plot_betadisper,
)

from .composition import (
    closure,
    multiplicative_replacement,
    power,
    clr,
    clr_inv,
    inner,
    ilr,
    ilr_inv,
    alr,
    alr_inv,
    sbp_basis,
    center,
    centralize,
)

from . import composition
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
    "rarefy",
    "rarecurve",
    "estimateR",
    "specaccum",
    "poolaccum",
    # Dissimilarity
    "vegdist",
    "adonis2",
    "anosim",
    "mrpp",
    "betadisper",
    "mantel",
    "mantel_partial",
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
    "plot_rarecurve",
    "plot_specaccum",
    "ordiellipse",
    "ordispider",
    "plot_betadisper",
    # Composition
    "closure",
    "multiplicative_replacement",
    "power",
    "clr",
    "clr_inv",
    "inner",
    "ilr",
    "ilr_inv",
    "alr",
    "alr_inv",
    "sbp_basis",
    "center",
    "centralize",
    "composition",
    # Datasets
    "datasets",
]
