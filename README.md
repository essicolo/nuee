**`nuee` has just been released, and is likely to contain errors and bugs. The code has been tested, but hasn't been thoroughly reviewed. Do not trust its results blindly.**

# `nuee`: Community Ecology Analysis in Python

`nuee` is a comprehensive Python implementation of the popular R package `vegan` for community ecology analysis. It provides tools for ordination, diversity analysis, dissimilarity measures, and statistical testing commonly used in ecological research.

## Installation

```bash
pip install nuee
```

## Documentation

Full documentation is available at: [https://essicolo.github.io/nuee/](https://essicolo.github.io/nuee/)

The documentation includes:
- Installation instructions
- Quick start guide
- User guide with detailed examples
- Complete API reference

To build the documentation locally:
```bash
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

## Features

### Ordination Methods
- **NMDS** (Non-metric Multidimensional Scaling) with `metaMDS()`
- **RDA** (Redundancy Analysis) with `rda()`
- **CCA** (Canonical Correspondence Analysis) with `cca()`
- **PCA** (Principal Component Analysis) with `pca()`
- **Environmental fitting** with `envfit()`
- **Procrustes analysis** with `procrustes()`

### Diversity Analysis
- **Shannon diversity** with `shannon()`
- **Gini-Simpson diversity** with `simpson()` (1 - sum(p^2))
- **Fisher's alpha** with `fisher_alpha()`
- **Renyi entropy** with `renyi()`
- **Species richness** with `specnumber()`
- **Evenness measures** with `evenness()`
- **Rarefaction** with `rarefy()` and `rarecurve()`

### Dissimilarity Measures
- **Bray-Curtis**, **Jaccard**, **Euclidean**, and 15+ other distances with `vegdist()`
- **PERMANOVA** with `adonis2()`
- **ANOSIM** with `anosim()`
- **MRPP** with `mrpp()`
- **Mantel test** with `mantel()`
- **Beta dispersion** with `betadisper()`

### Visualization
- **Ordination plots** with `plot_ordination()`
- **Biplots** with `biplot()`
- **Diversity plots** with `plot_diversity()`
- **Rarefaction curves** with `plot_rarecurve()`
- **Confidence ellipses** with `ordiellipse()`

### Sample Datasets
- **varespec** & **varechem**: Lichen species and environmental data
- **dune** & **dune_env**: Dutch dune meadow vegetation
- **BCI**: Barro Colorado Island tree data  
- **mite** & **mite_env**: Oribatid mite data

## Installation
## Quick Start

```python
import nuee 
import matplotlib.pyplot as plt

# Load sample data
species_data = nuee.datasets.varespec()
env_data = nuee.datasets.varechem()

# NMDS Ordination
nmds_result = nuee.metaMDS(species_data, k=2, distance="bray")
print(f"NMDS Stress: {nmds_result.stress:.3f}")

# Plot ordination
fig = nuee.plot_ordination(nmds_result, display="sites")
plt.show()

# Calculate diversity indices
shannon_div = nuee.shannon(species_data)
simpson_div = nuee.simpson(species_data)
print(f"Shannon diversity: {shannon_div.mean():.3f}")
print(f"Gini-Simpson diversity: {simpson_div.mean():.3f}")

# RDA with environmental variables
rda_result = nuee.rda(species_data, env_data)
fig = nuee.biplot(rda_result)
plt.show()

# PERMANOVA
distances = nuee.vegdist(species_data, method="bray")
permanova_result = nuee.adonis2(distances, env_data)
print(permanova_result)
```

## Advanced Examples

### Constrained Ordination with Formula Interface

```python
import nuee 
import pandas as pd

# Load data
species = nuee.datasets.dune()
env = nuee.datasets.dune_env()

# RDA with formula
rda_result = nuee.rda(species, formula="~ A1 + Management", data=env)

# Plot with groups
fig = nuee.plot_ordination(rda_result, groups=env['Management'])
plt.show()
```

### Diversity Analysis with Rarefaction

```python
import nuee 
import matplotlib.pyplot as plt

# Load data
species = nuee.datasets.BCI()

# Calculate multiple diversity indices
diversity_indices = {
    'Shannon': nuee.shannon(species),
    'Simpson': nuee.simpson(species), 
    'Fisher': nuee.fisher_alpha(species),
    'Richness': nuee.specnumber(species)
}

# Rarefaction curve
rarefaction = nuee.rarecurve(species, step=10)
fig = nuee.plot_rarecurve(rarefaction)
plt.show()
```

### Permutation Tests

```python
import nuee 

# Load data
species = nuee.datasets.mite()
env = nuee.datasets.mite_env()

# PERMANOVA
dist_matrix = nuee.vegdist(species, method="bray")
permanova_result = nuee.adonis2(dist_matrix, env[['SubsDens', 'WatrCont']])

# ANOSIM
anosim_result = nuee.anosim(dist_matrix, env['Substrate'])

# Mantel test
env_dist = nuee.vegdist(env[['SubsDens', 'WatrCont']], method="euclidean")
mantel_result = nuee.mantel(dist_matrix, env_dist)

print(f"PERMANOVA R^2: {permanova_result["r_squared"]:.3f}")
print(f"ANOSIM R: {anosim_result["r_statistic"]:.3f}")
print(f"Mantel r: {mantel_result["r_statistic"]:.3f}"))
```

## Dependencies

- **numpy** >= 1.20.0
- **scipy** >= 1.7.0  
- **pandas** >= 1.3.0
- **matplotlib** >= 3.4.0
- **seaborn** >= 0.11.0
- **scikit-learn** >= 1.0.0
- **patsy** >= 0.5.0 (for formula interface)
- **statsmodels** >= 0.12.0

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

nuee is inspired by the excellent R package `vegan` developed by Jari Oksanen and the `vegan` development team. Both `vegan` and `nuee` were inspired by the book ["Numerical Ecology"](https://shop.elsevier.com/books/numerical-ecology/legendre/978-0-444-53868-0), by Pierre Legendre and Louis Legendre (3rd edition, 2012). We acknowledge their pioneering work in developping the science of numerical ecology and making it accessible to researchers.
