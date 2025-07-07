# Mathematical Ecology with Python

This directory contains Marimo notebooks for the English translation of "Analyse et modélisation d'agroécosystèmes" by Serge-Étienne Parent, adapted for Python using the nuee package and modern scientific computing tools.

## Book Structure

### Part I: Foundations
1. **[Introduction](01_introduction.py)** - Mathematical ecology concepts and Python ecosystem
2. **[Python Fundamentals](02_python_fundamentals.py)** - Core Python for ecological analysis  
3. **[Data Organization](03_data_organization.py)** - Pandas for ecological data manipulation
4. **[Data Visualization](04_data_visualization.py)** - Holoviews for interactive ecological plots
5. **[Reproducible Science](05_reproducible_science.py)** - Version control and reproducible workflows

### Part II: Statistical Analysis
6. **[Biostatistics](06_biostatistics.py)** - Classical statistical approaches in ecology
7. **[Bayesian Biostatistics](07_bayesian_biostatistics.py)** - Bayesian methods for ecological data
8. **[Exploratory Analysis](08_exploratory_analysis.py)** - Advanced data exploration techniques

### Part III: Multivariate Methods
9. **[Ordination & Classification](08_ordination_classification.py)** - Multivariate analysis with nuee
10. **[Data Quality](09_data_quality.py)** - Outlier detection and missing data imputation

### Part IV: Advanced Methods
11. **[Machine Learning](10_machine_learning.py)** - ML applications in ecological modeling
12. **[Time Series](11_time_series.py)** - Temporal analysis of ecological data
13. **[Spatial Data](12_spatial_data.py)** - Geospatial analysis for ecology
14. **[Mechanistic Modeling](13_mechanistic_modeling.py)** - Process-based ecological models

## Python Package Requirements

All packages are Pyodide-compatible for web deployment:

### Core Data Science
- **pandas** >= 1.5.0 - Data manipulation and analysis
- **numpy** >= 1.21.0 - Numerical computing
- **scipy** >= 1.9.0 - Scientific computing and statistics
- **statsmodels** >= 0.13.0 - Statistical modeling
- **scikit-learn** >= 1.1.0 - Machine learning

### Ecological Analysis
- **nuee** >= 0.1.0 - Community ecology (Python port of R's vegan)

### Visualization
- **holoviews** >= 1.15.0 - Interactive data visualization
- **panel** >= 0.14.0 - Interactive dashboards
- **matplotlib** >= 3.5.0 - Static plotting
- **bokeh** >= 2.4.0 - Interactive web plots

### Notebook Environment
- **marimo** >= 0.10.0 - Reactive notebook environment

## Installation

```bash
# Install core packages
pip install pandas numpy scipy statsmodels scikit-learn

# Install visualization packages  
pip install holoviews panel matplotlib bokeh

# Install nuee (ecological analysis)
pip install nuee

# Install marimo for notebooks
pip install marimo
```

## Running the Notebooks

```bash
# Navigate to the notebooks directory
cd notebooks/

# Run a specific notebook
marimo run 01_introduction.py

# Edit a notebook
marimo edit 01_introduction.py
```

## Features

### Ecological Focus
- **Community ecology** analysis using nuee (ordination, diversity, dissimilarity)
- **Species-environment** relationships and modeling
- **Spatial ecology** and biogeography
- **Temporal dynamics** and time series analysis
- **Conservation biology** applications

### Modern Python Tools
- **Interactive visualizations** with holoviews
- **Reproducible workflows** with marimo notebooks
- **Statistical modeling** with statsmodels and scikit-learn
- **Geospatial analysis** with compatible libraries
- **Machine learning** applications in ecology

### Pedagogical Approach
- **Progressive complexity** from basic Python to advanced modeling
- **Real ecological datasets** and practical examples
- **Hands-on exercises** and reproducible analyses
- **Best practices** for ecological data science
- **Integration** of statistical theory with ecological applications

## Dataset Sources

The notebooks use classic ecological datasets including:
- **varespec/varechem** - Lichen species and environmental data
- **dune/dune.env** - Dutch dune meadow vegetation
- **BCI** - Barro Colorado Island tree census data
- **mite/mite.env** - Oribatid mite communities

## Contributing

This is a translation and adaptation project. To contribute:

1. Check the original French book: https://github.com/essicolo/ecologie-mathematique-R
2. Follow the established structure and pedagogical approach
3. Ensure all code uses Pyodide-compatible packages
4. Maintain the ecological focus and practical applications
5. Test notebooks in marimo environment

## License

This work is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0, 
following the original book's licensing terms.

## Acknowledgments

- **Original Author**: Serge-Étienne Parent
- **Original Book**: "Analyse et modélisation d'agroécosystèmes"
- **Python Translation**: Adapted for the nuee package and Python ecosystem
- **Notebook Environment**: Built with marimo reactive notebooks

---

**Mathematical Ecology with Python** - Bringing R's vegan package to the Python ecosystem with modern, interactive tools for ecological data science.