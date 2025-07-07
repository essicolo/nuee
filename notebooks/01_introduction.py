import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Mathematical Ecology with Python

    ## Chapter 1: Introduction to Mathematical Ecology

    Welcome to Mathematical Ecology with Python! This book provides a comprehensive 
    introduction to ecological data analysis and modeling using modern Python tools.

    Originally based on "Analyse et modélisation d'agroécosystèmes" by Serge-Étienne Parent,
    this English adaptation uses Python's scientific computing ecosystem to explore 
    mathematical ecology concepts.

    ### Learning Objectives

    - Understand mathematical approaches to ecological problems
    - Master Python tools for ecological data analysis
    - Apply statistical and machine learning methods to ecological datasets
    - Develop reproducible workflows for ecological research

    ### Python Ecosystem for Ecology
    This book leverages key Python packages that are Pyodide-compatible:

    - **pandas**: Data manipulation and analysis
    - **numpy**: Numerical computing
    - **scipy**: Scientific computing and statistics
    - **scikit-learn**: Machine learning
    - **statsmodels**: Statistical modeling
    - **nuee**: Community ecology (Python port of R's vegan)
    - **holoviews**: Interactive data visualization
    """
    )
    return


@app.cell
def _():
    # Import core packages
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    hv.extension('bokeh')
    return


@app.cell
def _():
    """
    ### About This Book

    **Target Audience**: Graduate students and researchers in:
    - Ecology and Environmental Science
    - Agronomy and Agricultural Systems
    - Conservation Biology
    - Environmental Engineering

    **Prerequisites**:
    - Basic knowledge of ecology and statistics
    - Some programming experience (R or Python)
    - Familiarity with linear algebra concepts

    **Book Structure**:
    - 14 chapters progressing from basic Python to advanced ecological modeling
    - Hands-on approach with real ecological datasets
    - Reproducible examples using Marimo notebooks
    - Integration of statistical theory with practical applications
    """
    return


@app.cell
def _():
    """
    ### Chapter Overview

    1. **Introduction** - Mathematical ecology concepts and Python ecosystem
    2. **Python Fundamentals** - Core Python for ecological analysis
    3. **Data Organization** - Pandas for ecological data manipulation
    4. **Data Visualization** - Holoviews for interactive ecological plots
    5. **Reproducible Science** - Version control and reproducible workflows
    6. **Biostatistics** - Classical statistical approaches in ecology
    7. **Bayesian Biostatistics** - Bayesian methods for ecological data
    8. **Exploratory Analysis** - Advanced data exploration techniques
    9. **Ordination & Classification** - Multivariate analysis with nuee
    10. **Data Quality** - Outlier detection and missing data imputation
    11. **Machine Learning** - ML applications in ecological modeling
    12. **Time Series** - Temporal analysis of ecological data
    13. **Spatial Data** - Geospatial analysis for ecology
    14. **Mechanistic Modeling** - Process-based ecological models
    """
    return


@app.cell
def _():
    """
    ### Mathematical Ecology: Core Concepts

    Mathematical ecology applies quantitative methods to understand:

    **Population Dynamics**: How species populations change over time
    - Growth models (exponential, logistic)
    - Predator-prey interactions
    - Metapopulation dynamics

    **Community Ecology**: Relationships between multiple species
    - Species diversity and evenness
    - Community ordination and classification
    - Species-environment relationships

    **Ecosystem Processes**: Energy flows and nutrient cycling
    - Food web structure and dynamics
    - Biogeochemical cycles
    - Ecosystem stability and resilience

    **Spatial Ecology**: Geographic patterns and processes
    - Species distribution modeling
    - Landscape ecology
    - Spatial autocorrelation and scale effects
    """
    return


@app.cell
def _():
    """
    ### Why Python for Ecology?

    **Advantages of Python**:
    - Open source and free
    - Extensive scientific computing ecosystem
    - Strong community support
    - Integration with web technologies (Pyodide, WASM)
    - Reproducible research workflows

    **Key Packages for Ecological Analysis**:
    - **nuee**: Port of R's vegan package for community ecology
    - **scikit-learn**: Machine learning for species classification and prediction
    - **scipy**: Statistical tests and optimization
    - **pandas**: Data manipulation and analysis
    - **holoviews**: Interactive and publication-quality visualizations
    """
    return


@app.cell
def _():
    """
    ## Next Steps

    In the following chapters, we'll progressively build your skills in:

    1. **Python Fundamentals** - Master the basics of Python programming
    2. **Data Manipulation** - Learn pandas for ecological data wrangling
    3. **Visualization** - Create compelling ecological visualizations
    4. **Statistical Analysis** - Apply classical and modern statistical methods
    5. **Advanced Methods** - Explore machine learning and spatial analysis

    Let's begin this journey into mathematical ecology with Python!

    **Continue to Chapter 2: Python Fundamentals →**
    """
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
