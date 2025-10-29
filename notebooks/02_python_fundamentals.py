import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 2: Python Fundamentals for Ecological Analysis

    This chapter introduces essential Python concepts for ecological data analysis,
    translating concepts from R to Python with a focus on ecological applications.

    ## Learning Objectives
    - Master Python syntax and data structures
    - Understand Python's approach to data analysis
    - Learn best practices for ecological programming
    - Set up a reproducible Python environment
    """
    )
    return


@app.cell
def _():
    # Essential imports for ecological analysis
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    return np, pd, stats


@app.cell
def _(mo):
    mo.md(r"""## Basic Python Data Types""")
    return


@app.cell
def _():
    # Numbers are intergers of float
    temperature = 25.5  # float
    sample_size = 100   # integer

    # Strings
    species_name = "Quercus alba"
    site_code = "SITE_001"

    # Booleans (True/False or when only two categories exist)
    is_endemic = True # capital on the first letter
    has_flowers = False

    # Lists
    species_counts = [12, 8, 15, 3, 22]
    site_names = ["Forest", "Meadow", "Wetland", "Urban", "Agricultural"]

    # Dictionaries are like tables but more flexible
    site_data = {
        "Forest": {"abundance": 120, "biomass": 1.2},
        "Meadow": {"abundance": 80, "biomass": 0.8},
        "Wetland": {"abundance": 150, "biomass": 2.1}
    }
    return


@app.cell
def _(mo):
    mo.md(r"""## Working with NumPy Arrays""")
    return


@app.cell
def _(np):
    # Create arrays
    abundance = np.array([12, 8, 15, 3, 22, 7, 18, 5])
    biomass = np.array([1.2, 0.8, 2.1, 0.3, 3.4, 0.9, 2.8, 0.6])

    # Basic statistics (print allow to show output at the moment it is triggered)
    print("Abundance Statistics:")
    print(f"Mean: {np.mean(abundance):.2f}")
    print(f"Median: {np.median(abundance):.2f}")
    print(f"Standard deviation: {np.std(abundance):.2f}")
    print(f"Range: {np.min(abundance)} - {np.max(abundance)}")

    # Array operations are vectorized, i.e., applied element-wise
    density = abundance / biomass  # Element-wise division
    log_abundance = np.log(abundance + 1)  # Log transformation (common in ecology)

    print(f"\nDensity (abundance/biomass): {density}")
    print(f"Log-transformed abundance: {log_abundance}")
    return abundance, biomass


@app.cell
def _(mo):
    mo.md(r"""## Creating DataFrames (R's data.frame equivalent)""")
    return


@app.cell
def _(abundance, biomass, pd):
    # Create a simple ecological dataset
    ecological_data = pd.DataFrame({
        'site': ['Forest', 'Meadow', 'Wetland', 'Urban', 'Agricultural', 'Desert', 'Mountain', 'Coastal'],
        'species_richness': [25, 18, 22, 8, 12, 6, 20, 15],
        'abundance': abundance,
        'biomass': biomass,
        'temperature': [22.5, 25.1, 18.3, 28.7, 24.2, 35.8, 15.6, 21.4],
        'precipitation': [1200, 800, 1500, 600, 900, 150, 2000, 1100],
        'elevation': [250, 120, 85, 45, 180, 1200, 1800, 15]
    })

    print("Ecological Dataset:")
    print(ecological_data.head())
    print(f"\nDataFrame shape: {ecological_data.shape}")
    print(f"Column types:\n{ecological_data.dtypes}")
    return (ecological_data,)


@app.cell
def _(mo):
    mo.md(r"""## Basic Data Exploration""")
    return


@app.cell
def _(ecological_data):
    # Descriptive statistics
    print("Summary Statistics:")
    print(ecological_data.describe())

    # Check for missing values
    print(f"\nMissing values:\n{ecological_data.isnull().sum()}")

    # Data types and info
    print(f"\nDataFrame info:")
    ecological_data.info()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Indexing and Selection""")
    return


@app.cell
def _(ecological_data):
    _richness = ecological_data['species_richness']
    _temp_precip = ecological_data[['temperature', 'precipitation']]

    # Select rows (similar to R's data[1:3,])
    first_three_sites = ecological_data.iloc[0:3]  # 0-based indexing!

    # Conditional selection (similar to R's subset())
    high_diversity = ecological_data[ecological_data['species_richness'] > 15]
    warm_sites = ecological_data[ecological_data['temperature'] > 25]

    print("High diversity sites:")
    print(high_diversity[['site', 'species_richness']])

    print(f"\nWarm sites: {list(warm_sites['site'])}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Functions in Python""")
    return


@app.cell
def _(np):
    def calculate_shannon_diversity(abundances):
        """
        Calculate Shannon diversity index
        Similar to vegan::diversity() in R
        """
        # Remove zeros and normalize
        abundances = np.array(abundances)
        abundances = abundances[abundances > 0]
        proportions = abundances / np.sum(abundances)

        # Shannon formula: H = -sum(p * log(p))
        shannon = -np.sum(proportions * np.log(proportions))
        return shannon

    def simpson_diversity(abundances):
        """
        Calculate Simpson diversity index
        """
        abundances = np.array(abundances)
        n = np.sum(abundances)
        simpson = 1 - np.sum((abundances * (abundances - 1)) / (n * (n - 1)))
        return simpson

    # Test with example data
    test_community = [10, 8, 6, 4, 2, 1, 1]
    shannon_H = calculate_shannon_diversity(test_community)
    simpson_D = simpson_diversity(test_community)

    print(f"Test community: {test_community}")
    print(f"Shannon diversity (H): {shannon_H:.3f}")
    print(f"Simpson diversity (D): {simpson_D:.3f}")
    return (calculate_shannon_diversity,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Control Structures: Loops and Conditionals

    (add example for conditionals)

    Variables staring with an underscore `_` won't leave the marimo cell. They are useful to be reused from a cell to anotherr, in particular with the `_i` in loops, used everywhere by convention.
    """
    )
    return


@app.cell
def _(calculate_shannon_diversity, ecological_data, np):
    _diversity_indices = []

    print("Calculating diversity for each site:")
    for _i, _site in enumerate(ecological_data['site']):
        _n_species = ecological_data.loc[_i, 'species_richness']
        _abundances = np.random.lognormal(2, 1, _n_species)
        _diversity = calculate_shannon_diversity(_abundances)
        _diversity_indices.append(_diversity)

    # Add diversity to dataframe
    ecological_data['shannon_diversity'] = _diversity_indices
    ecological_data
    return


@app.cell
def _(mo):
    mo.md(r"""## List Comprehensions (Pythonic approach)""")
    return


@app.cell
def _(ecological_data):
    # More concise way to create lists
    # Calculate elevation categories
    elevation_categories = ['Low' if elev < 200 else 'Medium' if elev < 1000 else 'High' 
                           for elev in ecological_data['elevation']]

    ecological_data['elevation_category'] = elevation_categories

    # Group by elevation category
    elevation_summary = ecological_data.groupby('elevation_category').agg({
        'species_richness': ['mean', 'std'],
        'shannon_diversity': ['mean', 'std'],
        'temperature': 'mean'
    }).round(2)

    elevation_summary
    return


@app.cell
def _(mo):
    mo.md(r"""## Error Handling""")
    return


@app.cell
def _(calculate_shannon_diversity, np):
    def safe_diversity_calculation(abundances):
        """
        Safely calculate diversity with error handling
        """
        try:
            if len(abundances) == 0:
                raise ValueError("Empty abundance vector")

            diversity = calculate_shannon_diversity(abundances)
            return diversity

        except ValueError as e:
            print(f"Error: {e}")
            return np.nan
        except Exception as e:
            print(f"Unexpected error: {e}")
            return np.nan

    # Test error handling
    test_cases = [
        [10, 5, 3, 2],      # Normal case
        [],                  # Empty list
        [0, 0, 0],          # All zeros
        "invalid"           # Wrong type
    ]

    for _i, _test in enumerate(test_cases):
        _result = safe_diversity_calculation(_test)
        print(f"Test {_i+1}: {_test} → Diversity: {_result}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Working with Packages""")
    return


@app.cell
def _(ecological_data, stats):
    # Scientific computing packages for ecology

    # Statistical tests (similar to R's built-in tests)
    # Test correlation between temperature and species richness
    temp = ecological_data['temperature']
    richness = ecological_data['species_richness']

    correlation, p_value = stats.pearsonr(temp, richness)

    print(f"Temperature vs Species Richness:")
    print(f"Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.3f}")

    # Test normality (similar to R's shapiro.test())
    statistic, p_norm = stats.shapiro(richness)
    print(f"\nNormality test for species richness:")
    print(f"Shapiro-Wilk statistic: {statistic:.3f}")
    print(f"P-value: {p_norm:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Best Practices for Ecological Programming

    1. **Use descriptive variable names**
       - `species_richness` instead of `sr`
       - `temperature_celsius` instead of `temp`

    2. **Comment your code**
       - Explain the context
       - Document data sources and transformations

    3. **Handle missing data explicitly**
       - Check for NaN values
       - Document missing data patterns

    4. **Use appropriate data types**
       - Categorical variables as strings or categories
       - Continuous variables as floats
       - Count data as integers

    5. **Validate your data**
       - Check ranges (e.g., abundances ≥ 0)
       - Verify units and scales
       - Test for outliers

    6. **Document your analysis**
       - Include metadata
       - Describe methods and assumptions
       - Provide reproducible examples

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary

    In this chapter, we covered:

    - **Python basics**: Data types, variables, and operations
    - **NumPy arrays**: Efficient numerical computing
    - **Pandas DataFrames**: Tabular data manipulation
    - **Functions**: Creating reusable analysis code
    - **Control structures**: Loops and conditionals
    - **Error handling**: Robust analysis workflows
    - **Statistical functions**: Basic ecological statistics

    **Next Chapter**: Data organization and manipulation with pandas

    **Key Takeaways**:
    - Python uses 0-based indexing (different from R)
    - pandas DataFrames are similar to R's data.frames
    - NumPy provides efficient numerical operations
    - List comprehensions offer concise data processing
    - Error handling makes analyses more robust
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
