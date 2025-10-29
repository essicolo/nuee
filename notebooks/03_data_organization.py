import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 3: Data Organization with Pandas

    This chapter covers data manipulation and organization techniques using pandas,
    the Python equivalent of R's tidyverse ecosystem.

    ## Learning Objectives
    - Master pandas DataFrames for ecological data
    - Learn data cleaning and transformation techniques
    - Understand grouping and aggregation operations
    - Handle missing data in ecological datasets
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    hv.extension('bokeh')
    return np, pd


@app.cell
def _(mo):
    mo.md(r"""## Creating Ecological DataFrames""")
    return


@app.cell
def _(np, pd):
    # Create a comprehensive ecological dataset
    np.random.seed(42)  # For reproducibility

    # Generate sample data
    n_sites = 50
    sites = [f"SITE_{i:03d}" for i in range(1, n_sites + 1)]

    ecological_data = pd.DataFrame({
        'site_id': sites,
        'habitat': np.random.choice(['Forest', 'Grassland', 'Wetland', 'Urban', 'Agricultural'], n_sites),
        'latitude': np.random.uniform(45.0, 48.0, n_sites),
        'longitude': np.random.uniform(-75.0, -70.0, n_sites),
        'elevation': np.random.uniform(50, 1200, n_sites),
        'temperature': np.random.normal(15, 5, n_sites),
        'precipitation': np.random.lognormal(6, 0.5, n_sites),
        'soil_pH': np.random.normal(6.5, 1.2, n_sites),
        'nitrogen': np.random.gamma(2, 5, n_sites),
        'phosphorus': np.random.gamma(1.5, 3, n_sites),
        'species_richness': np.random.poisson(15, n_sites),
        'total_abundance': np.random.lognormal(4, 0.8, n_sites)
    })

    # Add some missing values (realistic in ecological data)
    missing_indices = np.random.choice(n_sites, size=8, replace=False)
    ecological_data.loc[missing_indices[:4], 'soil_pH'] = np.nan
    ecological_data.loc[missing_indices[4:], 'nitrogen'] = np.nan
    ecological_data
    return ecological_data, sites


@app.cell
def _(mo):
    mo.md(r"""## Data Inspection and Summary""")
    return


@app.cell
def _(ecological_data):
    # Basic information about the dataset
    print("Dataset Information:")
    print(ecological_data.info())

    print("\nSummary Statistics:")
    print(ecological_data.describe())

    print("\nMissing Values:")
    print(ecological_data.isnull().sum())

    print("\nData Types:")
    print(ecological_data.dtypes)
    return


@app.cell
def _(mo):
    mo.md(r"""## Selecting and Filtering Data""")
    return


@app.cell
def _(ecological_data):
    # Select specific columns (similar to R's select())
    environmental_vars = ecological_data[['site_id', 'temperature', 'precipitation', 'soil_pH', 'nitrogen']]

    # Select columns by pattern (similar to R's starts_with(), ends_with())
    spatial_cols = [col for col in ecological_data.columns if col in ['latitude', 'longitude', 'elevation']]
    spatial_data = ecological_data[['site_id'] + spatial_cols]

    # Filter rows (similar to R's filter())
    forest_sites = ecological_data[ecological_data['habitat'] == 'Forest']
    high_diversity = ecological_data[ecological_data['species_richness'] > 15]
    northern_sites = ecological_data[ecological_data['latitude'] > 46.5]

    # Complex filtering with multiple conditions
    # each condition is enclosed in () to avoid any confusion with operator precedence
    rich_forest_sites = ecological_data[
        (ecological_data['habitat'] == 'Forest') & 
        (ecological_data['species_richness'] > 12) &
        (ecological_data['temperature'] > 10)
    ]

    print(f"Forest sites: {len(forest_sites)}")
    print(f"High diversity sites: {len(high_diversity)}")
    print(f"Rich forest sites: {len(rich_forest_sites)}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Creating New Variables""")
    return


@app.cell
def _(ecological_data, np, pd):
    # Add new calculated columns
    ecological_data_enhanced = ecological_data.copy()

    # Calculate derived variables
    ecological_data_enhanced['temp_fahrenheit'] = ecological_data_enhanced['temperature'] * 9/5 + 32
    ecological_data_enhanced['log_precipitation'] = np.log(ecological_data_enhanced['precipitation'])
    ecological_data_enhanced['diversity_per_abundance'] = (
        ecological_data_enhanced['species_richness'] / ecological_data_enhanced['total_abundance']
    )

    # Categorical variables from continuous ones
    ecological_data_enhanced['elevation_category'] = pd.cut(
        ecological_data_enhanced['elevation'], 
        bins=[0, 300, 600, 1200], 
        labels=['Low', 'Medium', 'High']
    )

    ecological_data_enhanced['temp_category'] = pd.cut(
        ecological_data_enhanced['temperature'],
        bins=[-10, 10, 20, 30],
        labels=['Cold', 'Moderate', 'Warm']
    )

    # Boolean flags
    ecological_data_enhanced['is_acidic'] = ecological_data_enhanced['soil_pH'] < 6.0
    ecological_data_enhanced['high_nitrogen'] = ecological_data_enhanced['nitrogen'] > ecological_data_enhanced['nitrogen'].median()

    print("Enhanced dataset columns:")
    print(list(ecological_data_enhanced.columns))
    return


@app.cell
def _(mo):
    mo.md(r"""## Sorting Data""")
    return


@app.cell
def _(ecological_data):
    # Sort by single column
    sorted_by_richness = ecological_data.sort_values('species_richness', ascending=False)

    # Sort by multiple columns
    sorted_multi = ecological_data.sort_values(['habitat', 'species_richness'], ascending=[True, False])

    # Sort by index
    sorted_by_site = ecological_data.sort_values('site_id')

    print("Top 5 sites by species richness:")
    print(sorted_by_richness[['site_id', 'habitat', 'species_richness']].head())

    print("\nSites sorted by habitat then richness:")
    print(sorted_multi[['site_id', 'habitat', 'species_richness']].head(10))
    return


@app.cell
def _(mo):
    mo.md(r"""## Grouping and Summarizing (R's group_by() and summarise())""")
    return


@app.cell
def _(ecological_data, np):
    # Group by habitat and calculate summary statistics
    habitat_summary = ecological_data.groupby('habitat').agg({
        'species_richness': ['mean', 'std', 'min', 'max'],
        'total_abundance': ['mean', 'median'],
        'temperature': 'mean',
        'precipitation': 'mean',
        'soil_pH': 'mean'
    }).round(2)

    print("Summary by habitat:")
    print(habitat_summary)

    # Custom aggregation functions
    def coefficient_of_variation(x):
        return x.std() / x.mean() if x.mean() != 0 else np.nan

    diversity_stats = ecological_data.groupby('habitat')['species_richness'].agg([
        'count', 'mean', 'std', coefficient_of_variation
    ]).round(3)

    print("\nDiversity statistics by habitat:")
    print(diversity_stats)
    return


@app.cell
def _(mo):
    mo.md(r"""## Reshaping Data""")
    return


@app.cell
def _(ecological_data, pd):
    # Create a dataset for reshaping demonstration
    sample_data = ecological_data[['site_id', 'habitat', 'species_richness', 'total_abundance']].head(10)

    # Pivot longer (melt): wide to long format
    long_format = pd.melt(
        sample_data, 
        id_vars=['site_id', 'habitat'],
        value_vars=['species_richness', 'total_abundance'],
        var_name='metric',
        value_name='value'
    )

    print("Long format (melted):")
    print(long_format.head(8))

    # Pivot wider: long to wide format
    wide_format = long_format.pivot_table(
        index=['site_id', 'habitat'], 
        columns='metric', 
        values='value'
    ).reset_index()

    print("\nWide format (pivoted):")
    print(wide_format.head())
    return


@app.cell
def _(mo):
    mo.md(r"""## Joining Datasets""")
    return


@app.cell
def _(ecological_data, pd):
    # Create separate datasets to demonstrate joins
    site_info = ecological_data[['site_id', 'habitat', 'latitude', 'longitude']].copy()
    environmental_data = ecological_data[['site_id', 'temperature', 'precipitation', 'soil_pH']].copy()
    species_data = ecological_data[['site_id', 'species_richness', 'total_abundance']].copy()

    # Add some extra sites to demonstrate different join types
    extra_sites = pd.DataFrame({
        'site_id': ['SITE_051', 'SITE_052'],
        'temperature': [18.5, 22.1],
        'precipitation': [800, 1200],
        'soil_pH': [6.8, 7.2]
    })
    environmental_extended = pd.concat([environmental_data, extra_sites], ignore_index=True)

    # Inner join (only matching records)
    inner_joined = pd.merge(site_info, environmental_data, on='site_id', how='inner')

    # Left join (all records from left table)
    left_joined = pd.merge(site_info, environmental_extended, on='site_id', how='left')

    # Outer join (all records from both tables)
    outer_joined = pd.merge(site_info, environmental_extended, on='site_id', how='outer')

    print(f"Original site_info: {len(site_info)} rows")
    print(f"Extended environmental: {len(environmental_extended)} rows")
    print(f"Inner join: {len(inner_joined)} rows")
    print(f"Left join: {len(left_joined)} rows")
    print(f"Outer join: {len(outer_joined)} rows")

    return


@app.cell
def _(ecological_data, pd):
    """
    ## Handling Missing Data
    """
    # Examine missing data patterns
    missing_summary = ecological_data.isnull().sum()
    missing_percentage = (ecological_data.isnull().sum() / len(ecological_data) * 100).round(1)

    missing_df = pd.DataFrame({
        'Missing_Count': missing_summary,
        'Missing_Percentage': missing_percentage
    })

    print("Missing data summary:")
    print(missing_df[missing_df['Missing_Count'] > 0])

    # Different strategies for handling missing data

    # 1. Drop rows with any missing values
    complete_cases = ecological_data.dropna()
    print(f"\nComplete cases: {len(complete_cases)} out of {len(ecological_data)}")

    # 2. Drop rows with missing values in specific columns
    ph_complete = ecological_data.dropna(subset=['soil_pH'])
    print(f"pH complete cases: {len(ph_complete)}")

    # 3. Fill missing values with mean/median
    data_filled = ecological_data.copy()
    data_filled['soil_pH'].fillna(data_filled['soil_pH'].mean(), inplace=True)
    data_filled['nitrogen'].fillna(data_filled['nitrogen'].median(), inplace=True)

    # 4. Forward fill or backward fill
    data_ffill = ecological_data.sort_values('site_id').fillna(method='ffill')

    # 5. Fill with group means
    data_group_fill = ecological_data.copy()
    data_group_fill['soil_pH'] = data_group_fill.groupby('habitat')['soil_pH'].transform(
        lambda x: x.fillna(x.mean())
    )

    print(f"After filling missing values: {data_filled.isnull().sum().sum()} missing values remain")
    return


@app.cell
def _(mo):
    mo.md(r"""## String Operations for Ecological Data""")
    return


@app.cell
def _(np, pd):
    # Create species data with scientific names
    species_names = [
        "Quercus alba", "Acer saccharum", "Betula papyrifera", 
        "Pinus strobus", "Fagus grandifolia", "Tsuga canadensis",
        "Fraxinus americana", "Tilia americana", "Ulmus americana"
    ]

    species_df = pd.DataFrame({
        'species_name': species_names,
        'abundance': np.random.poisson(10, len(species_names)),
        'origin': np.random.choice(['Native', 'Non-native', 'Unknown'], len(species_names))
    })

    # Extract genus and species
    species_df['genus'] = species_df['species_name'].str.split(' ').str[0]
    species_df['species'] = species_df['species_name'].str.split(' ').str[1]

    # String operations
    species_df['name_length'] = species_df['species_name'].str.len()
    species_df['has_americana'] = species_df['species_name'].str.contains('americana')
    species_df['genus_upper'] = species_df['genus'].str.upper()

    # Filter by string patterns
    acer_species = species_df[species_df['genus'] == 'Acer']
    americana_species = species_df[species_df['species_name'].str.contains('americana')]

    print("Species data with string operations:")
    print(species_df)

    print(f"\nAcer species: {len(acer_species)}")
    print(f"Americana species: {len(americana_species)}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Working with Dates and Times""")
    return


@app.cell
def _(np, pd, sites):
    """

    """
    # Create temporal ecological data
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2023-12-31')

    # Generate random sampling dates
    n_samples = 100
    random_dates = pd.to_datetime(
        np.random.choice(
            pd.date_range(start_date, end_date), 
            size=n_samples
        )
    )

    temporal_data = pd.DataFrame({
        'sampling_date': random_dates,
        'site_id': np.random.choice(sites[:20], n_samples),
        'bird_count': np.random.poisson(8, n_samples),
        'temperature': np.random.normal(15, 10, n_samples)
    })

    # Extract date components
    temporal_data['year'] = temporal_data['sampling_date'].dt.year
    temporal_data['month'] = temporal_data['sampling_date'].dt.month
    temporal_data['day_of_year'] = temporal_data['sampling_date'].dt.dayofyear
    temporal_data['season'] = temporal_data['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    # Group by temporal units
    monthly_summary = temporal_data.groupby(['year', 'month']).agg({
        'bird_count': ['mean', 'sum'],
        'temperature': 'mean'
    }).round(2)

    seasonal_summary = temporal_data.groupby('season')['bird_count'].agg(['mean', 'std']).round(2)

    print("Temporal data sample:")
    print(temporal_data.head())

    print("\nSeasonal bird count summary:")
    print(seasonal_summary)
    return


@app.cell
def _(mo):
    mo.md(r"""## Chaining Operations""")
    return


@app.cell
def _(ecological_data, np):
    # Combine multiple operations in a single chain
    # Similar to R's pipe operator %>%

    processed_data = (
        ecological_data
        .query('species_richness > 10')  # Filter
        .assign(                         # Add new columns
            temp_celsius=lambda x: x['temperature'],
            temp_kelvin=lambda x: x['temperature'] + 273.15,
            log_abundance=lambda x: np.log(x['total_abundance'])
        )
        .groupby('habitat')              # Group by habitat
        .agg({                          # Aggregate
            'species_richness': 'mean',
            'temp_celsius': 'mean',
            'log_abundance': 'mean'
        })
        .round(2)                       # Round results
        .sort_values('species_richness', ascending=False)  # Sort
    )

    print("Chained operations result:")
    print(processed_data)

    # Alternative syntax using backslashes for line continuation
    processed_data_alt = ecological_data \
        .query('temperature > 10 and precipitation > 500') \
        .groupby('habitat')['species_richness'] \
        .agg(['count', 'mean', 'std']) \
        .round(2)

    print("\nAlternative chaining syntax:")
    print(processed_data_alt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Performance Tips for Large Ecological Datasets

    **Memory optimization**:
    - Use appropriate data types (category for strings, int32 for small integers)
    - Read data in chunks for very large files
    - Use query() instead of boolean indexing for better performance

    **Efficient operations**:
    - Vectorized operations are faster than loops
    - Use groupby().agg() instead of multiple groupby operations
    - Prefer loc/iloc for indexing over chained indexing

    **Best practices**:
    - Validate data types after reading
    - Check for memory usage with .memory_usage()
    - Use .copy() when modifying subsets to avoid warnings
    """
    )
    return


@app.cell
def _(ecological_data):
    # Demonstrate data type optimization
    optimized_data = ecological_data.copy()

    # Convert habitat to category (saves memory)
    optimized_data['habitat'] = optimized_data['habitat'].astype('category')

    # Check memory usage
    original_memory = ecological_data.memory_usage(deep=True).sum()
    optimized_memory = optimized_data.memory_usage(deep=True).sum()

    print(f"Original memory usage: {original_memory:,} bytes")
    print(f"Optimized memory usage: {optimized_memory:,} bytes")
    print(f"Memory saved: {original_memory - optimized_memory:,} bytes ({((original_memory - optimized_memory)/original_memory*100):.1f}%)")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary

    In this chapter, we covered essential pandas operations for ecological data:

    - **DataFrame creation and inspection**
    - **Selecting and filtering** (select, filter equivalents)
    - **Creating new variables** (mutate equivalent)
    - **Sorting data** (arrange equivalent)
    - **Grouping and summarizing** (group_by, summarise equivalents)
    - **Reshaping data** (pivot_longer, pivot_wider equivalents)
    - **Joining datasets** (left_join, inner_join equivalents)
    - **Handling missing data** (various strategies)
    - **String operations** for species names and text data
    - **Date/time operations** for temporal ecological data
    - **Method chaining** for readable data pipelines

    **Next chapter**: Data visualization with holoviews

    **Key pandas functions for ecology**:
    - `.query()` for filtering with complex conditions
    - `.groupby().agg()` for habitat/species summaries
    - `.pivot_table()` for reshaping community matrices
    - `.merge()` for combining environmental and species data
    - `.fillna()` and `.dropna()` for missing data handling
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
