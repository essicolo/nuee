import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Chapter 3: Data Organization and Table Operations

        ## Learning Objectives

        By the end of this chapter, you will:

        - Understand the rules guiding table creation and management
        - Know how to import and export data
        - Know how to perform cascading operations with pandas, including:
          - Filtering rows
          - Selecting columns
          - Statistical summaries
          - Joining tables

        ---

        Data is used at every stage in scientific workflows. It fuels analysis and modeling. The results that emerge are also data that can feed subsequent work. Good data organization facilitates workflow.

        > **Adage**: Proportions of time devoted to scientific computing: 80% cleaning poorly organized data, 20% computation.

        What is data? Abstractly, it's a value associated with a variable. A variable can be a dimension, a date, a color, the result of a statistical test, to which we assign the quantitative or qualitative value of a number, a string, a conventional symbol, etc. For example, when you order a vegan latte, *latte* is the value you attribute to the variable *type of coffee*, and *vegan* is the value of the variable *type of milk*.

        This chapter covers importing, using, and exporting structured data in Python, in the form of arrays, matrices, tables, and sets of tables (databases).

        Although it's always preferable to organize the structures that will hold experimental data before even collecting data, analysts must expect to reorganize their data along the way. However, well-organized data from the start will also facilitate their reorganization.

        ## Data Collections

        We've seen that the preferred way to organize data is as **tables** (DataFrames in pandas). Generally, a data table is a two-dimensional organization of data, containing *rows* and *columns*. It's preferable to respect the convention where **rows are observations and columns are variables**. Thus, a table is a collection of vectors of the same length, each vector representing a variable. Each variable is free to take the appropriate data type. The position of a datum in the vector corresponds to an observation.

        Imagine you're recording meteorological data like total precipitation or average temperature for each day during a week at sites A, B, and C. Each site has its own characteristics, like longitude-latitude position. It's redundant to repeat the site position for each day of the week. You'll prefer to create two tables: one to describe your observations, and another to describe the sites. This way, you create a collection of interrelated tables: a **database**.
        """
    )
    return


@app.cell
def __():
    # Essential imports
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    # Set display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print("Pandas version:", pd.__version__)
    return np, pd, warnings


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Organizing a Data Table

        To locate each cell in a table, we assign each row and column a *unique* identifier, called an *index* for rows and *column names* (or headers) for columns.

        > **Rule #1**: One variable per column, one observation per row, one value per cell.

        Experimental units are described by one or more variables using numbers or letters. Each variable should be present in a single column, and each row should correspond to an experimental unit where these variables were measured. The rule seems simple, but it's rarely respected. Take for example this table:
        """
    )
    return


@app.cell
def _(np, pd):
    # Example of WIDE format (BAD for analysis)
    data_wide = pd.DataFrame({
        'Site': ['Sainte-Souris', 'Sainte-Fourmi', 'Saint-Ours'],
        'Treatment A': [4.1, 5.8, 2.9],
        'Treatment B': [8.2, 5.9, 3.4],
        'Treatment C': [6.8, np.nan, 4.6]
    })

    print("WIDE format (less ideal for analysis):")
    print(data_wide)
    return data_wide,


@app.cell
def _(mo):
    mo.md(
        r"""
        What's wrong with this table? Each row is an observation, but contains multiple observations of the same variable, the yield, which is spread across multiple columns. *Thinking about it*, treatment type is one variable and yield is another:
        """
    )
    return


@app.cell
def _(data_wide):
    # Convert to LONG format (GOOD for analysis)
    data_long = data_wide.melt(
        id_vars='Site',
        var_name='Treatment',
        value_name='Yield'
    )

    print("LONG format (better for analysis):")
    print(data_long)
    return data_long,


@app.cell
def _(mo):
    mo.md(
        r"""
        > **Rule #2**: One table per observational unit: don't repeat information.

        Let's return to the same experiment. Suppose you measure precipitation at the site level:
        """
    )
    return


@app.cell
def _(data_long, np):
    # Bad: Repeated precipitation values
    data_long_precip = data_long.copy()
    data_long_precip['Precipitation'] = [813, 813, 813, 642, 642, 642, 1028, 1028, 1028]

    print("Table with repeated information (NOT IDEAL):")
    print(data_long_precip)
    return data_long_precip,


@app.cell
def _(mo):
    mo.md(
        r"""
        Segmenting the information into two tables would be preferable:
        """
    )
    return


@app.cell
def _(pd):
    # Better: Separate tables
    sites_info = pd.DataFrame({
        'Site': ['Sainte-Souris', 'Sainte-Fourmi', 'Saint-Ours'],
        'Precipitation': [813, 642, 1028]
    })

    print("Sites information table:")
    print(sites_info)
    return sites_info,


@app.cell
def _(mo):
    mo.md(
        r"""
        These two tables together form a database (organized collection of tables). Merge operations between tables can be performed using join functions (`merge()` or `join()` in pandas).

        > **Rule #3**: Don't mess up the data.

        For example:

        - **Adding comments in cells**: If a cell deserves to be commented, it's preferable to place comments either in a file describing the data table, or in a comment column next to the variable column to be commented. For example, if you didn't measure pH for an observation, don't write "contaminated sample" in the cell, but annotate in an explanation file that sample no. X was contaminated. If comments are systematic, it can be convenient to record them in a `pH_comment` column.

        - **Non-systematic notation**: It often happens that categories of a variable or missing values are annotated differently. It even happens that the decimal separator is non-systematic, sometimes noted by a period, sometimes by a comma. For example, once imported into your session, the categories `St-Ours` and `Saint-Ours` will be treated as two distinct categories. Similarly, cells corresponding to missing values should not be entered sometimes with an empty cell, sometimes with a period, sometimes with a dash or with the mention `NA`. The simplest is to systematically leave these cells empty (or use `np.nan`).

        - **Including notes in a table**: The rule "one column, one variable" is not respected if you add notes anywhere under or next to the table.

        - **Adding summaries**: If you add a row under a table containing the average of each column, what will happen when you import your table into your working session? The row will be considered an additional observation.

        - **Including hierarchy in headers**: To record soil texture data, including the proportion of sand, silt, and clay, you organize your header in multiple rows. One row for the data category, *Texture*, merged over three columns, then three columns titled *Sand*, *Silt*, and *Clay*. Your table is pretty, but it cannot be imported properly into your computing session: we're looking for *a unique header per column*. Your data table should instead have headers *Texture_sand*, *Texture_silt*, and *Texture_clay*. Advice: reserve aesthetic work for the very end of a workflow.

        ## Data Formats

        ### CSV

        The CSV format, for *comma separated values*, is a text file that you can open with any plain text editor. Each column must be delimited by a consistent character (conventionally a comma, but a semicolon or tab in some regions) and each table row is a line break. CSV files can be opened and edited in text editors, but it's more convenient to open them with spreadsheet applications.

        **Text file encoding**: Since CSV format is a text file, particular care must be taken with how text is encoded. Accented characters could be imported incorrectly if you import your table specifying the wrong encoding. For files in Western languages, UTF-8 encoding should be used.

        ### JSON

        Like CSV format, JSON format indicates a plain text file. By allowing nested table structures and not requiring each column to have the same length, JSON format allows more flexibility than CSV format, but it's more complicated to view and takes more disk space than CSV. It's used more for sharing data from web applications, but for this course's material, this format is mainly used for georeferenced data.

        ### Parquet

        Parquet is a columnar storage format that's very efficient for large datasets. It's compressed, fast to read, and preserves data types perfectly. It's becoming increasingly popular in data science.

        ### SQLite

        SQLite is an application for SQL-type relational databases that doesn't need a server to function. SQLite databases are encoded in files with the `.db` extension, which can be easily shared.

        ### Suggestion

        CSV for small tables, Parquet for medium to large datasets, SQLite for more complex databases. This course focuses mainly on CSV-type data.

        ## Importing Data into Your Working Session

        Suppose you've organized your data in tidy mode. To import them into your session and start inspecting them, you'll use one of the pandas read functions:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Reading Data Files

        Main pandas functions for reading data:

        - `pd.read_csv()` - if the column separator is a comma
        - `pd.read_csv(sep=';')` - if the column separator is a semicolon
        - `pd.read_csv(sep='\t')` - if the column separator is a tab
        - `pd.read_table()` - if the column separator is whitespace
        - `pd.read_excel()` - for Excel files (xls, xlsx)
        - `pd.read_parquet()` - for Parquet files
        - `pd.read_json()` - for JSON files

        Key arguments:

        - `filepath_or_buffer`: path to the file (can be local path or URL)
        - `sep`: the symbol delimiting columns
        - `header`: row number to use as column names (default=0, first row)
        - `names`: list of column names to use
        - `na_values`: additional strings to recognize as NA/NaN
        - `encoding`: file encoding (usually 'utf-8')
        - `dtype`: specify data types for columns
        - `parse_dates`: list of columns to parse as dates

        ### Useful Commands to Inspect a DataFrame

        - `df.head()` - presents the table header, i.e., its first 5 rows
        - `df.tail()` - presents the last 5 rows
        - `df.info()` - presents the variables of the table and their type
        - `df.describe()` - presents basic statistics of the table
        - `df.columns` - returns column names as a list
        - `df.shape` - gives the dimensions of the table
        - `df.dtypes` - shows the data type of each column

        Let's create and inspect a sample ecological dataset:
        """
    )
    return


@app.cell
def _(np, pd):
    # Create a sample ecological dataset (cloudberry cultivation)
    np.random.seed(42)

    cloudberry = pd.DataFrame({
        'Site': ['BEAU'] * 10 + ['MB'] * 10 + ['WTP'] * 10,
        'PeatlandCode': ['BEAU'] * 10 + ['MB'] * 10 + ['WTP'] * 10,
        'Latitude_m': np.random.normal(48.5, 0.5, 30),
        'Longitude_m': np.random.normal(-71.2, 0.5, 30),
        'N_perc': np.random.uniform(0.5, 2.5, 30),
        'P_perc': np.random.uniform(0.05, 0.3, 30),
        'K_perc': np.random.uniform(0.1, 0.6, 30),
        'Ca_perc': np.random.uniform(0.2, 0.8, 30),
        'Yield_g_5m2': np.random.normal(150, 50, 30),
        'TotalFloral_number_m2': np.random.poisson(25, 30)
    })

    print("Cloudberry dataset:")
    print(cloudberry.head())
    print(f"\nShape: {cloudberry.shape}")
    print(f"\nData types:\n{cloudberry.dtypes}")
    return cloudberry,


@app.cell
def _(cloudberry):
    # Inspection commands
    print("Summary statistics:")
    print(cloudberry.describe())

    print("\n\nDataFrame info:")
    cloudberry.info()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Selecting and Filtering Data

        The term **select** is used when we want to choose one or more rows and columns of a table (most often columns). The action of **filtering** means selecting rows according to certain criteria.

        ### Selecting Columns

        In pandas, there are several ways to select columns:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Method 1: Using column names directly
    print("Method 1 - Direct column access:")
    print(cloudberry['Site'].head())

    # Method 2: Using dot notation (only for valid Python identifiers)
    print("\n\nMethod 2 - Dot notation:")
    print(cloudberry.Site.head())

    # Method 3: Selecting multiple columns
    print("\n\nMethod 3 - Multiple columns:")
    print(cloudberry[['Site', 'Latitude_m', 'Longitude_m']].head())

    # Method 4: Using loc for label-based indexing
    print("\n\nMethod 4 - loc for label-based indexing:")
    print(cloudberry.loc[:, ['Site', 'N_perc', 'P_perc']].head())

    # Method 5: Using iloc for integer-based indexing
    print("\n\nMethod 5 - iloc for position-based indexing:")
    print(cloudberry.iloc[:, 0:3].head())  # First 3 columns
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Selecting Columns by Pattern

        Pandas provides useful methods for selecting columns by pattern:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Select columns containing "perc"
    perc_cols = cloudberry.filter(like='perc')
    print("Columns containing 'perc':")
    print(perc_cols.head())

    # Select columns ending with "m"
    m_cols = cloudberry.filter(regex='_m$')
    print("\n\nColumns ending with '_m':")
    print(m_cols.head())

    # Select columns starting with a pattern
    coord_cols = cloudberry.filter(regex='^(Latitude|Longitude)')
    print("\n\nCoordinate columns:")
    print(coord_cols.head())
    return coord_cols, m_cols, perc_cols


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Filtering Rows

        Filtering is done using boolean indexing or the `query()` method:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Method 1: Boolean indexing (classic method)
    beau_sites = cloudberry[cloudberry['PeatlandCode'] == 'BEAU']
    print("Sites with PeatlandCode == 'BEAU':")
    print(beau_sites.head())

    # Method 2: Using query() (more readable for complex queries)
    filtered = cloudberry.query("Ca_perc < 0.4 and PeatlandCode in ['BEAU', 'MB', 'WTP']")
    print("\n\nFiltered using query:")
    print(filtered[['PeatlandCode', 'Ca_perc']].head())

    # Method 3: Multiple conditions with &, |
    multi_filter = cloudberry[(cloudberry['N_perc'] > 1.0) & (cloudberry['Yield_g_5m2'] > 150)]
    print("\n\nMultiple conditions:")
    print(multi_filter[['N_perc', 'Yield_g_5m2']].head())
    return beau_sites, filtered, multi_filter


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Method Chaining (Pandas Pipeline)

        In pandas, we can chain operations similar to R's tidyverse pipe (`%>%`). This is done by chaining methods with dots:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Chained operations
    result = (cloudberry
              .query("Ca_perc < 0.4 and PeatlandCode in ['BEAU', 'MB']")
              .filter(like='perc')
              .head(10))

    print("Chained operations result:")
    print(result)
    return result,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Long and Wide Format

        In the cloudberry table, each element has its own column. If we wanted to graph boxplots of concentrations of nitrogen, phosphorus, and potassium in different peatlands, we'd need a single concentration column.

        To do this, we use `melt()` (equivalent to R's `pivot_longer()`):
        """
    )
    return


@app.cell
def _(cloudberry):
    # Convert to long format
    cloudberry_long = (cloudberry
                       [['PeatlandCode', 'N_perc', 'P_perc', 'K_perc']]
                       .reset_index()
                       .rename(columns={'index': 'ID'})
                       .melt(id_vars=['ID', 'PeatlandCode'],
                             var_name='nutrient',
                             value_name='concentration'))

    print("Long format:")
    print(cloudberry_long.sample(10).sort_index())
    return cloudberry_long,


@app.cell
def _(mo):
    mo.md(
        r"""
        The inverse operation is `pivot()` or `pivot_table()` (equivalent to R's `pivot_wider()`):
        """
    )
    return


@app.cell
def _(cloudberry_long):
    # Convert back to wide format
    cloudberry_wide = (cloudberry_long
                       .pivot(index='ID',
                              columns='nutrient',
                              values='concentration')
                       .reset_index())

    print("Wide format (reconstructed):")
    print(cloudberry_wide.sample(10))
    return cloudberry_wide,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Combining Tables

        We introduced the notion of database earlier. We might want to use peatland codes to include their name, the type of test conducted at these peatlands, etc. Let's create a peatland information table:
        """
    )
    return


@app.cell
def _(pd):
    # Create peatland information table
    peatlands = pd.DataFrame({
        'PeatlandCode': ['BEAU', 'MB', 'WTP'],
        'PeatlandName': ['Beaumont', 'Mont-Blanc', 'Water Treatment Plant'],
        'TrialType': ['Fertilization', 'Control', 'Fertilization']
    })

    print("Peatland information:")
    print(peatlands)
    return peatlands,


@app.cell
def _(mo):
    mo.md(
        r"""
        Our information is organized in two tables, linked by the `PeatlandCode` column. How to merge the information so it can be used as a whole? The `merge()` function performs this typical database operation:
        """
    )
    return


@app.cell
def _(cloudberry, peatlands):
    # Left join (equivalent to R's left_join)
    cloudberry_merged = cloudberry.merge(peatlands, on='PeatlandCode', how='left')

    print("Merged table:")
    print(cloudberry_merged.sample(4))
    print(f"\nNew columns: {cloudberry_merged.columns.tolist()}")
    return cloudberry_merged,


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Types of Joins

        Different types of joins are possible in pandas:

        - **`merge(df1, df2, how='left')`** - keeps all rows from left table, adds matching rows from right
        - **`merge(df1, df2, how='right')`** - keeps all rows from right table, adds matching rows from left
        - **`merge(df1, df2, how='inner')`** - keeps only rows that match in both tables
        - **`merge(df1, df2, how='outer')`** - keeps all rows from both tables (full join)

        Example of different join types:
        """
    )
    return


@app.cell
def _(pd):
    # Create example tables
    df_a = pd.DataFrame({'key': ['A', 'B', 'C'], 'value_a': [1, 2, 3]})
    df_b = pd.DataFrame({'key': ['B', 'C', 'D'], 'value_b': [4, 5, 6]})

    print("Table A:")
    print(df_a)
    print("\nTable B:")
    print(df_b)

    print("\n\nInner join:")
    print(pd.merge(df_a, df_b, on='key', how='inner'))

    print("\n\nLeft join:")
    print(pd.merge(df_a, df_b, on='key', how='left'))

    print("\n\nRight join:")
    print(pd.merge(df_a, df_b, on='key', how='right'))

    print("\n\nOuter join:")
    print(pd.merge(df_a, df_b, on='key', how='outer'))
    return df_a, df_b


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Operations on Tables

        Tables can be segmented into elements on which we'll calculate whatever we want.

        We might want to obtain:

        - Sum with `sum()`
        - Mean with `mean()` or median with `median()`
        - Standard deviation with `std()`
        - Maximum and minimum with `max()` and `min()`
        - Count of occurrences with `count()` or `value_counts()`

        ### Basic Statistics
        """
    )
    return


@app.cell
def _(cloudberry):
    # Single column statistics
    print(f"Mean yield: {cloudberry['Yield_g_5m2'].mean():.2f}")
    print(f"Median yield: {cloudberry['Yield_g_5m2'].median():.2f}")
    print(f"Std yield: {cloudberry['Yield_g_5m2'].std():.2f}")

    # Statistics on multiple columns
    print("\n\nNutrient statistics:")
    print(cloudberry.filter(like='perc').mean())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### GroupBy Operations

        The `groupby()` method is one of the most powerful features in pandas. It allows you to:
        1. Split the data into groups
        2. Apply a function to each group independently
        3. Combine the results

        This is equivalent to R's `group_by()` + `summarise()`:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Group by and calculate statistics
    grouped_stats = (cloudberry
                     .groupby('PeatlandCode')
                     .agg({
                         'TotalFloral_number_m2': ['mean', 'std'],
                         'Yield_g_5m2': ['mean', 'std']
                     })
                     .round(2))

    print("Statistics by peatland:")
    print(grouped_stats)
    return grouped_stats,


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Multiple Aggregations

        We can apply multiple aggregation functions at once:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Multiple aggregations with agg()
    nutrient_stats = (cloudberry
                      .groupby('PeatlandCode')[['N_perc', 'P_perc', 'K_perc']]
                      .agg(['mean', 'std', 'min', 'max'])
                      .round(3))

    print("Nutrient statistics by peatland:")
    print(nutrient_stats)
    return nutrient_stats,


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Adding Computed Columns

        The `assign()` method (equivalent to R's `mutate()`) adds new columns:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Add computed columns
    cloudberry_with_computed = (cloudberry
                                .assign(
                                    total_NPK=lambda df: df['N_perc'] + df['P_perc'] + df['K_perc'],
                                    yield_per_flower=lambda df: df['Yield_g_5m2'] / df['TotalFloral_number_m2'],
                                    high_yield=lambda df: df['Yield_g_5m2'] > 150
                                ))

    print("Table with computed columns:")
    print(cloudberry_with_computed[['Site', 'Yield_g_5m2', 'yield_per_flower', 'high_yield']].head())
    return cloudberry_with_computed,


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Complex Pipeline Example

        Let's create a complex analysis pipeline similar to the endangered species example from the R book:
        """
    )
    return


@app.cell
def _(np, pd):
    # Create a synthetic endangered species dataset
    np.random.seed(42)

    species_data = pd.DataFrame({
        'Country': np.random.choice(['USA', 'Canada', 'Brazil', 'Ecuador', 'Madagascar',
                                    'Indonesia', 'Australia', 'China', 'India', 'Mexico'], 200),
        'IUCN': np.random.choice(['CRITICAL', 'ENDANGERED', 'VULNERABLE'], 200),
        'Species_Type': np.random.choice(['VASCULAR_PLANT', 'MAMMAL', 'BIRD', 'REPTILE'], 200),
        'Count': np.random.randint(1, 50, 200)
    })

    # Complex pipeline
    top_countries = (species_data
                     .query("IUCN == 'CRITICAL' and Species_Type == 'VASCULAR_PLANT'")
                     [['Country', 'Count']]
                     .groupby('Country')
                     .agg(n_critical_plants=('Count', 'sum'))
                     .sort_values('n_critical_plants', ascending=False)
                     .head(10))

    print("Top 10 countries with critical vascular plants:")
    print(top_countries)
    return species_data, top_countries


@app.cell
def _(mo):
    mo.md(
        r"""
        This pipeline consists of:

        ```
        Take the species_data table, then
          Filter to get only critical species in the vascular plant category, then
          Select the country and count columns, then
          Group the table by country, then
          Apply the sum function for each group (then recombine these summaries), then
          Sort countries in descending order of species count, then
          Display the top 10
        ```

        ## Sorting Data

        The `sort_values()` method sorts data by one or more columns:
        """
    )
    return


@app.cell
def _(cloudberry):
    # Sort by single column
    sorted_yield = cloudberry.sort_values('Yield_g_5m2', ascending=False)
    print("Sorted by yield (descending):")
    print(sorted_yield[['Site', 'Yield_g_5m2']].head())

    # Sort by multiple columns
    sorted_multi = cloudberry.sort_values(['PeatlandCode', 'N_perc'], ascending=[True, False])
    print("\n\nSorted by peatland (asc) and N percentage (desc):")
    print(sorted_multi[['PeatlandCode', 'N_perc']].head())
    return sorted_multi, sorted_yield


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Exporting Data

        Once you've cleaned and processed your data, you'll want to export it:
        """
    )
    return


@app.cell
def _(cloudberry_merged):
    # Export to CSV
    # cloudberry_merged.to_csv('data/cloudberry_processed.csv', index=False)

    # Export to Parquet (more efficient for large datasets)
    # cloudberry_merged.to_parquet('data/cloudberry_processed.parquet')

    # Export to Excel
    # cloudberry_merged.to_excel('data/cloudberry_processed.xlsx', index=False)

    print("Export methods available:")
    print("- to_csv(): CSV format")
    print("- to_parquet(): Parquet format (recommended for large data)")
    print("- to_excel(): Excel format")
    print("- to_json(): JSON format")
    print("- to_sql(): SQL database")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        In this chapter, you learned:

        - **Data organization principles**: Tidy data rules for effective analysis
        - **Data formats**: CSV, JSON, Parquet, Excel, SQLite
        - **Importing data**: Using `pd.read_csv()` and related functions
        - **Selecting data**: Multiple methods to select columns and rows
        - **Filtering data**: Boolean indexing and `query()` method
        - **Reshaping data**: `melt()` for long format, `pivot()` for wide format
        - **Combining tables**: `merge()` for different types of joins
        - **Aggregating data**: `groupby()` and `agg()` for group-wise operations
        - **Adding columns**: `assign()` for computed columns
        - **Chaining operations**: Method chaining for readable pipelines
        - **Exporting data**: Saving processed data in various formats

        ### Key Pandas Methods Summary

        | Operation | Pandas Method | R tidyverse Equivalent |
        |-----------|---------------|------------------------|
        | Select columns | `df[['col1', 'col2']]` or `df.filter()` | `select()` |
        | Filter rows | `df.query()` or `df[condition]` | `filter()` |
        | Add columns | `df.assign()` | `mutate()` |
        | Group by | `df.groupby()` | `group_by()` |
        | Summarize | `df.agg()` | `summarise()` |
        | Sort | `df.sort_values()` | `arrange()` |
        | Pivot longer | `df.melt()` | `pivot_longer()` |
        | Pivot wider | `df.pivot()` | `pivot_wider()` |
        | Join tables | `df.merge()` | `left_join()`, etc. |

        In the next chapter, we'll explore data visualization using lets-plot, the Python equivalent of ggplot2!
        """
    )
    return


if __name__ == "__main__":
    app.run()
