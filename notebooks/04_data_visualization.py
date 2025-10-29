import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 4: Data Visualization with plotnine

    This chapter covers creating effective visualizations for ecological data using plotnine, a powerful declarative visualization library that works seamlessly in Pyodide environments.

    ## Learning Objectives

    - Master plotnine for ecological data visualization
    - Create publication-quality plots for ecological research
    - Build interactive visualizations and dashboards
    - Understand design principles for ecological graphics
    """
    )
    return


@app.cell
def _():
    # Essential imports for visualization
    import pandas as pd
    import numpy as np
    import plotnine as p9
    return np, p9, pd


@app.cell
def _(mo):
    mo.md(r"""## Create Sample Ecological Dataset""")
    return


@app.cell
def _(np, pd):
    # Generate comprehensive ecological data for visualization examples
    np.random.seed(42)

    # Site-level data
    _n_sites = 100
    sites = [f"SITE_{i:03d}" for i in range(1, _n_sites + 1)]

    ecological_data1 = pd.DataFrame({
        'site_id': sites,
        'habitat': np.random.choice(['Forest', 'Grassland', 'Wetland', 'Urban', 'Shrubland'], _n_sites, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'latitude': np.random.uniform(45.0, 48.0, _n_sites),
        'longitude': np.random.uniform(-75.0, -70.0, _n_sites),
        'elevation': np.random.uniform(50, 1200, _n_sites),
        'temperature': np.random.normal(15, 5, _n_sites),
        'precipitation': np.random.lognormal(6, 0.5, _n_sites),
        'soil_pH': np.random.normal(6.5, 1.2, _n_sites),
        'nitrogen': np.random.gamma(2, 5, _n_sites),
        'species_richness': np.random.poisson(15, _n_sites),
        'shannon_diversity': np.random.gamma(2, 0.7, _n_sites),
        'total_abundance': np.random.lognormal(4, 0.8, _n_sites)
    })

    # Add derived variables
    ecological_data1['simpson_diversity'] = 1 - (1 / (ecological_data1['shannon_diversity'] + 1))
    ecological_data1['evenness'] = ecological_data1['shannon_diversity'] / np.log(ecological_data1['species_richness'])

    ecological_data1['habitat'] = ecological_data1['habitat'].astype('category')

    print(f"Dataset created: {ecological_data1.shape}")
    print(ecological_data1.head())
    return (ecological_data1,)


@app.cell
def _(mo):
    mo.md(r"""## Basic Scatter Plots""")
    return


@app.cell
def _(ecological_data1, p9):
    (
        p9.ggplot(ecological_data1, p9.aes(x='temperature', y='species_richness'))
        + p9.geom_point()
        + p9.labs(
            title='Species Richness vs Temperature by Habitat',
            x='Temperature (°C)',
            y='Species Richness'
        )
    ).show()
    return


@app.cell
def _(ecological_data1, p9):
    (
        p9.ggplot(ecological_data1, p9.aes(x='temperature', y='species_richness'))
        + p9.geom_point(p9.aes(color='habitat'), size=3, alpha=0.7)
        + p9.theme_bw() # a theme
        + p9.labs(
            title='Species Richness vs Temperature by Habitat',
            x='Temperature (°C)',
            y='Species Richness'
        )
    ).show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Distributions and Histograms""")
    return


@app.cell
def _(ecological_data1, p9):
    (
        p9.ggplot(ecological_data1, p9.aes(x='species_richness'))
        + p9.geom_histogram(bins=20)
    ).draw()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Correlation and Relationship Plots

    To plot a correlation matrix, you must compute it then make a table out of it. The table must be in long format to have the variable in a column, the variable is correlates to in another column, then the value of the correlation in the other.
    """
    )
    return


@app.cell
def _(ecological_data1):
    numeric_cols = ['temperature', 'precipitation', 'soil_pH', 'nitrogen', 
                   'species_richness', 'shannon_diversity', 'total_abundance']
    corr_matrix = ecological_data1[numeric_cols].corr()
    corr_matrix = corr_matrix.corr().round(2)
    corr_long = (
        corr_matrix
        .reset_index() # the row-wise variable is stored in the index, resetting put the variable in a column named "index" and makes a new index
        .rename(columns={'index':'var1'})
        .melt(id_vars='var1', var_name='var2', value_name='corr')
    
    )
    corr_long
    return (corr_long,)


@app.cell
def _(corr_long, p9):
    (
        p9.ggplot(corr_long, p9.aes(x='var1', y='var2', fill='corr'))
        + p9.geom_tile()
        + p9.geom_text(p9.aes(label='corr'), size=8)
        + p9.scale_fill_gradient2(low='#2166ac', mid='#f7f7f7', high='#b2182b', midpoint=0)
        + p9.labs(x='', y='', title='Correlation matrix')
        + p9.theme_minimal()
        + p9.theme(
            axis_text_x=p9.element_text(rotation=45, ha='right', size=8),
            axis_text_y=p9.element_text(size=8),
            plot_title=p9.element_text(size=11, weight='bold'),
            figure_size=(6,6)
        )
    ).draw()
    return


@app.cell
def _():
    ## Time Series and Temporal Data
    return


@app.cell
def _(hv, np, pd):
    """

    """
    # Create temporal data for demonstration
    start_date = pd.to_datetime('2020-01-01')
    dates = pd.date_range(start_date, periods=365*3, freq='D')

    # Simulate seasonal patterns in biodiversity
    day_of_year = dates.dayofyear
    seasonal_pattern = np.sin(2 * np.pi * day_of_year / 365) * 5 + 15
    random_noise = np.random.normal(0, 2, len(dates))
    temperature_ts = seasonal_pattern + random_noise

    # Simulate species observations with seasonal patterns
    species_activity = np.sin(2 * np.pi * day_of_year / 365 + np.pi/4) * 10 + 20 + np.random.normal(0, 3, len(dates))
    species_activity = np.maximum(species_activity, 0)  # No negative observations

    temporal_data = pd.DataFrame({
        'date': dates,
        'temperature': temperature_ts,
        'species_observations': species_activity,
        'year': dates.year,
        'month': dates.month,
        'day_of_year': day_of_year
    })

    # Time series plot
    temp_ts = hv.Curve(temporal_data, 'date', 'temperature').opts(
        title="Temperature Time Series",
        xlabel="Date",
        ylabel="Temperature (°C)",
        color='red',
        line_width=1
    )

    species_ts = hv.Curve(temporal_data, 'date', 'species_observations').opts(
        title="Species Observations Over Time",
        xlabel="Date", 
        ylabel="Number of Observations",
        color='green',
        line_width=1
    )

    # Seasonal decomposition visualization
    monthly_avg = temporal_data.groupby(['year', 'month']).agg({
        'temperature': 'mean',
        'species_observations': 'mean'
    }).reset_index()

    monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))

    monthly_temp = hv.Curve(monthly_avg, 'date', 'temperature').opts(
        title="Monthly Average Temperature",
        color='orange',
        line_width=3
    )

    monthly_species = hv.Curve(monthly_avg, 'date', 'species_observations').opts(
        title="Monthly Average Species Observations",
        color='blue',
        line_width=3
    )

    print("Temporal patterns:")
    (temp_ts + species_ts + monthly_temp + monthly_species).cols(2)
    return


@app.cell
def _(ecological_data, hv, pn):
    """
    ## Interactive Visualizations
    """
    # Interactive scatter plot with widgets
    def interactive_scatter(x_var='temperature', y_var='species_richness', habitat_filter='All'):
        data = ecological_data.copy()
        if habitat_filter != 'All':
            data = data[data['habitat'] == habitat_filter]

        return hv.Scatter(data, x_var, y_var).opts(
            title=f"{y_var} vs {x_var}" + (f" ({habitat_filter})" if habitat_filter != 'All' else ""),
            size=8,
            alpha=0.7,
            tools=['hover']
        )

    # Create interactive plot with panel widgets
    x_select = pn.widgets.Select(
        name='X Variable', 
        value='temperature',
        options=['temperature', 'precipitation', 'soil_pH', 'nitrogen', 'elevation']
    )

    y_select = pn.widgets.Select(
        name='Y Variable',
        value='species_richness', 
        options=['species_richness', 'shannon_diversity', 'total_abundance', 'simpson_diversity']
    )

    habitat_select = pn.widgets.Select(
        name='Habitat Filter',
        value='All',
        options=['All'] + list(ecological_data['habitat'].unique())
    )

    # Linked brushing example
    scatter1 = hv.Scatter(ecological_data, 'temperature', 'species_richness').opts(
        tools=['box_select', 'lasso_select'],
        title="Temperature vs Species Richness"
    )

    scatter2 = hv.Scatter(ecological_data, 'precipitation', 'shannon_diversity').opts(
        tools=['box_select', 'lasso_select'],
        title="Precipitation vs Shannon Diversity"
    )

    # Selection streams for brushing
    selection1 = hv.streams.Selection1D(source=scatter1)
    selection2 = hv.streams.Selection1D(source=scatter2)

    print("Interactive plots with selection:")
    scatter1 + scatter2
    return


@app.cell
def _(ecological_data, hv, np, pd):
    """
    ## Ecological Specialized Plots
    """
    # Species accumulation curve
    np.random.seed(42)
    n_samples = 20
    cumulative_species = []

    for i in range(1, n_samples + 1):
        # Simulate cumulative species discovery
        base_species = 5 * np.log(i + 1) + np.random.normal(0, 0.5)
        cumulative_species.append(max(1, base_species))

    accumulation_data = pd.DataFrame({
        'samples': range(1, n_samples + 1),
        'species_count': np.cumsum([1, 2, 1, 3, 1, 1, 2, 0, 1, 2, 1, 0, 1, 1, 0, 2, 1, 0, 1, 1])
    })

    accumulation_curve = hv.Curve(accumulation_data, 'samples', 'species_count').opts(
        title="Species Accumulation Curve",
        xlabel="Number of Samples",
        ylabel="Cumulative Species Count",
        line_width=3,
        color='green'
    )

    # Rank-abundance plot (species abundance distribution)
    # Simulate species abundances following a log-normal distribution
    n_species = 25
    abundances = np.random.lognormal(2, 1.5, n_species)
    abundances = sorted(abundances, reverse=True)
    ranks = range(1, len(abundances) + 1)

    rank_abundance_data = pd.DataFrame({
        'rank': ranks,
        'abundance': abundances,
        'log_abundance': np.log(abundances)
    })

    rank_abundance = hv.Scatter(rank_abundance_data, 'rank', 'log_abundance').opts(
        title="Rank-Abundance Plot",
        xlabel="Species Rank",
        ylabel="Log(Abundance)",
        size=8,
        color='orange'
    )

    # Rarefaction curves for different habitats
    habitats = ecological_data['habitat'].unique()
    rarefaction_data = []

    for habitat in habitats:
        habitat_data = ecological_data[ecological_data['habitat'] == habitat]
        n_sites = len(habitat_data)

        for n in range(1, min(n_sites + 1, 21)):
            # Simulate rarefied species richness
            mean_richness = habitat_data['species_richness'].mean()
            rarefied = mean_richness * (1 - np.exp(-n/5)) + np.random.normal(0, 1)
            rarefaction_data.append({
                'habitat': habitat,
                'n_sites': n,
                'rarefied_richness': max(1, rarefied)
            })

    rarefaction_df = pd.DataFrame(rarefaction_data)

    rarefaction_curves = hv.Curve(rarefaction_df, 'n_sites', 'rarefied_richness', groupby='habitat').opts(
        title="Rarefaction Curves by Habitat",
        xlabel="Number of Sites",
        ylabel="Rarefied Species Richness",
        line_width=2
    ).overlay()

    print("Ecological diversity plots:")
    (accumulation_curve + rank_abundance + rarefaction_curves).cols(2)
    return


@app.cell
def _(ecological_data, hv, np, pd):
    """
    ## Ordination Plots
    """
    # Simulate ordination results for demonstration
    np.random.seed(42)
    n_sites = len(ecological_data)

    # Simulate PCA scores
    pca_axis1 = np.random.normal(0, 2, n_sites)
    pca_axis2 = np.random.normal(0, 1.5, n_sites)

    # Make scores somewhat related to environmental variables
    temp_effect = (ecological_data['temperature'] - ecological_data['temperature'].mean()) / ecological_data['temperature'].std()
    precip_effect = (ecological_data['precipitation'] - ecological_data['precipitation'].mean()) / ecological_data['precipitation'].std()

    pca_axis1 += temp_effect * 0.5
    pca_axis2 += precip_effect * 0.3

    ordination_data = ecological_data.copy()
    ordination_data['PCA1'] = pca_axis1
    ordination_data['PCA2'] = pca_axis2

    # PCA biplot
    site_plot = hv.Scatter(ordination_data, 'PCA1', 'PCA2').opts(
        color='habitat',
        size=8,
        alpha=0.7,
        title="PCA Ordination of Sites",
        xlabel="PC1",
        ylabel="PC2",
        cmap='Category10'
    )

    # Add environmental vectors
    vector_data = pd.DataFrame({
        'variable': ['Temperature', 'Precipitation', 'Soil pH', 'Nitrogen'],
        'PCA1': [0.8, -0.3, 0.1, 0.6],
        'PCA2': [0.2, 0.7, -0.8, 0.4],
        'PCA1_end': [0, 0, 0, 0],
        'PCA2_end': [0, 0, 0, 0]
    })

    # Create arrows for environmental vectors
    arrows = []
    for _, row in vector_data.iterrows():
        arrow = hv.Arrow(0, 0, row['PCA1'], row['PCA2']).opts(
            color='red',
            line_width=2,
            arrow_size=10
        )

        # Add text labels
        text = hv.Text(row['PCA1']*1.1, row['PCA2']*1.1, row['variable']).opts(
            color='red',
            fontsize=10
        )

        arrows.append(arrow * text)

    biplot = site_plot
    for arrow in arrows:
        biplot *= arrow

    biplot = biplot.opts(
        title="PCA Biplot with Environmental Vectors"
    )

    # NMDS stress plot
    stress_values = [0.25, 0.18, 0.12, 0.08, 0.06, 0.05, 0.049, 0.048]
    dimensions = list(range(1, len(stress_values) + 1))

    stress_plot = hv.Curve((dimensions, stress_values)).opts(
        title="NMDS Stress vs Dimensions",
        xlabel="Number of Dimensions",
        ylabel="Stress",
        line_width=3,
        color='purple'
    ) * hv.Scatter((dimensions, stress_values)).opts(
        size=10,
        color='purple'
    )

    print("Ordination visualizations:")
    biplot + stress_plot
    return


@app.cell
def _(ecological_data, hv, np, stats):
    """
    ## Publication-Quality Plots
    """
    # Create a publication-ready figure
    # Main plot: Species richness vs temperature by habitat
    main_plot = hv.Scatter(
        ecological_data, 
        'temperature', 
        'species_richness'
    ).opts(
        color='habitat',
        size=10,
        alpha=0.8,
        cmap='Set1',
        xlabel="Temperature (°C)",
        ylabel="Species Richness",
        title="Species Richness along Temperature Gradient",
        fontsize={'title': 14, 'labels': 12, 'ticks': 10},
        show_grid=True,
        grid_opts={'grid_line_alpha': 0.3},
        legend_position='top_left',
        width=600,
        height=400
    )

    # Add regression lines for each habitat
    regression_lines = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, habitat in enumerate(ecological_data['habitat'].unique()):
        habitat_data = ecological_data[ecological_data['habitat'] == habitat]

        if len(habitat_data) > 2:  # Need at least 3 points for regression
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                habitat_data['temperature'], 
                habitat_data['species_richness']
            )

            x_range = np.linspace(
                habitat_data['temperature'].min(), 
                habitat_data['temperature'].max(), 
                50
            )
            y_pred = slope * x_range + intercept

            reg_line = hv.Curve((x_range, y_pred)).opts(
                color=colors[i % len(colors)],
                line_width=2,
                line_dash='dashed',
                alpha=0.8
            )

            regression_lines.append(reg_line)

    # Combine main plot with regression lines
    publication_plot = main_plot
    for line in regression_lines:
        publication_plot *= line

    # Marginal distributions
    temp_hist = hv.Histogram(np.histogram(ecological_data['temperature'], bins=20)).opts(
        xlabel="Temperature (°C)",
        ylabel="Frequency",
        alpha=0.7,
        color='lightblue'
    )

    richness_hist = hv.Histogram(np.histogram(ecological_data['species_richness'], bins=20)).opts(
        xlabel="Species Richness", 
        ylabel="Frequency",
        alpha=0.7,
        color='lightgreen'
    )

    print("Publication-quality visualization:")
    publication_plot
    return


@app.cell
def _():
    """
    ## Design Principles for Ecological Graphics

    ### Key Principles:

    1. **Clarity**: Make the main message immediately apparent
    2. **Accuracy**: Represent data truthfully without distortion
    3. **Efficiency**: Maximize information per unit of plot space
    4. **Aesthetics**: Create visually appealing and professional graphics

    ### Ecological-Specific Guidelines:

    **Color Choices**:
    - Use colorblind-friendly palettes
    - Green/brown for terrestrial, blue for aquatic themes
    - Consistent colors for habitats across all plots
    - Avoid red-green combinations

    **Scale Considerations**:
    - Log-transform highly skewed abundance data
    - Use appropriate axis breaks for ecological ranges
    - Consider sqrt transformation for count data

    **Context**:
    - Always include sample sizes
    - Show variability (error bars, confidence intervals)
    - Indicate statistical significance when relevant
    - Provide clear legends and labels

    **Common Plot Types in Ecology**:
    - **Scatter plots**: Species-environment relationships
    - **Box plots**: Comparing groups (habitats, treatments)
    - **Time series**: Seasonal patterns, long-term trends
    - **Ordination plots**: Community composition
    - **Accumulation curves**: Sampling completeness
    - **Rank-abundance**: Community structure
    """
    return


@app.cell
def _(ecological_data, hv):
    """
    ## Interactive Dashboard Example
    """
    # Create a simple dashboard combining multiple visualizations
    def create_dashboard():
        # Summary statistics table
        summary_stats = ecological_data.groupby('habitat').agg({
            'species_richness': ['count', 'mean', 'std'],
            'shannon_diversity': 'mean',
            'temperature': 'mean'
        }).round(2)

        # Main scatter plot
        scatter = hv.Scatter(
            ecological_data, 
            'temperature', 
            'species_richness'
        ).opts(
            color='habitat',
            size=8,
            alpha=0.7,
            title="Species-Environment Relationships",
            tools=['hover'],
            width=500,
            height=350
        )

        # Distribution plot
        dist_plot = hv.Distribution(
            ecological_data, 
            'species_richness', 
            groupby='habitat'
        ).opts(
            title="Species Richness Distributions",
            alpha=0.6,
            width=400,
            height=250
        ).overlay()

        # Correlation heatmap (simplified)
        simple_corr = ecological_data[['temperature', 'precipitation', 'species_richness', 'shannon_diversity']].corr()

        return scatter, dist_plot, summary_stats, simple_corr

    dashboard_plots = create_dashboard()

    print("Dashboard components created")
    print("Summary statistics by habitat:")
    print(dashboard_plots[2])  # summary_stats

    # Display main visualizations
    dashboard_plots[0] + dashboard_plots[1]  # scatter + dist_plot
    return


@app.cell
def _(ecological_data, hv):
    """
    ## Saving and Exporting Plots
    """
    # Example of how to save plots for publication
    # Note: In Marimo/Pyodide environment, direct file saving may be limited

    # Configure high-quality output
    publication_scatter = hv.Scatter(
        ecological_data, 
        'temperature', 
        'species_richness'
    ).opts(
        color='habitat',
        size=10,
        alpha=0.8,
        xlabel="Temperature (°C)",
        ylabel="Species Richness", 
        title="Species Richness vs Temperature",
        fontsize={'title': 16, 'labels': 14, 'ticks': 12},
        fig_size=300,  # DPI equivalent
        width=800,
        height=600
    )

    # Tips for saving plots:
    print("""
    Saving Plots for Publication:

    1. Use high DPI (300+ for print, 150+ for web)
    2. Choose appropriate file formats:
       - SVG: Vector graphics, perfect for line plots
       - PNG: High-quality raster, good for complex plots
       - PDF: Vector format, ideal for publications

    3. Standard figure sizes:
       - Single column: 3.5 inches width
       - Double column: 7 inches width
       - Full page: 8.5 x 11 inches

    4. Font considerations:
       - Use sans-serif fonts (Arial, Helvetica)
       - Minimum 8pt font size
       - Consistent font sizing across figures

    5. Color considerations:
       - Test in grayscale
       - Use colorblind-friendly palettes
       - High contrast for accessibility
    """)

    publication_scatter
    return


@app.cell
def _():
    """
    ## Summary

    In this chapter, we covered comprehensive data visualization for ecological research:

    ✓ **Basic plots**: Scatter plots, histograms, box plots
    ✓ **Correlation analysis**: Heatmaps, pair plots
    ✓ **Geographic visualization**: Spatial scatter plots
    ✓ **Time series**: Temporal patterns and trends
    ✓ **Interactive plots**: Widgets and linked brushing
    ✓ **Ecological specialties**: Accumulation curves, rank-abundance, ordination
    ✓ **Publication quality**: Professional formatting and design
    ✓ **Dashboard creation**: Combining multiple visualizations

    **Next chapter**: Reproducible science with version control

    **Key visualization packages**:
    - **holoviews**: Declarative visualization
    - **panel**: Interactive dashboards
    - **bokeh**: Interactive web plots
    - **matplotlib**: Static publication plots

    **Best practices**:
    - Choose appropriate plot types for data and message
    - Use consistent colors and themes
    - Include proper labels and legends
    - Consider accessibility and colorblind-friendliness
    - Test plots at intended publication size
    """
    print("✓ Chapter 4 complete! Ready for reproducible science workflows.")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
