import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 8: Association, Partitioning, and Ordination

    This chapter covers multivariate analysis techniques for ecological data,
    focusing on ordination methods using the nuee package (Python port of R's vegan).

    ## Learning Objectives
    - Understand different ordination methods and their applications
    - Perform PCA, CA, RDA, and CCA analyses
    - Interpret ordination results in ecological context
    - Visualize community patterns and species-environment relationships
    """
    )
    return


@app.cell
def __():
    # Essential imports for ordination analysis
    import pandas as pd
    import numpy as np
    import holoviews as hv
    from holoviews import opts
    import scipy.stats as stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import nuee
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Packages loaded for ordination analysis")
    return PCA, StandardScaler, hv, np, nuee, opts, pd, stats, warnings


@app.cell
def __():
    """
    ## Introduction to Ordination

    **Ordination** arranges samples or species along axes that represent the main 
    patterns of variation in ecological data.

    **Common methods**:
    - **PCA**: Principal Component Analysis (linear, unconstrained)
    - **CA**: Correspondence Analysis (unimodal, unconstrained)  
    - **RDA**: Redundancy Analysis (linear, constrained)
    - **CCA**: Canonical Correspondence Analysis (unimodal, constrained)
    - **NMDS**: Non-metric Multidimensional Scaling (distance-based)

    **When to use each method**:
    - **PCA**: Short environmental gradients, linear species responses
    - **CA**: Long gradients, unimodal species responses
    - **RDA**: Relate community composition to environmental variables (linear)
    - **CCA**: Relate community composition to environmental variables (unimodal)
    - **NMDS**: Complex, non-linear relationships
    """
    return


@app.cell
def __():
    """
    ## Load Example Data
    """
    # Load classic ecological datasets from nuee
    varespec = nuee.datasets.varespec()  # Lichen species data
    varechem = nuee.datasets.varechem()  # Environmental variables
    
    print("Varespec dataset (species composition):")
    print(f"Shape: {varespec.shape}")
    print(varespec.iloc[:5, :8])  # First 5 sites, 8 species
    
    print("\nVarechem dataset (environmental variables):")
    print(f"Shape: {varechem.shape}")
    print(varechem.head())
    return varechem, varespec


@app.cell
def __():
    """
    ## Data Exploration and Preparation
    """
    # Basic statistics
    print("Species data summary:")
    print(f"Total species: {varespec.shape[1]}")
    print(f"Total sites: {varespec.shape[0]}")
    print(f"Total abundance: {varespec.sum().sum()}")
    print(f"Species per site (mean ± SD): {varespec.sum(axis=1).mean():.1f} ± {varespec.sum(axis=1).std():.1f}")
    print(f"Sites per species (mean ± SD): {(varespec > 0).sum(axis=0).mean():.1f} ± {(varespec > 0).sum(axis=0).std():.1f}")
    
    # Check for empty sites or rare species
    site_totals = varespec.sum(axis=1)
    species_totals = varespec.sum(axis=0)
    
    print(f"\nEmpty sites: {(site_totals == 0).sum()}")
    print(f"Absent species: {(species_totals == 0).sum()}")
    print(f"Rare species (< 3 occurrences): {(species_totals < 3).sum()}")
    return site_totals, species_totals


@app.cell
def __():
    """
    ## Principal Component Analysis (PCA)

    PCA is appropriate for short environmental gradients where species responses 
    are approximately linear.
    """
    # Perform PCA using nuee
    pca_result = nuee.pca(varespec)
    
    print("PCA Results:")
    print(f"Eigenvalues: {pca_result.eigenvalues[:4]}")
    print(f"Proportion explained: {pca_result.proportion_explained[:4]}")
    print(f"Cumulative proportion: {pca_result.cumulative_proportion[:4]}")
    
    # How much variance is explained by first two axes?
    var_explained = pca_result.cumulative_proportion[1] * 100
    print(f"\nFirst two axes explain {var_explained:.1f}% of variance")
    return pca_result, var_explained


@app.cell
def __():
    """
    ## PCA Visualization
    """
    # Create PCA biplot
    pca_plot = pca_result.biplot(
        title="PCA of Lichen Species Composition",
        figsize=(10, 8)
    )
    
    # Display the plot
    pca_plot
    return pca_plot,


@app.cell
def __():
    """
    ## Correspondence Analysis (CA)

    CA is appropriate for longer environmental gradients where species show 
    unimodal responses.
    """
    # Perform CA using nuee
    ca_result = nuee.cca(varespec)  # CA is CCA without constraints
    
    print("CA Results:")
    print(f"Eigenvalues: {ca_result.eigenvalues[:4]}")
    print(f"Proportion explained: {ca_result.proportion_explained[:4]}")
    print(f"Cumulative proportion: {ca_result.cumulative_proportion[:4]}")
    
    # CA typically explains less variance per axis than PCA
    ca_var_explained = ca_result.cumulative_proportion[1] * 100
    print(f"\nFirst two axes explain {ca_var_explained:.1f}% of variance")
    return ca_result, ca_var_explained


@app.cell
def __():
    """
    ## CA Visualization
    """
    # Create CA biplot
    ca_plot = ca_result.biplot(
        title="CA of Lichen Species Composition",
        figsize=(10, 8)
    )
    
    ca_plot
    return ca_plot,


@app.cell
def __():
    """
    ## Redundancy Analysis (RDA)

    RDA relates species composition to environmental variables.
    It's a constrained ordination method (canonical analysis).
    """
    # Perform RDA with environmental constraints
    rda_result = nuee.rda(varespec, varechem)
    
    print("RDA Results:")
    print(f"Constrained eigenvalues: {rda_result.constrained_eigenvalues[:4]}")
    print(f"Unconstrained eigenvalues: {rda_result.unconstrained_eigenvalues[:4]}")
    print(f"Proportion explained (constrained): {rda_result.constrained_proportion[:4]}")
    print(f"Proportion explained (total): {rda_result.total_proportion[:4]}")
    
    # How much variation is explained by environmental variables?
    constrained_var = rda_result.constrained_proportion.sum() * 100
    print(f"\nEnvironmental variables explain {constrained_var:.1f}% of species variation")
    return constrained_var, rda_result


@app.cell
def __():
    """
    ## RDA Visualization
    """
    # Create RDA triplot (sites, species, and environmental variables)
    rda_plot = rda_result.biplot(
        title="RDA: Species-Environment Relationships",
        figsize=(12, 10)
    )
    
    rda_plot
    return rda_plot,


@app.cell
def __():
    """
    ## Environmental Fitting (envfit)

    Test significance of environmental variables in ordination space.
    """
    # Fit environmental variables to ordination
    envfit_result = nuee.envfit(pca_result, varechem)
    
    print("Environmental fitting results:")
    print("Variable correlations with ordination axes:")
    print(envfit_result.correlations)
    
    print(f"\nSignificant variables (p < 0.05):")
    significant_vars = envfit_result.p_values < 0.05
    for var, p_val in zip(varechem.columns[significant_vars], 
                         envfit_result.p_values[significant_vars]):
        print(f"  {var}: p = {p_val:.3f}")
    return envfit_result, significant_vars, var


@app.cell
def __():
    """
    ## Non-metric Multidimensional Scaling (NMDS)

    NMDS is a robust ordination method that can handle non-linear relationships
    and different distance measures.
    """
    # Perform NMDS
    nmds_result = nuee.metaMDS(varespec, distance='bray', k=2)
    
    print("NMDS Results:")
    print(f"Stress: {nmds_result.stress:.3f}")
    print(f"Converged: {nmds_result.converged}")
    print(f"Number of runs: {nmds_result.nruns}")
    
    # Stress interpretation
    if nmds_result.stress < 0.05:
        stress_interpretation = "excellent"
    elif nmds_result.stress < 0.1:
        stress_interpretation = "good" 
    elif nmds_result.stress < 0.2:
        stress_interpretation = "fair"
    else:
        stress_interpretation = "poor"
    
    print(f"Stress interpretation: {stress_interpretation}")
    return nmds_result, stress_interpretation


@app.cell
def __():
    """
    ## NMDS Visualization
    """
    # Create NMDS plot
    nmds_plot = nmds_result.plot(
        title=f"NMDS of Lichen Communities (stress = {nmds_result.stress:.3f})",
        figsize=(10, 8)
    )
    
    nmds_plot
    return nmds_plot,


@app.cell
def __():
    """
    ## Canonical Correspondence Analysis (CCA)

    CCA is the unimodal equivalent of RDA, appropriate for long environmental gradients.
    """
    # Perform CCA with environmental constraints
    cca_result = nuee.cca(varespec, varechem)
    
    print("CCA Results:")
    print(f"Constrained eigenvalues: {cca_result.constrained_eigenvalues[:4]}")
    print(f"Unconstrained eigenvalues: {cca_result.unconstrained_eigenvalues[:4]}")
    print(f"Proportion explained (constrained): {cca_result.constrained_proportion[:4]}")
    
    # Total inertia in CCA is the sum of all eigenvalues
    total_inertia = cca_result.total_inertia
    constrained_inertia = cca_result.constrained_inertia
    
    print(f"\nTotal inertia: {total_inertia:.3f}")
    print(f"Constrained inertia: {constrained_inertia:.3f}")
    print(f"Proportion constrained: {constrained_inertia/total_inertia:.3f}")
    return cca_result, constrained_inertia, total_inertia


@app.cell
def __():
    """
    ## CCA Visualization
    """
    # Create CCA triplot
    cca_plot = cca_result.biplot(
        title="CCA: Canonical Correspondence Analysis",
        figsize=(12, 10)
    )
    
    cca_plot
    return cca_plot,


@app.cell
def __():
    """
    ## Permutation Tests for Constrained Ordination

    Test the significance of constrained ordination results.
    """
    # Test overall significance of RDA
    rda_perm = nuee.permutest(rda_result, permutations=999)
    
    print("RDA Permutation Test:")
    print(f"F-statistic: {rda_perm.statistic:.3f}")
    print(f"P-value: {rda_perm.p_value:.3f}")
    print(f"Permutations: {rda_perm.permutations}")
    
    # Test significance of each axis
    print(f"\nAxis significance:")
    for i, (f_stat, p_val) in enumerate(zip(rda_perm.axis_statistics, rda_perm.axis_p_values)):
        print(f"  Axis {i+1}: F = {f_stat:.3f}, p = {p_val:.3f}")
    return i, rda_perm


@app.cell
def __():
    """
    ## Variable Selection in Constrained Ordination

    Use forward selection to identify the most important environmental variables.
    """
    # Forward selection for RDA
    rda_selection = nuee.ordistep(varespec, varechem, method='forward')
    
    print("Forward selection results:")
    print(f"Selected variables: {rda_selection.selected_variables}")
    print(f"AIC values: {rda_selection.aic_values}")
    print(f"Final model AIC: {rda_selection.final_aic:.2f}")
    
    # Perform RDA with selected variables only
    selected_env = varechem[rda_selection.selected_variables]
    rda_reduced = nuee.rda(varespec, selected_env)
    
    print(f"\nReduced model:")
    print(f"Variables: {len(rda_selection.selected_variables)}")
    print(f"Variance explained: {rda_reduced.constrained_proportion.sum():.3f}")
    return rda_reduced, rda_selection, selected_env


@app.cell
def __():
    """
    ## Comparing Ordination Methods

    Compare different ordination approaches for the same dataset.
    """
    # Create comparison table
    comparison_data = {
        'Method': ['PCA', 'CA', 'RDA', 'CCA', 'NMDS'],
        'Type': ['Unconstrained', 'Unconstrained', 'Constrained', 'Constrained', 'Distance-based'],
        'Response_model': ['Linear', 'Unimodal', 'Linear', 'Unimodal', 'Non-parametric'],
        'Axis1_variance': [
            pca_result.proportion_explained[0],
            ca_result.proportion_explained[0], 
            rda_result.constrained_proportion[0],
            cca_result.constrained_proportion[0],
            np.nan  # NMDS doesn't have explained variance
        ],
        'Axis2_variance': [
            pca_result.proportion_explained[1],
            ca_result.proportion_explained[1],
            rda_result.constrained_proportion[1], 
            cca_result.constrained_proportion[1],
            np.nan
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Ordination Method Comparison:")
    print(comparison_df.round(3))
    return comparison_data, comparison_df


@app.cell
def __():
    """
    ## Interpreting Ordination Results

    **Key principles for interpretation**:

    1. **Eigenvalues**: Measure the amount of variation explained by each axis
    2. **Species scores**: Show species positions in ordination space
    3. **Site scores**: Show sample positions in ordination space  
    4. **Environmental arrows**: Show direction and strength of environmental gradients

    **Ecological interpretation**:
    - **Proximity**: Similar sites/species are close in ordination space
    - **Gradients**: Environmental arrows show the direction of increasing variable values
    - **Length**: Longer arrows indicate stronger environmental effects
    - **Angles**: Small angles between arrows indicate correlated variables

    **Statistical considerations**:
    - Check for outliers that might distort ordination
    - Consider data transformations (log, square root)
    - Validate results with independent data when possible
    - Use permutation tests to assess significance
    """
    return


@app.cell
def __():
    """
    ## Practical Guidelines

    **Choosing an ordination method**:

    1. **Gradient length**: Use DCA or check β-diversity
       - Short gradients (< 3 SD): Linear methods (PCA, RDA)
       - Long gradients (> 4 SD): Unimodal methods (CA, CCA)

    2. **Research question**:
       - Explore patterns: Unconstrained ordination (PCA, CA, NMDS)
       - Test hypotheses: Constrained ordination (RDA, CCA)

    3. **Data characteristics**:
       - Species abundances: All methods applicable
       - Presence/absence: CA, CCA may be preferred
       - Highly skewed: Consider transformations or NMDS

    4. **Sample size**:
       - Small samples: NMDS may be unstable
       - Large samples: All methods applicable

    **Common pitfalls**:
    - Don't over-interpret axes with low eigenvalues
    - Be cautious with rare species (consider removing)
    - Check for outlying sites that dominate ordination
    - Validate environmental relationships independently
    """
    return


@app.cell
def __():
    """
    ## Summary and Next Steps

    In this chapter, we covered:

    ✓ **Ordination concepts**: Different methods and their applications
    ✓ **PCA**: Linear unconstrained ordination
    ✓ **CA**: Unimodal unconstrained ordination  
    ✓ **RDA**: Linear constrained ordination
    ✓ **CCA**: Unimodal constrained ordination
    ✓ **NMDS**: Distance-based ordination
    ✓ **Environmental fitting**: Testing variable significance
    ✓ **Variable selection**: Identifying important predictors
    ✓ **Interpretation**: Understanding ecological patterns

    **Next chapter**: Data quality assessment - outlier detection and imputation

    **Key takeaways**:
    - Choose ordination method based on gradient length and research question
    - Constrained ordination tests specific hypotheses about species-environment relationships
    - Permutation tests provide statistical validation
    - Environmental fitting helps identify important variables
    - Visualization is crucial for interpretation
    """
    print("✓ Chapter 8 complete! Ready for data quality assessment.")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()