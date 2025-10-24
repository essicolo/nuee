import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 6: Biostatistics for Ecological Data

    This chapter covers classical statistical approaches commonly used in ecological 
    research, using Python's statistical libraries to perform tests and analyses 
    equivalent to those in R.

    ## Learning Objectives
    - Master fundamental statistical tests for ecological data
    - Understand assumptions and appropriate use of different tests
    - Perform regression analysis and model selection
    - Handle non-normal data and transformations
    - Interpret statistical results in ecological context
    """
    )
    return


@app.cell
def __():
    # Essential imports for biostatistics
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Biostatistics packages loaded")
    return (
        hv,
        np,
        opts,
        pd,
        sm,
        smf,
        stats,
        variance_inflation_factor,
        warnings,
        het_white,
    )


@app.cell
def __():
    """
    ## Create Ecological Dataset for Statistical Analysis
    """
    # Generate realistic ecological data
    np.random.seed(42)
    n_sites = 200
    
    # Environmental variables
    temperature = np.random.normal(15, 5, n_sites)
    precipitation = np.random.lognormal(6, 0.5, n_sites)
    soil_pH = np.random.normal(6.5, 1.2, n_sites)
    elevation = np.random.uniform(100, 1500, n_sites)
    
    # Habitat types with different probabilities
    habitat_types = np.random.choice(['Forest', 'Grassland', 'Wetland', 'Shrubland'], 
                                   n_sites, p=[0.4, 0.3, 0.2, 0.1])
    
    # Create species richness with realistic ecological relationships
    # Species richness decreases with elevation, increases with precipitation
    species_richness = (
        20 + 
        0.3 * precipitation/100 +  # Positive relationship with precipitation
        -0.005 * elevation +       # Negative relationship with elevation
        2 * (soil_pH - 6.5) +     # Optimum around pH 6.5
        np.random.normal(0, 3, n_sites)  # Random variation
    )
    species_richness = np.maximum(species_richness, 1).astype(int)  # Minimum 1 species
    
    # Shannon diversity correlated with richness but with some independence
    shannon_diversity = np.log(species_richness) + np.random.normal(0, 0.3, n_sites)
    shannon_diversity = np.maximum(shannon_diversity, 0)
    
    # Total abundance with habitat effects
    habitat_effects = {'Forest': 1.2, 'Grassland': 1.0, 'Wetland': 1.1, 'Shrubland': 0.8}
    abundance_base = np.random.lognormal(4, 0.6, n_sites)
    total_abundance = np.array([abundance_base[i] * habitat_effects[habitat_types[i]] 
                               for i in range(n_sites)])
    
    # Create DataFrame
    ecological_data = pd.DataFrame({
        'site_id': [f"SITE_{i:03d}" for i in range(1, n_sites + 1)],
        'temperature': temperature,
        'precipitation': precipitation,
        'soil_pH': soil_pH,
        'elevation': elevation,
        'habitat': habitat_types,
        'species_richness': species_richness,
        'shannon_diversity': shannon_diversity,
        'total_abundance': total_abundance
    })
    
    print(f"Ecological dataset created: {ecological_data.shape}")
    print("\nDataset summary:")
    print(ecological_data.describe())
    return (
        abundance_base,
        ecological_data,
        elevation,
        habitat_effects,
        habitat_types,
        n_sites,
        precipitation,
        shannon_diversity,
        soil_pH,
        species_richness,
        temperature,
        total_abundance,
    )


@app.cell
def __():
    """
    ## Descriptive Statistics and Data Exploration
    """
    # Basic descriptive statistics
    print("Descriptive Statistics by Habitat:")
    habitat_summary = ecological_data.groupby('habitat').agg({
        'species_richness': ['count', 'mean', 'std', 'min', 'max'],
        'shannon_diversity': ['mean', 'std'],
        'temperature': ['mean', 'std'],
        'precipitation': ['mean', 'std']
    }).round(2)
    
    print(habitat_summary)
    
    # Test for normality (Shapiro-Wilk test)
    print("\nNormality Tests (Shapiro-Wilk):")
    continuous_vars = ['species_richness', 'shannon_diversity', 'temperature', 'precipitation']
    
    normality_results = {}
    for var in continuous_vars:
        statistic, p_value = stats.shapiro(ecological_data[var])
        normality_results[var] = {'statistic': statistic, 'p_value': p_value}
        print(f"{var}: W = {statistic:.4f}, p = {p_value:.4f}")
        
    # Check for outliers using IQR method
    print("\nOutlier Detection (IQR method):")
    for var in continuous_vars:
        Q1 = ecological_data[var].quantile(0.25)
        Q3 = ecological_data[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ecological_data[(ecological_data[var] < lower_bound) | 
                                 (ecological_data[var] > upper_bound)]
        print(f"{var}: {len(outliers)} outliers detected")
    
    return continuous_vars, habitat_summary, normality_results


@app.cell
def __():
    """
    ## One-Sample Tests
    """
    # One-sample t-test (equivalent to R's t.test())
    # Test if mean species richness differs from a hypothetical value
    hypothetical_mean = 20
    
    t_statistic, p_value = stats.ttest_1samp(ecological_data['species_richness'], hypothetical_mean)
    
    print(f"One-sample t-test:")
    print(f"H0: Mean species richness = {hypothetical_mean}")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Sample mean: {ecological_data['species_richness'].mean():.2f}")
    
    # Confidence interval for the mean
    confidence_level = 0.95
    mean_richness = ecological_data['species_richness'].mean()
    sem = stats.sem(ecological_data['species_richness'])
    confidence_interval = stats.t.interval(
        confidence_level, 
        len(ecological_data['species_richness']) - 1, 
        loc=mean_richness, 
        scale=sem
    )
    
    print(f"95% Confidence interval: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
    
    # One-sample Wilcoxon test (non-parametric alternative)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
        ecological_data['species_richness'] - hypothetical_mean
    )
    
    print(f"\nOne-sample Wilcoxon test (non-parametric):")
    print(f"Statistic: {wilcoxon_stat:.4f}")
    print(f"p-value: {wilcoxon_p:.4f}")
    
    return (
        confidence_interval,
        confidence_level,
        hypothetical_mean,
        mean_richness,
        p_value,
        sem,
        t_statistic,
        wilcoxon_p,
        wilcoxon_stat,
    )


@app.cell
def __():
    """
    ## Two-Sample Tests
    """
    # Compare species richness between two habitat types
    forest_data = ecological_data[ecological_data['habitat'] == 'Forest']['species_richness']
    grassland_data = ecological_data[ecological_data['habitat'] == 'Grassland']['species_richness']
    
    # Independent t-test (equivalent to R's t.test())
    t_stat, p_val = stats.ttest_ind(forest_data, grassland_data)
    
    print("Independent t-test: Forest vs Grassland species richness")
    print(f"Forest mean: {forest_data.mean():.2f} ± {forest_data.std():.2f}")
    print(f"Grassland mean: {grassland_data.mean():.2f} ± {grassland_data.std():.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4f}")
    
    # Welch's t-test (unequal variances)
    t_stat_welch, p_val_welch = stats.ttest_ind(forest_data, grassland_data, equal_var=False)
    print(f"Welch's t-test p-value: {p_val_welch:.4f}")
    
    # Test for equal variances (Levene's test)
    levene_stat, levene_p = stats.levene(forest_data, grassland_data)
    print(f"Levene's test for equal variances: p = {levene_p:.4f}")
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_p = stats.mannwhitneyu(forest_data, grassland_data, alternative='two-sided')
    print(f"Mann-Whitney U test p-value: {u_p:.4f}")
    
    # Effect size (Cohen's d)
    def cohens_d(group1, group2):
        pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                             (len(group2) - 1) * group2.var()) / 
                            (len(group1) + len(group2) - 2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    effect_size = cohens_d(forest_data, grassland_data)
    print(f"Cohen's d (effect size): {effect_size:.4f}")
    
    return (
        cohens_d,
        effect_size,
        forest_data,
        grassland_data,
        levene_p,
        levene_stat,
        p_val,
        p_val_welch,
        t_stat,
        t_stat_welch,
        u_p,
        u_stat,
    )


@app.cell
def __():
    """
    ## Analysis of Variance (ANOVA)
    """
    # One-way ANOVA (equivalent to R's aov())
    # Test if species richness differs among habitat types
    
    # Prepare data for ANOVA
    habitat_groups = [
        ecological_data[ecological_data['habitat'] == habitat]['species_richness'] 
        for habitat in ecological_data['habitat'].unique()
    ]
    
    # Perform one-way ANOVA
    f_stat, anova_p = stats.f_oneway(*habitat_groups)
    
    print("One-way ANOVA: Species richness by habitat")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {anova_p:.4f}")
    
    # Using statsmodels for more detailed ANOVA
    model = smf.ols('species_richness ~ habitat', data=ecological_data)
    fitted_model = model.fit()
    
    print("\nDetailed ANOVA table:")
    print(sm.stats.anova_lm(fitted_model, typ=2))
    
    # Post-hoc tests (Tukey's HSD equivalent)
    from scipy.stats import tukey_hsd
    
    # Tukey's HSD test
    tukey_result = tukey_hsd(*habitat_groups)
    
    print("\nTukey's HSD post-hoc test:")
    print(f"Statistic: {tukey_result.statistic}")
    print(f"P-values:\n{tukey_result.pvalue}")
    
    # Check ANOVA assumptions
    residuals = fitted_model.resid
    fitted_values = fitted_model.fittedvalues
    
    # Test residual normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\nResidual normality test: p = {shapiro_p:.4f}")
    
    # Test homogeneity of variance (Bartlett's test)
    bartlett_stat, bartlett_p = stats.bartlett(*habitat_groups)
    print(f"Bartlett's test for equal variances: p = {bartlett_p:.4f}")
    
    return (
        anova_p,
        bartlett_p,
        bartlett_stat,
        f_stat,
        fitted_model,
        fitted_values,
        habitat_groups,
        model,
        residuals,
        shapiro_p,
        shapiro_stat,
        tukey_hsd,
        tukey_result,
    )


@app.cell
def __():
    """
    ## Correlation Analysis
    """
    # Pearson correlation (equivalent to R's cor.test())
    correlation_vars = ['temperature', 'precipitation', 'soil_pH', 'elevation', 'species_richness']
    
    print("Pearson Correlation Analysis:")
    print("=" * 50)
    
    # Calculate correlation matrix
    correlation_matrix = ecological_data[correlation_vars].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Test individual correlations
    print("\nIndividual Correlation Tests:")
    for i, var1 in enumerate(correlation_vars):
        for var2 in correlation_vars[i+1:]:
            r, p = stats.pearsonr(ecological_data[var1], ecological_data[var2])
            print(f"{var1} vs {var2}: r = {r:.3f}, p = {p:.4f}")
    
    # Spearman rank correlation (non-parametric)
    print("\nSpearman Rank Correlations:")
    spearman_matrix = ecological_data[correlation_vars].corr(method='spearman')
    print(spearman_matrix.round(3))
    
    # Partial correlation (controlling for a third variable)
    def partial_correlation(df, x, y, control):
        """Calculate partial correlation between x and y controlling for control"""
        # Regress x on control
        x_residuals = sm.OLS(df[x], sm.add_constant(df[control])).fit().resid
        # Regress y on control  
        y_residuals = sm.OLS(df[y], sm.add_constant(df[control])).fit().resid
        # Correlation of residuals
        return stats.pearsonr(x_residuals, y_residuals)
    
    # Example: correlation between species richness and temperature, controlling for elevation
    partial_r, partial_p = partial_correlation(
        ecological_data, 'species_richness', 'temperature', 'elevation'
    )
    
    print(f"\nPartial correlation (richness vs temperature, controlling for elevation):")
    print(f"r = {partial_r:.3f}, p = {partial_p:.4f}")
    
    return (
        correlation_matrix,
        correlation_vars,
        partial_correlation,
        partial_p,
        partial_r,
        spearman_matrix,
    )


@app.cell
def __():
    """
    ## Simple Linear Regression
    """
    # Simple linear regression (equivalent to R's lm())
    # Predict species richness from precipitation
    
    X = ecological_data['precipitation']
    y = ecological_data['species_richness']
    
    # Add intercept term
    X_with_intercept = sm.add_constant(X)
    
    # Fit the model
    regression_model = sm.OLS(y, X_with_intercept).fit()
    
    print("Simple Linear Regression: Species Richness ~ Precipitation")
    print("=" * 60)
    print(regression_model.summary())
    
    # Extract key results
    slope = regression_model.params['precipitation']
    intercept = regression_model.params['const']
    r_squared = regression_model.rsquared
    p_value_slope = regression_model.pvalues['precipitation']
    
    print(f"\nKey Results:")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"P-value for slope: {p_value_slope:.4f}")
    
    # Confidence intervals for parameters
    conf_int = regression_model.conf_int()
    print(f"\n95% Confidence Intervals:")
    print(conf_int)
    
    # Predictions and prediction intervals
    new_precipitation = np.array([500, 750, 1000, 1250, 1500])
    new_X = sm.add_constant(new_precipitation)
    predictions = regression_model.predict(new_X)
    
    # Prediction intervals
    prediction_summary = regression_model.get_prediction(new_X)
    prediction_intervals = prediction_summary.conf_int(alpha=0.05)
    
    print(f"\nPredictions for new precipitation values:")
    for i, precip in enumerate(new_precipitation):
        print(f"Precipitation {precip}: Predicted richness = {predictions.iloc[i]:.2f} "
              f"[{prediction_intervals.iloc[i, 0]:.2f}, {prediction_intervals.iloc[i, 1]:.2f}]")
    
    return (
        X,
        X_with_intercept,
        conf_int,
        intercept,
        new_X,
        new_precipitation,
        p_value_slope,
        prediction_intervals,
        prediction_summary,
        predictions,
        r_squared,
        regression_model,
        slope,
        y,
    )


@app.cell
def __():
    """
    ## Multiple Linear Regression
    """
    # Multiple regression with several predictors
    predictors = ['temperature', 'precipitation', 'soil_pH', 'elevation']
    response = 'species_richness'
    
    # Fit multiple regression model
    formula = f"{response} ~ {' + '.join(predictors)}"
    multiple_model = smf.ols(formula, data=ecological_data).fit()
    
    print("Multiple Linear Regression:")
    print("=" * 50)
    print(multiple_model.summary())
    
    # Model diagnostics
    print("\nModel Diagnostics:")
    print("=" * 30)
    
    # 1. Check for multicollinearity (VIF)
    X_multi = ecological_data[predictors]
    X_multi_with_const = sm.add_constant(X_multi)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_multi.columns
    vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i) 
                       for i in range(X_multi.shape[1])]
    
    print("Variance Inflation Factors (VIF):")
    print(vif_data)
    print("VIF > 5 indicates potential multicollinearity issues")
    
    # 2. Test for heteroscedasticity (White's test)
    white_test = het_white(multiple_model.resid, multiple_model.model.exog)
    white_p_value = white_test[1]
    
    print(f"\nWhite's test for heteroscedasticity: p = {white_p_value:.4f}")
    if white_p_value < 0.05:
        print("Evidence of heteroscedasticity (non-constant variance)")
    else:
        print("No evidence of heteroscedasticity")
    
    # 3. Test residual normality
    residuals_multi = multiple_model.resid
    shapiro_multi, shapiro_p_multi = stats.shapiro(residuals_multi)
    
    print(f"Residual normality test: p = {shapiro_p_multi:.4f}")
    
    # 4. Model comparison with AIC/BIC
    print(f"\nModel Selection Criteria:")
    print(f"AIC: {multiple_model.aic:.2f}")
    print(f"BIC: {multiple_model.bic:.2f}")
    print(f"Adjusted R-squared: {multiple_model.rsquared_adj:.4f}")
    
    return (
        X_multi,
        X_multi_with_const,
        formula,
        multiple_model,
        predictors,
        residuals_multi,
        response,
        shapiro_multi,
        shapiro_p_multi,
        vif_data,
        white_p_value,
        white_test,
    )


@app.cell
def __():
    """
    ## Generalized Linear Models (GLM)
    """
    # Poisson regression for count data (species richness)
    # Species richness is count data, so Poisson distribution may be appropriate
    
    print("Poisson Regression for Species Richness:")
    print("=" * 45)
    
    # Fit Poisson GLM
    poisson_formula = "species_richness ~ temperature + precipitation + soil_pH"
    poisson_model = smf.glm(
        poisson_formula, 
        data=ecological_data, 
        family=sm.families.Poisson()
    ).fit()
    
    print(poisson_model.summary())
    
    # Check for overdispersion
    pearson_chi2 = poisson_model.pearson_chi2
    df_resid = poisson_model.df_resid
    overdispersion_ratio = pearson_chi2 / df_resid
    
    print(f"\nOverdispersion check:")
    print(f"Pearson chi-squared: {pearson_chi2:.2f}")
    print(f"Degrees of freedom: {df_resid}")
    print(f"Overdispersion ratio: {overdispersion_ratio:.2f}")
    
    if overdispersion_ratio > 1.5:
        print("Overdispersion detected - consider negative binomial model")
        
        # Negative binomial regression for overdispersed count data
        nb_model = smf.glm(
            poisson_formula,
            data=ecological_data,
            family=sm.families.NegativeBinomial()
        ).fit()
        
        print("\nNegative Binomial Regression (for overdispersion):")
        print(nb_model.summary())
        
        # Compare models using AIC
        print(f"\nModel Comparison:")
        print(f"Poisson AIC: {poisson_model.aic:.2f}")
        print(f"Negative Binomial AIC: {nb_model.aic:.2f}")
    else:
        print("No significant overdispersion detected")
        nb_model = None
    
    # Logistic regression example (binary outcome)
    # Create binary variable: high diversity (above median)
    median_diversity = ecological_data['shannon_diversity'].median()
    ecological_data['high_diversity'] = (ecological_data['shannon_diversity'] > median_diversity).astype(int)
    
    print(f"\nLogistic Regression for High Diversity (> {median_diversity:.2f}):")
    print("=" * 55)
    
    logistic_formula = "high_diversity ~ temperature + precipitation + soil_pH"
    logistic_model = smf.glm(
        logistic_formula,
        data=ecological_data,
        family=sm.families.Binomial()
    ).fit()
    
    print(logistic_model.summary())
    
    # Calculate odds ratios
    odds_ratios = np.exp(logistic_model.params)
    print(f"\nOdds Ratios:")
    for var, or_val in odds_ratios.items():
        if var != 'Intercept':
            print(f"{var}: {or_val:.3f}")
    
    return (
        df_resid,
        logistic_formula,
        logistic_model,
        median_diversity,
        nb_model,
        odds_ratios,
        overdispersion_ratio,
        pearson_chi2,
        poisson_formula,
        poisson_model,
    )


@app.cell
def __():
    """
    ## Non-parametric Tests
    """
    print("Non-parametric Statistical Tests:")
    print("=" * 40)
    
    # Kruskal-Wallis test (non-parametric ANOVA)
    # Compare species richness among habitats
    habitat_groups_kw = [
        ecological_data[ecological_data['habitat'] == habitat]['species_richness']
        for habitat in ecological_data['habitat'].unique()
    ]
    
    kw_stat, kw_p = stats.kruskal(*habitat_groups_kw)
    
    print("Kruskal-Wallis Test (non-parametric ANOVA):")
    print(f"H-statistic: {kw_stat:.4f}")
    print(f"p-value: {kw_p:.4f}")
    
    # If significant, perform post-hoc pairwise comparisons
    if kw_p < 0.05:
        print("\nPost-hoc pairwise comparisons (Mann-Whitney U):")
        habitats = ecological_data['habitat'].unique()
        
        for i, hab1 in enumerate(habitats):
            for hab2 in habitats[i+1:]:
                group1 = ecological_data[ecological_data['habitat'] == hab1]['species_richness']
                group2 = ecological_data[ecological_data['habitat'] == hab2]['species_richness']
                
                u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                
                # Bonferroni correction
                n_comparisons = len(habitats) * (len(habitats) - 1) / 2
                corrected_p = u_p * n_comparisons
                
                print(f"{hab1} vs {hab2}: p = {u_p:.4f}, corrected p = {corrected_p:.4f}")
    
    # Friedman test (repeated measures non-parametric ANOVA)
    # For demonstration, we'll create paired data
    # Simulate before/after measurements at same sites
    np.random.seed(123)
    n_paired_sites = 30
    before_richness = ecological_data['species_richness'][:n_paired_sites]
    after_richness = before_richness + np.random.normal(2, 3, n_paired_sites)  # Small increase
    control_richness = before_richness + np.random.normal(0, 2, n_paired_sites)  # No change
    
    friedman_stat, friedman_p = stats.friedmanchisquare(
        before_richness, after_richness, control_richness
    )
    
    print(f"\nFriedman Test (repeated measures):")
    print(f"Chi-square statistic: {friedman_stat:.4f}")
    print(f"p-value: {friedman_p:.4f}")
    
    # Kolmogorov-Smirnov test (compare distributions)
    forest_richness = ecological_data[ecological_data['habitat'] == 'Forest']['species_richness']
    grassland_richness = ecological_data[ecological_data['habitat'] == 'Grassland']['species_richness']
    
    ks_stat, ks_p = stats.ks_2samp(forest_richness, grassland_richness)
    
    print(f"\nKolmogorov-Smirnov Test (distribution comparison):")
    print(f"D-statistic: {ks_stat:.4f}")
    print(f"p-value: {ks_p:.4f}")
    
    return (
        after_richness,
        before_richness,
        control_richness,
        corrected_p,
        forest_richness,
        friedman_p,
        friedman_stat,
        grassland_richness,
        group1,
        group2,
        hab1,
        hab2,
        habitat_groups_kw,
        habitats,
        ks_p,
        ks_stat,
        kw_p,
        kw_stat,
        n_comparisons,
        n_paired_sites,
        u_p,
        u_stat,
    )


@app.cell
def __():
    """
    ## Statistical Power and Effect Size
    """
    from scipy.stats import power
    
    print("Statistical Power Analysis:")
    print("=" * 30)
    
    # Calculate effect size for t-test between habitats
    forest_mean = ecological_data[ecological_data['habitat'] == 'Forest']['species_richness'].mean()
    grassland_mean = ecological_data[ecological_data['habitat'] == 'Grassland']['species_richness'].mean()
    pooled_std = np.sqrt((
        ecological_data[ecological_data['habitat'] == 'Forest']['species_richness'].var() +
        ecological_data[ecological_data['habitat'] == 'Grassland']['species_richness'].var()
    ) / 2)
    
    cohens_d = abs(forest_mean - grassland_mean) / pooled_std
    
    print(f"Cohen's d (Forest vs Grassland): {cohens_d:.3f}")
    
    # Interpret effect size
    if cohens_d < 0.2:
        effect_interpretation = "small"
    elif cohens_d < 0.5:
        effect_interpretation = "small to medium"
    elif cohens_d < 0.8:
        effect_interpretation = "medium to large"
    else:
        effect_interpretation = "large"
    
    print(f"Effect size interpretation: {effect_interpretation}")
    
    # Sample size calculation for future studies
    def calculate_sample_size(effect_size, alpha=0.05, power=0.8):
        """
        Calculate required sample size for two-sample t-test
        """
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    # Sample sizes for different effect sizes
    effect_sizes = [0.2, 0.5, 0.8]
    powers = [0.7, 0.8, 0.9]
    
    print(f"\nSample Size Requirements (per group):")
    print("Effect Size | Power = 0.7 | Power = 0.8 | Power = 0.9")
    print("-" * 50)
    
    for es in effect_sizes:
        sample_sizes = [calculate_sample_size(es, power=p) for p in powers]
        print(f"    {es:3.1f}     |     {sample_sizes[0]:3d}     |     {sample_sizes[1]:3d}     |     {sample_sizes[2]:3d}")
    
    return (
        calculate_sample_size,
        cohens_d,
        effect_interpretation,
        effect_sizes,
        forest_mean,
        grassland_mean,
        pooled_std,
        power,
        powers,
        sample_sizes,
    )


@app.cell
def __():
    """
    ## Data Transformations
    """
    print("Data Transformations for Non-normal Data:")
    print("=" * 45)
    
    # Original data distribution
    original_data = ecological_data['total_abundance']
    
    # Test normality of original data
    shapiro_orig, p_orig = stats.shapiro(original_data)
    print(f"Original data normality: W = {shapiro_orig:.4f}, p = {p_orig:.4f}")
    
    # Common transformations
    transformations = {
        'log': np.log(original_data + 1),  # Add 1 to handle zeros
        'sqrt': np.sqrt(original_data),
        'log10': np.log10(original_data + 1),
        'reciprocal': 1 / (original_data + 1),
        'box_cox': None  # Will calculate below
    }
    
    # Box-Cox transformation
    box_cox_transformed, lambda_bc = stats.boxcox(original_data + 1)  # Add 1 for positive values
    transformations['box_cox'] = box_cox_transformed
    
    print(f"\nBox-Cox lambda: {lambda_bc:.4f}")
    
    # Test normality of transformed data
    print(f"\nNormality tests for transformed data:")
    print("Transformation | Shapiro-Wilk W | p-value")
    print("-" * 40)
    
    for transform_name, transformed_data in transformations.items():
        if transformed_data is not None:
            shapiro_stat, p_val = stats.shapiro(transformed_data)
            print(f"{transform_name:12} | {shapiro_stat:13.4f} | {p_val:.4f}")
    
    # Recommend best transformation
    best_transform = None
    best_p = 0
    
    for transform_name, transformed_data in transformations.items():
        if transformed_data is not None:
            _, p_val = stats.shapiro(transformed_data)
            if p_val > best_p:
                best_p = p_val
                best_transform = transform_name
    
    print(f"\nRecommended transformation: {best_transform} (p = {best_p:.4f})")
    
    # Add transformed variable to dataset
    if best_transform == 'log':
        ecological_data['abundance_transformed'] = np.log(ecological_data['total_abundance'] + 1)
    elif best_transform == 'sqrt':
        ecological_data['abundance_transformed'] = np.sqrt(ecological_data['total_abundance'])
    elif best_transform == 'box_cox':
        ecological_data['abundance_transformed'] = box_cox_transformed
    
    return (
        best_p,
        best_transform,
        box_cox_transformed,
        lambda_bc,
        original_data,
        p_orig,
        shapiro_orig,
        transform_name,
        transformed_data,
        transformations,
    )


@app.cell
def __():
    """
    ## Model Selection and Validation
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    print("Model Selection and Cross-Validation:")
    print("=" * 40)
    
    # Prepare different model formulations
    models = {
        'Model 1': 'species_richness ~ temperature',
        'Model 2': 'species_richness ~ temperature + precipitation', 
        'Model 3': 'species_richness ~ temperature + precipitation + soil_pH',
        'Model 4': 'species_richness ~ temperature + precipitation + soil_pH + elevation',
        'Model 5': 'species_richness ~ temperature * precipitation + soil_pH + elevation'  # Interaction
    }
    
    # Fit models and compare
    model_results = {}
    
    for model_name, formula in models.items():
        model = smf.ols(formula, data=ecological_data).fit()
        
        model_results[model_name] = {
            'aic': model.aic,
            'bic': model.bic,
            'rsquared': model.rsquared,
            'rsquared_adj': model.rsquared_adj,
            'n_params': len(model.params)
        }
    
    # Display model comparison
    print("Model Comparison:")
    print("Model    | AIC     | BIC     | R²      | Adj R²  | Params")
    print("-" * 55)
    
    for model_name, results in model_results.items():
        print(f"{model_name:8} | {results['aic']:7.1f} | {results['bic']:7.1f} | "
              f"{results['rsquared']:7.3f} | {results['rsquared_adj']:7.3f} | {results['n_params']:6d}")
    
    # Find best model by AIC
    best_model_aic = min(model_results.keys(), key=lambda x: model_results[x]['aic'])
    print(f"\nBest model by AIC: {best_model_aic}")
    
    # Cross-validation for selected model
    best_formula = models[best_model_aic]
    
    # Extract variables from formula for sklearn
    import re
    variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', best_formula)
    variables = [v for v in variables if v != 'species_richness']
    
    # Prepare data for cross-validation
    X_cv = ecological_data[variables].values
    y_cv = ecological_data['species_richness'].values
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(LinearRegression(), X_cv, y_cv, cv=5, scoring='r2')
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Individual fold scores: {cv_scores}")
    
    return (
        LinearRegression,
        X_cv,
        best_formula,
        best_model_aic,
        cross_val_score,
        cv_scores,
        mean_squared_error,
        model_results,
        models,
        r2_score,
        re,
        variables,
        y_cv,
    )


@app.cell
def __():
    """
    ## Summary and Best Practices

    In this chapter, we covered essential biostatistical methods for ecological data:

    ✓ **Descriptive statistics** and data exploration
    ✓ **Normality testing** and assumption checking
    ✓ **One-sample and two-sample tests** (t-tests, Wilcoxon)
    ✓ **ANOVA** and post-hoc comparisons
    ✓ **Correlation analysis** (Pearson, Spearman, partial)
    ✓ **Linear regression** (simple and multiple)
    ✓ **Generalized Linear Models** (Poisson, logistic regression)
    ✓ **Non-parametric tests** (Kruskal-Wallis, Mann-Whitney)
    ✓ **Effect sizes and power analysis**
    ✓ **Data transformations** for non-normal data
    ✓ **Model selection** and cross-validation

    ### Key Statistical Packages in Python:
    - **scipy.stats**: Basic statistical tests and distributions
    - **statsmodels**: Advanced regression and GLM modeling
    - **sklearn**: Machine learning and cross-validation
    - **pandas**: Data manipulation and groupwise operations

    ### Best Practices for Ecological Statistics:
    1. **Always check assumptions** before applying parametric tests
    2. **Use appropriate transformations** for non-normal data
    3. **Report effect sizes** along with p-values
    4. **Consider multiple comparisons** corrections
    5. **Validate models** using cross-validation
    6. **Use biological knowledge** to inform statistical choices
    7. **Be transparent** about model selection procedures

    ### Common Pitfalls to Avoid:
    - Using parametric tests on non-normal data without checking
    - Multiple testing without correction
    - Overfitting with too many variables
    - Ignoring biological meaningful effect sizes
    - Pseudoreplication in experimental design
    """
    
    print("Statistical Analysis Guidelines for Ecologists:")
    print("=" * 50)
    
    guidelines = {
        "Test Selection": [
            "Check normality before choosing parametric vs non-parametric tests",
            "Use appropriate tests for count data (Poisson/negative binomial)",
            "Consider paired tests for before/after or matched data"
        ],
        "Effect Sizes": [
            "Always report effect sizes (Cohen's d, R², eta-squared)",
            "Focus on biological significance, not just statistical significance",
            "Use confidence intervals to show uncertainty"
        ],
        "Model Building": [
            "Start with biologically motivated variables",
            "Check for multicollinearity (VIF < 5)",
            "Validate models with independent data when possible"
        ],
        "Assumptions": [
            "Test residual normality and homoscedasticity", 
            "Check for outliers and influential points",
            "Transform data when assumptions are violated"
        ]
    }
    
    for category, tips in guidelines.items():
        print(f"\n{category}:")
        for tip in tips:
            print(f"  • {tip}")
    
    print("\n✓ Chapter 6 complete! Ready for Bayesian biostatistics.")
    
    return guidelines,


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()