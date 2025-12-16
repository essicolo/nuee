import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 9: Data Quality Assessment and Outlier Detection

    This chapter covers methods for assessing and improving data quality in 
    ecological datasets, including outlier detection, missing data imputation,
    and data validation techniques.

    ## Learning Objectives
    - Identify and handle outliers in ecological data
    - Implement missing data imputation strategies
    - Validate data consistency and logical constraints
    - Apply robust statistical methods for noisy data
    - Develop data quality control workflows
    """
    )
    return


@app.cell
def __():
    # Essential imports for data quality assessment
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Data quality assessment packages loaded")
    return (
        IsolationForest,
        KNNImputer,
        LocalOutlierFactor,
        SimpleImputer,
        StandardScaler,
        hv,
        np,
        opts,
        pd,
        stats,
        warnings,
    )


@app.cell
def __():
    """
    ## Create Ecological Dataset with Quality Issues
    """
    # Generate realistic ecological data with various quality issues
    np.random.seed(42)
    n_sites = 300
    
    # Base environmental data
    temperature = np.random.normal(15, 5, n_sites)
    precipitation = np.random.lognormal(6, 0.5, n_sites)
    elevation = np.random.uniform(0, 2000, n_sites)
    soil_pH = np.random.normal(6.5, 1.2, n_sites)
    
    # Species richness with ecological relationships
    species_richness = (
        20 + 
        0.3 * precipitation/100 +
        -0.005 * elevation +
        2 * (soil_pH - 6.5) +
        np.random.normal(0, 3, n_sites)
    )
    species_richness = np.maximum(species_richness, 1).astype(int)
    
    # Total abundance
    total_abundance = np.random.lognormal(np.log(species_richness * 10), 0.5)
    
    # Shannon diversity
    shannon_diversity = np.log(species_richness) + np.random.normal(0, 0.2, n_sites)
    
    # Create DataFrame
    quality_data = pd.DataFrame({
        'site_id': [f"SITE_{i:03d}" for i in range(1, n_sites + 1)],
        'temperature': temperature,
        'precipitation': precipitation,
        'elevation': elevation,
        'soil_pH': soil_pH,
        'species_richness': species_richness,
        'total_abundance': total_abundance,
        'shannon_diversity': shannon_diversity
    })
    
    # Introduce various data quality issues
    
    # 1. Outliers (measurement errors)
    outlier_indices = np.random.choice(n_sites, size=10, replace=False)
    quality_data.loc[outlier_indices[:3], 'temperature'] += np.random.choice([-20, 30], 3)
    quality_data.loc[outlier_indices[3:6], 'species_richness'] *= 3  # Extremely high richness
    quality_data.loc[outlier_indices[6:], 'soil_pH'] = np.random.uniform(2, 12, 4)  # Extreme pH
    
    # 2. Missing values (various patterns)
    missing_random = np.random.choice(n_sites, size=15, replace=False)
    quality_data.loc[missing_random, 'soil_pH'] = np.nan
    
    # Missing not at random (high elevation sites missing temperature)
    high_elevation = quality_data['elevation'] > 1500
    missing_temp = np.random.choice(quality_data[high_elevation].index, size=8, replace=False)
    quality_data.loc[missing_temp, 'temperature'] = np.nan
    
    # 3. Impossible values
    impossible_indices = np.random.choice(n_sites, size=5, replace=False)
    quality_data.loc[impossible_indices, 'species_richness'] = 0  # No species but positive abundance
    quality_data.loc[impossible_indices, 'total_abundance'] = np.random.uniform(10, 50, 5)
    
    # 4. Data entry errors
    entry_errors = np.random.choice(n_sites, size=3, replace=False)
    quality_data.loc[entry_errors, 'precipitation'] *= 100  # Unit error (mm instead of cm)
    
    print(f"Dataset with quality issues created: {quality_data.shape}")
    print(f"Missing values: {quality_data.isnull().sum().sum()}")
    
    return (
        elevation,
        entry_errors,
        high_elevation,
        impossible_indices,
        missing_random,
        missing_temp,
        n_sites,
        outlier_indices,
        precipitation,
        quality_data,
        shannon_diversity,
        soil_pH,
        species_richness,
        temperature,
        total_abundance,
    )


@app.cell
def __():
    """
    ## Missing Data Analysis
    """
    # Analyze missing data patterns
    def analyze_missing_data(df):
        """Comprehensive missing data analysis"""
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes
        })
        
        # Missing data patterns
        missing_patterns = df.isnull().groupby(df.isnull().sum(axis=1)).size()
        
        # Co-occurrence of missing values
        missing_corr = df.isnull().corr()
        
        return missing_summary, missing_patterns, missing_corr
    
    missing_summary, missing_patterns, missing_corr = analyze_missing_data(quality_data)
    
    print("Missing Data Analysis:")
    print("=" * 30)
    print(missing_summary[missing_summary['Missing_Count'] > 0])
    
    print(f"\nMissing Data Patterns:")
    print("Variables Missing | Number of Cases")
    print("-" * 35)
    for n_missing, count in missing_patterns.items():
        print(f"       {n_missing:2d}         |      {count:3d}")
    
    print(f"\nMissing Data Correlations:")
    print(missing_corr.round(3))
    
    # Test for Missing Completely at Random (MCAR)
    def little_mcar_test_simplified(df):
        """Simplified test for MCAR assumption"""
        # Check if missingness is related to observed values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        mcar_results = {}
        for col in numeric_cols:
            if df[col].isnull().any():
                # Compare means of other variables for missing vs non-missing
                missing_mask = df[col].isnull()
                
                for other_col in numeric_cols:
                    if other_col != col and not df[other_col].isnull().all():
                        present_values = df.loc[~missing_mask, other_col].dropna()
                        missing_group_values = df.loc[missing_mask, other_col].dropna()
                        
                        if len(present_values) > 5 and len(missing_group_values) > 5:
                            t_stat, p_value = stats.ttest_ind(present_values, missing_group_values)
                            mcar_results[f"{col}_vs_{other_col}"] = {
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
        
        return mcar_results
    
    mcar_results = little_mcar_test_simplified(quality_data)
    
    print(f"\nMCAR Test Results (simplified):") 
    significant_tests = [k for k, v in mcar_results.items() if v['significant']]
    
    if significant_tests:
        print(f"Significant dependencies found: {len(significant_tests)}")
        print("Data may not be Missing Completely at Random")
    else:
        print("No strong evidence against MCAR assumption")
    
    return (
        analyze_missing_data,
        little_mcar_test_simplified,
        mcar_results,
        missing_corr,
        missing_patterns,
        missing_summary,
        significant_tests,
    )


@app.cell
def __():
    """
    ## Outlier Detection Methods
    """
    # 1. Univariate outlier detection
    def detect_univariate_outliers(series, method='iqr', threshold=1.5):
        """Detect outliers using various univariate methods"""
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = pd.Series(False, index=series.index)
            outliers.loc[series.dropna().index] = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
            
        return outliers
    
    # Apply univariate outlier detection
    outlier_results = {}
    
    for column in ['temperature', 'precipitation', 'species_richness', 'soil_pH']:
        outlier_results[column] = {
            'iqr': detect_univariate_outliers(quality_data[column], 'iqr'),
            'zscore': detect_univariate_outliers(quality_data[column], 'zscore', 3),
            'modified_zscore': detect_univariate_outliers(quality_data[column], 'modified_zscore', 3.5)
        }
    
    # Display outlier detection results
    print("Univariate Outlier Detection Results:")
    print("=" * 45)
    print("Variable      | IQR | Z-Score | Modified Z-Score")
    print("-" * 45)
    
    for column, methods in outlier_results.items():
        iqr_count = methods['iqr'].sum()
        zscore_count = methods['zscore'].sum()
        mod_zscore_count = methods['modified_zscore'].sum()
        print(f"{column:12} | {iqr_count:3d} |    {zscore_count:3d}  |       {mod_zscore_count:3d}")
    
    # 2. Multivariate outlier detection
    def detect_multivariate_outliers(df, contamination=0.1):
        """Detect multivariate outliers using multiple methods"""
        # Prepare data (numeric columns only, no missing values)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data_clean = df[numeric_cols].dropna()
        
        if len(data_clean) < 10:
            return pd.Series(False, index=df.index)
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        
        methods = {}
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_outliers = iso_forest.fit_predict(data_scaled) == -1
        methods['isolation_forest'] = pd.Series(False, index=df.index)
        methods['isolation_forest'].loc[data_clean.index] = iso_outliers
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=contamination)
        lof_outliers = lof.fit_predict(data_scaled) == -1
        methods['lof'] = pd.Series(False, index=df.index)
        methods['lof'].loc[data_clean.index] = lof_outliers
        
        # Mahalanobis distance
        try:
            cov_matrix = np.cov(data_scaled.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mean_vector = np.mean(data_scaled, axis=0)
            
            mahal_distances = []
            for i in range(len(data_scaled)):
                diff = data_scaled[i] - mean_vector
                mahal_dist = np.sqrt(diff @ inv_cov_matrix @ diff.T)
                mahal_distances.append(mahal_dist)
            
            mahal_threshold = np.percentile(mahal_distances, (1-contamination)*100)
            mahal_outliers = np.array(mahal_distances) > mahal_threshold
            methods['mahalanobis'] = pd.Series(False, index=df.index)
            methods['mahalanobis'].loc[data_clean.index] = mahal_outliers
            
        except np.linalg.LinAlgError:
            methods['mahalanobis'] = pd.Series(False, index=df.index)
        
        return methods
    
    multivariate_outliers = detect_multivariate_outliers(quality_data)
    
    print(f"\nMultivariate Outlier Detection:")
    for method, outliers in multivariate_outliers.items():
        print(f"{method.replace('_', ' ').title()}: {outliers.sum()} outliers")
    
    # Consensus outliers (detected by multiple methods)
    outlier_counts = pd.DataFrame(multivariate_outliers).sum(axis=1)
    consensus_outliers = outlier_counts >= 2
    
    print(f"\nConsensus outliers (2+ methods): {consensus_outliers.sum()}")
    
    return (
        consensus_outliers,
        detect_multivariate_outliers,
        detect_univariate_outliers,
        iqr_count,
        methods,
        mod_zscore_count,
        multivariate_outliers,
        outlier_counts,
        outlier_results,
        zscore_count,
    )


@app.cell
def __():
    """
    ## Data Validation and Logical Constraints
    """
    def validate_ecological_data(df):
        """Validate ecological data against known constraints"""
        validation_issues = []
        
        # 1. Range checks
        range_constraints = {
            'temperature': (-50, 60),  # Celsius
            'precipitation': (0, 5000),  # mm/year
            'elevation': (0, 9000),  # meters
            'soil_pH': (0, 14),
            'species_richness': (0, 1000),
            'total_abundance': (0, 100000),
            'shannon_diversity': (0, 10)
        }
        
        for column, (min_val, max_val) in range_constraints.items():
            if column in df.columns:
                out_of_range = (df[column] < min_val) | (df[column] > max_val)
                if out_of_range.any():
                    validation_issues.append({
                        'type': 'range_violation',
                        'column': column,
                        'count': out_of_range.sum(),
                        'indices': df.index[out_of_range].tolist()
                    })
        
        # 2. Logical consistency checks
        
        # Species richness should be <= total abundance (can't have more species than individuals)
        if 'species_richness' in df.columns and 'total_abundance' in df.columns:
            richness_abundance_issue = df['species_richness'] > df['total_abundance']
            if richness_abundance_issue.any():
                validation_issues.append({
                    'type': 'logical_inconsistency',
                    'description': 'Species richness > total abundance',
                    'count': richness_abundance_issue.sum(),
                    'indices': df.index[richness_abundance_issue].tolist()
                })
        
        # Shannon diversity should be <= log(species richness)
        if 'shannon_diversity' in df.columns and 'species_richness' in df.columns:
            max_shannon = np.log(df['species_richness'].replace(0, 1))  # Avoid log(0)
            shannon_too_high = df['shannon_diversity'] > max_shannon * 1.1  # Small tolerance
            if shannon_too_high.any():
                validation_issues.append({
                    'type': 'logical_inconsistency',
                    'description': 'Shannon diversity too high for species richness',
                    'count': shannon_too_high.sum(),
                    'indices': df.index[shannon_too_high].tolist()
                })
        
        # Zero species but positive abundance
        if 'species_richness' in df.columns and 'total_abundance' in df.columns:
            zero_species_pos_abundance = (df['species_richness'] == 0) & (df['total_abundance'] > 0)
            if zero_species_pos_abundance.any():
                validation_issues.append({
                    'type': 'logical_inconsistency',
                    'description': 'Zero species but positive abundance',
                    'count': zero_species_pos_abundance.sum(),
                    'indices': df.index[zero_species_pos_abundance].tolist()
                })
        
        # 3. Statistical outliers in relationships
        
        # Check for unusual temperature-elevation relationships
        if 'temperature' in df.columns and 'elevation' in df.columns:
            # Expected: temperature decreases with elevation (~6.5°C per 1000m)
            expected_temp = 20 - (df['elevation'] / 1000) * 6.5
            temp_residuals = df['temperature'] - expected_temp
            extreme_temp_elev = np.abs(temp_residuals) > 15  # > 15°C deviation
            
            if extreme_temp_elev.any():
                validation_issues.append({
                    'type': 'relationship_outlier',
                    'description': 'Unusual temperature-elevation relationship',
                    'count': extreme_temp_elev.sum(),
                    'indices': df.index[extreme_temp_elev].tolist()
                })
        
        return validation_issues
    
    # Validate the dataset
    validation_results = validate_ecological_data(quality_data)
    
    print("Data Validation Results:")
    print("=" * 30)
    
    if not validation_results:
        print("No validation issues detected.")
    else:
        for issue in validation_results:
            print(f"\n{issue['type'].replace('_', ' ').title()}:")
            if 'column' in issue:
                print(f"  Column: {issue['column']}")
            if 'description' in issue:
                print(f"  Description: {issue['description']}")
            print(f"  Count: {issue['count']}")
            print(f"  Sample indices: {issue['indices'][:5]}...")  # Show first 5
    
    # Create a data quality score
    def calculate_quality_score(df, validation_issues):
        """Calculate overall data quality score (0-100)"""
        total_cells = df.shape[0] * df.shape[1]
        
        # Penalties
        missing_penalty = df.isnull().sum().sum() / total_cells * 100
        
        validation_penalty = 0
        for issue in validation_issues:
            validation_penalty += issue['count']
        validation_penalty = (validation_penalty / df.shape[0]) * 100
        
        # Quality score (higher is better)
        quality_score = 100 - missing_penalty - validation_penalty
        quality_score = max(0, quality_score)  # Ensure non-negative
        
        return {
            'overall_score': quality_score,
            'missing_penalty': missing_penalty,
            'validation_penalty': validation_penalty
        }
    
    quality_score = calculate_quality_score(quality_data, validation_results)
    
    print(f"\nData Quality Assessment:")
    print(f"Overall Quality Score: {quality_score['overall_score']:.1f}/100")
    print(f"Missing Data Penalty: {quality_score['missing_penalty']:.1f}%")
    print(f"Validation Issues Penalty: {quality_score['validation_penalty']:.1f}%")
    
    return (
        calculate_quality_score,
        expected_temp,
        extreme_temp_elev,
        max_shannon,
        quality_score,
        range_constraints,
        richness_abundance_issue,
        shannon_too_high,
        temp_residuals,
        validate_ecological_data,
        validation_results,
        zero_species_pos_abundance,
    )


@app.cell
def __():
    """
    ## Missing Data Imputation
    """
    # Prepare data for imputation
    def impute_missing_data(df, method='knn'):
        """Impute missing data using various methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].copy()
        
        imputation_results = {}
        
        # 1. Simple imputation methods
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            imputed_data = pd.DataFrame(
                imputer.fit_transform(df_numeric),
                columns=df_numeric.columns,
                index=df_numeric.index
            )
            
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            imputed_data = pd.DataFrame(
                imputer.fit_transform(df_numeric),
                columns=df_numeric.columns,
                index=df_numeric.index
            )
            
        elif method == 'knn':
            # K-Nearest Neighbors imputation
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = pd.DataFrame(
                imputer.fit_transform(df_numeric),
                columns=df_numeric.columns,
                index=df_numeric.index
            )
        
        # Track which values were imputed
        imputed_mask = df_numeric.isnull()
        
        return imputed_data, imputed_mask
    
    # Compare different imputation methods
    imputation_methods = ['mean', 'median', 'knn']
    imputation_comparison = {}
    
    for method in imputation_methods:
        imputed_data, imputed_mask = impute_missing_data(quality_data, method)
        imputation_comparison[method] = {
            'data': imputed_data,
            'mask': imputed_mask
        }
    
    # Evaluate imputation quality (where we have complete data)
    def evaluate_imputation_quality(original_data, imputed_data, imputed_mask):
        """Evaluate imputation quality using various metrics"""
        metrics = {}
        
        for column in imputed_data.columns:
            if imputed_mask[column].any():
                # Compare imputed values to original distribution
                original_values = original_data[column].dropna()
                imputed_values = imputed_data.loc[imputed_mask[column], column]
                
                if len(imputed_values) > 0 and len(original_values) > 0:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(original_values, imputed_values)
                    
                    # Mean and variance preservation
                    mean_diff = abs(imputed_values.mean() - original_values.mean())
                    var_ratio = imputed_values.var() / original_values.var()
                    
                    metrics[column] = {
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'mean_difference': mean_diff,
                        'variance_ratio': var_ratio,
                        'n_imputed': len(imputed_values)
                    }
        
        return metrics
    
    print("Imputation Method Comparison:")
    print("=" * 35)
    
    for method, results in imputation_comparison.items():
        print(f"\n{method.upper()} Imputation:")
        
        # Evaluate quality
        quality_metrics = evaluate_imputation_quality(
            quality_data, results['data'], results['mask']
        )
        
        for column, metrics in quality_metrics.items():
            print(f"  {column}:")
            print(f"    Imputed values: {metrics['n_imputed']}")
            print(f"    KS p-value: {metrics['ks_p_value']:.3f}")
            print(f"    Mean difference: {metrics['mean_difference']:.3f}")
            print(f"    Variance ratio: {metrics['variance_ratio']:.3f}")
    
    # Select best imputation method based on overall performance
    def select_best_imputation(comparison_results):
        """Select best imputation method based on multiple criteria"""
        method_scores = {}
        
        for method, results in comparison_results.items():
            quality_metrics = evaluate_imputation_quality(
                quality_data, results['data'], results['mask']
            )
            
            # Calculate composite score
            score = 0
            count = 0
            
            for column, metrics in quality_metrics.items():
                # Higher p-value is better (distributions more similar)
                score += metrics['ks_p_value']
                # Variance ratio closer to 1 is better
                score += (1 - abs(1 - metrics['variance_ratio']))
                count += 2
            
            if count > 0:
                method_scores[method] = score / count
        
        best_method = max(method_scores, key=method_scores.get)
        return best_method, method_scores
    
    best_method, method_scores = select_best_imputation(imputation_comparison)
    
    print(f"\nImputation Method Scores:")
    for method, score in method_scores.items():
        print(f"  {method}: {score:.3f}")
    
    print(f"\nBest imputation method: {best_method.upper()}")
    
    # Apply best imputation
    final_imputed_data = imputation_comparison[best_method]['data']
    
    return (
        best_method,
        evaluate_imputation_quality,
        final_imputed_data,
        imputation_comparison,
        imputation_methods,
        impute_missing_data,
        method_scores,
        quality_metrics,
        select_best_imputation,
    )


@app.cell
def __():
    """
    ## Robust Statistical Methods
    """
    # Implement robust alternatives for noisy data
    def robust_statistics(series):
        """Calculate robust statistical measures"""
        cleaned_series = series.dropna()
        
        if len(cleaned_series) == 0:
            return {}
        
        robust_stats = {
            # Central tendency
            'mean': cleaned_series.mean(),
            'median': cleaned_series.median(),
            'trimmed_mean_10': stats.trim_mean(cleaned_series, 0.1),
            'trimmed_mean_20': stats.trim_mean(cleaned_series, 0.2),
            
            # Variability
            'std': cleaned_series.std(),
            'mad': np.median(np.abs(cleaned_series - cleaned_series.median())),  # Median Absolute Deviation
            'iqr': cleaned_series.quantile(0.75) - cleaned_series.quantile(0.25),
            
            # Percentiles
            'q25': cleaned_series.quantile(0.25),
            'q75': cleaned_series.quantile(0.75),
            
            # Robust correlation (would need pairs of variables)
            'n_observations': len(cleaned_series)
        }
        
        return robust_stats
    
    # Calculate robust statistics for key variables
    print("Robust Statistics Comparison:")
    print("=" * 40)
    
    for column in ['temperature', 'species_richness', 'soil_pH']:
        if column in quality_data.columns:
            original_stats = robust_statistics(quality_data[column])
            imputed_stats = robust_statistics(final_imputed_data[column])
            
            print(f"\n{column.upper()}:")
            print(f"                    | Original | Imputed ")
            print("-" * 40)
            
            for stat_name in ['mean', 'median', 'trimmed_mean_10', 'std', 'mad']:
                if stat_name in original_stats and stat_name in imputed_stats:
                    orig_val = original_stats[stat_name]
                    imp_val = imputed_stats[stat_name]
                    print(f"{stat_name:18} | {orig_val:7.2f}  | {imp_val:7.2f}")
    
    # Robust correlation analysis
    def robust_correlation(df, method='spearman'):
        """Calculate robust correlations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'spearman':
            corr_matrix = df[numeric_cols].corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = df[numeric_cols].corr(method='kendall')
        else:
            corr_matrix = df[numeric_cols].corr(method='pearson')
        
        return corr_matrix
    
    # Compare correlation matrices
    pearson_corr = robust_correlation(final_imputed_data, 'pearson')
    spearman_corr = robust_correlation(final_imputed_data, 'spearman')
    
    print(f"\nCorrelation Comparison (Species Richness vs Temperature):")
    if 'species_richness' in pearson_corr.columns and 'temperature' in pearson_corr.columns:
        pearson_val = pearson_corr.loc['species_richness', 'temperature']
        spearman_val = spearman_corr.loc['species_richness', 'temperature']
        
        print(f"Pearson correlation:  {pearson_val:.3f}")
        print(f"Spearman correlation: {spearman_val:.3f}")
        
        if abs(pearson_val - spearman_val) > 0.1:
            print("Large difference suggests non-linear relationship or outliers")
    
    return (
        imputed_stats,
        original_stats,
        pearson_corr,
        pearson_val,
        robust_correlation,
        robust_statistics,
        spearman_corr,
        spearman_val,
    )


@app.cell
def __():
    """
    ## Data Cleaning Workflow
    """
    def comprehensive_data_cleaning(df, outlier_method='consensus', imputation_method='knn'):
        """Comprehensive data cleaning workflow"""
        
        cleaning_log = []
        df_cleaned = df.copy()
        
        # Step 1: Initial data assessment
        initial_shape = df_cleaned.shape
        initial_missing = df_cleaned.isnull().sum().sum()
        cleaning_log.append(f"Initial data: {initial_shape[0]} rows, {initial_shape[1]} columns")
        cleaning_log.append(f"Initial missing values: {initial_missing}")
        
        # Step 2: Remove completely empty rows/columns
        # Remove rows that are completely empty
        empty_rows = df_cleaned.isnull().all(axis=1)
        if empty_rows.any():
            df_cleaned = df_cleaned[~empty_rows]
            cleaning_log.append(f"Removed {empty_rows.sum()} completely empty rows")
        
        # Remove columns that are completely empty
        empty_cols = df_cleaned.isnull().all(axis=0)
        if empty_cols.any():
            df_cleaned = df_cleaned.loc[:, ~empty_cols]
            cleaning_log.append(f"Removed {empty_cols.sum()} completely empty columns")
        
        # Step 3: Handle extreme outliers (likely data entry errors)
        outliers_removed = 0
        
        if outlier_method == 'consensus':
            # Use consensus from multiple methods
            multivariate_outliers_new = detect_multivariate_outliers(df_cleaned, contamination=0.05)
            
            if multivariate_outliers_new:
                outlier_counts_new = pd.DataFrame(multivariate_outliers_new).sum(axis=1)
                extreme_outliers = outlier_counts_new >= 2
                
                if extreme_outliers.any():
                    outliers_removed = extreme_outliers.sum()
                    df_cleaned = df_cleaned[~extreme_outliers]
                    cleaning_log.append(f"Removed {outliers_removed} extreme outliers")
        
        # Step 4: Fix obvious data entry errors
        # Example: Fix precipitation unit errors (values > 3000 likely in wrong units)
        if 'precipitation' in df_cleaned.columns:
            precip_errors = df_cleaned['precipitation'] > 3000
            if precip_errors.any():
                df_cleaned.loc[precip_errors, 'precipitation'] /= 10  # Convert mm to cm
                cleaning_log.append(f"Fixed {precip_errors.sum()} precipitation unit errors")
        
        # Step 5: Handle impossible values
        impossible_fixed = 0
        
        # Zero species with positive abundance
        if 'species_richness' in df_cleaned.columns and 'total_abundance' in df_cleaned.columns:
            impossible_cases = (df_cleaned['species_richness'] == 0) & (df_cleaned['total_abundance'] > 0)
            if impossible_cases.any():
                # Set species richness to 1 (minimum possible)
                df_cleaned.loc[impossible_cases, 'species_richness'] = 1
                impossible_fixed += impossible_cases.sum()
        
        if impossible_fixed > 0:
            cleaning_log.append(f"Fixed {impossible_fixed} impossible value combinations")
        
        # Step 6: Impute missing values
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        missing_before = df_cleaned[numeric_cols].isnull().sum().sum()
        
        if missing_before > 0:
            if imputation_method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            elif imputation_method == 'median':
                imputer = SimpleImputer(strategy='median')
            else:
                imputer = SimpleImputer(strategy='mean')
            
            df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
            cleaning_log.append(f"Imputed {missing_before} missing values using {imputation_method}")
        
        # Step 7: Final validation
        final_validation = validate_ecological_data(df_cleaned)
        remaining_issues = len(final_validation)
        cleaning_log.append(f"Final validation: {remaining_issues} issues remaining")
        
        # Step 8: Calculate final quality score
        final_quality = calculate_quality_score(df_cleaned, final_validation)
        cleaning_log.append(f"Final quality score: {final_quality['overall_score']:.1f}/100")
        
        return df_cleaned, cleaning_log, final_validation
    
    # Apply comprehensive cleaning
    cleaned_data, cleaning_log, final_validation = comprehensive_data_cleaning(quality_data)
    
    print("Data Cleaning Workflow:")
    print("=" * 30)
    
    for step in cleaning_log:
        print(f"• {step}")
    
    # Compare before and after
    print(f"\nBefore vs After Cleaning:")
    print(f"Shape: {quality_data.shape} → {cleaned_data.shape}")
    print(f"Missing values: {quality_data.isnull().sum().sum()} → {cleaned_data.isnull().sum().sum()}")
    print(f"Validation issues: {len(validation_results)} → {len(final_validation)}")
    
    # Final data quality assessment
    original_quality = calculate_quality_score(quality_data, validation_results)
    final_quality_score = calculate_quality_score(cleaned_data, final_validation)
    
    print(f"\nQuality Score Improvement:")
    print(f"Original: {original_quality['overall_score']:.1f}/100")
    print(f"Cleaned:  {final_quality_score['overall_score']:.1f}/100")
    print(f"Improvement: {final_quality_score['overall_score'] - original_quality['overall_score']:+.1f} points")
    
    return (
        cleaned_data,
        cleaning_log,
        comprehensive_data_cleaning,
        final_quality_score,
        final_validation,
        impossible_fixed,
        outliers_removed,
    )


@app.cell
def __():
    """
    ## Summary and Best Practices\n
    In this chapter, we covered comprehensive data quality assessment and improvement:\n
    ✓ **Missing data analysis**: Patterns, correlations, and MCAR testing
    ✓ **Outlier detection**: Univariate and multivariate methods
    ✓ **Data validation**: Range checks and logical constraints
    ✓ **Missing data imputation**: Multiple methods and quality evaluation
    ✓ **Robust statistics**: Methods for noisy ecological data
    ✓ **Comprehensive cleaning**: End-to-end workflow\n
    ### Key Python Packages for Data Quality:
    - **pandas**: Data manipulation and missing value handling
    - **numpy**: Numerical operations and array processing
    - **scipy.stats**: Statistical tests and robust methods
    - **sklearn**: Machine learning-based outlier detection and imputation
    - **holoviews**: Data quality visualization\n
    ### Best Practices for Ecological Data Quality:
    1. **Document everything**: Keep detailed logs of all cleaning steps
    2. **Validate domain knowledge**: Use ecological constraints to check data
    3. **Multiple methods**: Use several outlier detection approaches
    4. **Preserve originals**: Always keep raw data unchanged
    5. **Iterative process**: Clean, validate, and repeat as needed
    6. **Quality metrics**: Quantify and track data quality improvements
    7. **Bias awareness**: Consider how cleaning might introduce bias
    8. **Expert review**: Have domain experts review flagged issues\n
    ### Common Data Quality Issues in Ecology:
    - **Measurement errors**: Equipment malfunction, human error
    - **Unit confusion**: Mixed units (mm vs cm, °C vs °F)
    - **Temporal misalignment**: Wrong dates or time zones
    - **Spatial errors**: Incorrect coordinates or projections
    - **Species misidentification**: Taxonomic errors
    - **Sampling bias**: Non-representative data collection
    - **Transcription errors**: Manual data entry mistakes
    """
    
    quality_summary = {
        'Data Quality Assessment': {
            'Original Quality Score': f"{original_quality['overall_score']:.1f}/100",
            'Final Quality Score': f"{final_quality_score['overall_score']:.1f}/100",
            'Improvement': f"{final_quality_score['overall_score'] - original_quality['overall_score']:+.1f} points"
        },
        'Issues Detected': {
            'Missing Values': f"{quality_data.isnull().sum().sum()}",
            'Validation Issues': f"{len(validation_results)}",
            'Consensus Outliers': f"{consensus_outliers.sum()}"
        },
        'Cleaning Actions': {
            'Outliers Removed': f"{outliers_removed}",
            'Values Imputed': f"{quality_data.isnull().sum().sum()}",
            'Issues Fixed': f"{impossible_fixed}"
        },
        'Final Dataset': {
            'Shape': f"{cleaned_data.shape[0]} × {cleaned_data.shape[1]}",
            'Missing Values': f"{cleaned_data.isnull().sum().sum()}",
            'Remaining Issues': f"{len(final_validation)}"
        }
    }
    
    print("Data Quality Assessment Summary:")
    print("=" * 40)
    
    for category, details in quality_summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Chapter 9 complete! Ready for machine learning applications.")
    
    return quality_summary,


if __name__ == "__main__":
    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
