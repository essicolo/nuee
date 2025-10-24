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
    
    print(f"\\nMissing Data Patterns:")
    print("Variables Missing | Number of Cases")
    print("-" * 35)
    for n_missing, count in missing_patterns.items():
        print(f"       {n_missing:2d}         |      {count:3d}")
    
    print(f"\\nMissing Data Correlations:")
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
    
    print(f"\\nMCAR Test Results (simplified):") 
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
    outlier_results = {}\n    
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
        
        methods = {}\n        
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
            
            mahal_distances = []\n            for i in range(len(data_scaled)):\n                diff = data_scaled[i] - mean_vector\n                mahal_dist = np.sqrt(diff @ inv_cov_matrix @ diff.T)\n                mahal_distances.append(mahal_dist)\n            \n            mahal_threshold = np.percentile(mahal_distances, (1-contamination)*100)\n            mahal_outliers = np.array(mahal_distances) > mahal_threshold\n            methods['mahalanobis'] = pd.Series(False, index=df.index)\n            methods['mahalanobis'].loc[data_clean.index] = mahal_outliers\n            \n        except np.linalg.LinAlgError:\n            methods['mahalanobis'] = pd.Series(False, index=df.index)\n        \n        return methods\n    \n    multivariate_outliers = detect_multivariate_outliers(quality_data)\n    \n    print(f\"\\nMultivariate Outlier Detection:\")\n    for method, outliers in multivariate_outliers.items():\n        print(f\"{method.replace('_', ' ').title()}: {outliers.sum()} outliers\")\n    \n    # Consensus outliers (detected by multiple methods)\n    outlier_counts = pd.DataFrame(multivariate_outliers).sum(axis=1)\n    consensus_outliers = outlier_counts >= 2\n    \n    print(f\"\\nConsensus outliers (2+ methods): {consensus_outliers.sum()}\")\n    \n    return (\n        consensus_outliers,\n        detect_multivariate_outliers,\n        detect_univariate_outliers,\n        iqr_count,\n        methods,\n        mod_zscore_count,\n        multivariate_outliers,\n        outlier_counts,\n        outlier_results,\n        zscore_count,\n    )


@app.cell
def __():
    """
    ## Data Validation and Logical Constraints
    """
    def validate_ecological_data(df):
        \"\"\"Validate ecological data against known constraints\"\"\"\n        validation_issues = []\n        \n        # 1. Range checks\n        range_constraints = {\n            'temperature': (-50, 60),  # Celsius\n            'precipitation': (0, 5000),  # mm/year\n            'elevation': (0, 9000),  # meters\n            'soil_pH': (0, 14),\n            'species_richness': (0, 1000),\n            'total_abundance': (0, 100000),\n            'shannon_diversity': (0, 10)\n        }\n        \n        for column, (min_val, max_val) in range_constraints.items():\n            if column in df.columns:\n                out_of_range = (df[column] < min_val) | (df[column] > max_val)\n                if out_of_range.any():\n                    validation_issues.append({\n                        'type': 'range_violation',\n                        'column': column,\n                        'count': out_of_range.sum(),\n                        'indices': df.index[out_of_range].tolist()\n                    })\n        \n        # 2. Logical consistency checks\n        \n        # Species richness should be <= total abundance (can't have more species than individuals)\n        if 'species_richness' in df.columns and 'total_abundance' in df.columns:\n            richness_abundance_issue = df['species_richness'] > df['total_abundance']\n            if richness_abundance_issue.any():\n                validation_issues.append({\n                    'type': 'logical_inconsistency',\n                    'description': 'Species richness > total abundance',\n                    'count': richness_abundance_issue.sum(),\n                    'indices': df.index[richness_abundance_issue].tolist()\n                })\n        \n        # Shannon diversity should be <= log(species richness)\n        if 'shannon_diversity' in df.columns and 'species_richness' in df.columns:\n            max_shannon = np.log(df['species_richness'].replace(0, 1))  # Avoid log(0)\n            shannon_too_high = df['shannon_diversity'] > max_shannon * 1.1  # Small tolerance\n            if shannon_too_high.any():\n                validation_issues.append({\n                    'type': 'logical_inconsistency',\n                    'description': 'Shannon diversity too high for species richness',\n                    'count': shannon_too_high.sum(),\n                    'indices': df.index[shannon_too_high].tolist()\n                })\n        \n        # Zero species but positive abundance\n        if 'species_richness' in df.columns and 'total_abundance' in df.columns:\n            zero_species_pos_abundance = (df['species_richness'] == 0) & (df['total_abundance'] > 0)\n            if zero_species_pos_abundance.any():\n                validation_issues.append({\n                    'type': 'logical_inconsistency',\n                    'description': 'Zero species but positive abundance',\n                    'count': zero_species_pos_abundance.sum(),\n                    'indices': df.index[zero_species_pos_abundance].tolist()\n                })\n        \n        # 3. Statistical outliers in relationships\n        \n        # Check for unusual temperature-elevation relationships\n        if 'temperature' in df.columns and 'elevation' in df.columns:\n            # Expected: temperature decreases with elevation (~6.5°C per 1000m)\n            expected_temp = 20 - (df['elevation'] / 1000) * 6.5\n            temp_residuals = df['temperature'] - expected_temp\n            extreme_temp_elev = np.abs(temp_residuals) > 15  # > 15°C deviation\n            \n            if extreme_temp_elev.any():\n                validation_issues.append({\n                    'type': 'relationship_outlier',\n                    'description': 'Unusual temperature-elevation relationship',\n                    'count': extreme_temp_elev.sum(),\n                    'indices': df.index[extreme_temp_elev].tolist()\n                })\n        \n        return validation_issues\n    \n    # Validate the dataset\n    validation_results = validate_ecological_data(quality_data)\n    \n    print(\"Data Validation Results:\")\n    print(\"=\" * 30)\n    \n    if not validation_results:\n        print(\"No validation issues detected.\")\n    else:\n        for issue in validation_results:\n            print(f\"\\n{issue['type'].replace('_', ' ').title()}:\")\n            if 'column' in issue:\n                print(f\"  Column: {issue['column']}\")\n            if 'description' in issue:\n                print(f\"  Description: {issue['description']}\")\n            print(f\"  Count: {issue['count']}\")\n            print(f\"  Sample indices: {issue['indices'][:5]}...\")  # Show first 5\n    \n    # Create a data quality score\n    def calculate_quality_score(df, validation_issues):\n        \"\"\"Calculate overall data quality score (0-100)\"\"\"\n        total_cells = df.shape[0] * df.shape[1]\n        \n        # Penalties\n        missing_penalty = df.isnull().sum().sum() / total_cells * 100\n        \n        validation_penalty = 0\n        for issue in validation_issues:\n            validation_penalty += issue['count']\n        validation_penalty = (validation_penalty / df.shape[0]) * 100\n        \n        # Quality score (higher is better)\n        quality_score = 100 - missing_penalty - validation_penalty\n        quality_score = max(0, quality_score)  # Ensure non-negative\n        \n        return {\n            'overall_score': quality_score,\n            'missing_penalty': missing_penalty,\n            'validation_penalty': validation_penalty\n        }\n    \n    quality_score = calculate_quality_score(quality_data, validation_results)\n    \n    print(f\"\\nData Quality Assessment:\")\n    print(f\"Overall Quality Score: {quality_score['overall_score']:.1f}/100\")\n    print(f\"Missing Data Penalty: {quality_score['missing_penalty']:.1f}%\")\n    print(f\"Validation Issues Penalty: {quality_score['validation_penalty']:.1f}%\")\n    \n    return (\n        calculate_quality_score,\n        expected_temp,\n        extreme_temp_elev,\n        max_shannon,\n        quality_score,\n        range_constraints,\n        richness_abundance_issue,\n        shannon_too_high,\n        temp_residuals,\n        validate_ecological_data,\n        validation_results,\n        zero_species_pos_abundance,\n    )


@app.cell
def __():
    \"\"\"
    ## Missing Data Imputation
    \"\"\"\n    # Prepare data for imputation\n    def impute_missing_data(df, method='knn'):\n        \"\"\"Impute missing data using various methods\"\"\"\n        numeric_cols = df.select_dtypes(include=[np.number]).columns\n        df_numeric = df[numeric_cols].copy()\n        \n        imputation_results = {}\n        \n        # 1. Simple imputation methods\n        if method == 'mean':\n            imputer = SimpleImputer(strategy='mean')\n            imputed_data = pd.DataFrame(\n                imputer.fit_transform(df_numeric),\n                columns=df_numeric.columns,\n                index=df_numeric.index\n            )\n            \n        elif method == 'median':\n            imputer = SimpleImputer(strategy='median')\n            imputed_data = pd.DataFrame(\n                imputer.fit_transform(df_numeric),\n                columns=df_numeric.columns,\n                index=df_numeric.index\n            )\n            \n        elif method == 'knn':\n            # K-Nearest Neighbors imputation\n            imputer = KNNImputer(n_neighbors=5)\n            imputed_data = pd.DataFrame(\n                imputer.fit_transform(df_numeric),\n                columns=df_numeric.columns,\n                index=df_numeric.index\n            )\n        \n        # Track which values were imputed\n        imputed_mask = df_numeric.isnull()\n        \n        return imputed_data, imputed_mask\n    \n    # Compare different imputation methods\n    imputation_methods = ['mean', 'median', 'knn']\n    imputation_comparison = {}\n    \n    for method in imputation_methods:\n        imputed_data, imputed_mask = impute_missing_data(quality_data, method)\n        imputation_comparison[method] = {\n            'data': imputed_data,\n            'mask': imputed_mask\n        }\n    \n    # Evaluate imputation quality (where we have complete data)\n    def evaluate_imputation_quality(original_data, imputed_data, imputed_mask):\n        \"\"\"Evaluate imputation quality using various metrics\"\"\"\n        metrics = {}\n        \n        for column in imputed_data.columns:\n            if imputed_mask[column].any():\n                # Compare imputed values to original distribution\n                original_values = original_data[column].dropna()\n                imputed_values = imputed_data.loc[imputed_mask[column], column]\n                \n                if len(imputed_values) > 0 and len(original_values) > 0:\n                    # Kolmogorov-Smirnov test\n                    ks_stat, ks_p = stats.ks_2samp(original_values, imputed_values)\n                    \n                    # Mean and variance preservation\n                    mean_diff = abs(imputed_values.mean() - original_values.mean())\n                    var_ratio = imputed_values.var() / original_values.var()\n                    \n                    metrics[column] = {\n                        'ks_statistic': ks_stat,\n                        'ks_p_value': ks_p,\n                        'mean_difference': mean_diff,\n                        'variance_ratio': var_ratio,\n                        'n_imputed': len(imputed_values)\n                    }\n        \n        return metrics\n    \n    print(\"Imputation Method Comparison:\")\n    print(\"=\" * 35)\n    \n    for method, results in imputation_comparison.items():\n        print(f\"\\n{method.upper()} Imputation:\")\n        \n        # Evaluate quality\n        quality_metrics = evaluate_imputation_quality(\n            quality_data, results['data'], results['mask']\n        )\n        \n        for column, metrics in quality_metrics.items():\n            print(f\"  {column}:\")\n            print(f\"    Imputed values: {metrics['n_imputed']}\")\n            print(f\"    KS p-value: {metrics['ks_p_value']:.3f}\")\n            print(f\"    Mean difference: {metrics['mean_difference']:.3f}\")\n            print(f\"    Variance ratio: {metrics['variance_ratio']:.3f}\")\n    \n    # Select best imputation method based on overall performance\n    def select_best_imputation(comparison_results):\n        \"\"\"Select best imputation method based on multiple criteria\"\"\"\n        method_scores = {}\n        \n        for method, results in comparison_results.items():\n            quality_metrics = evaluate_imputation_quality(\n                quality_data, results['data'], results['mask']\n            )\n            \n            # Calculate composite score\n            score = 0\n            count = 0\n            \n            for column, metrics in quality_metrics.items():\n                # Higher p-value is better (distributions more similar)\n                score += metrics['ks_p_value']\n                # Variance ratio closer to 1 is better\n                score += (1 - abs(1 - metrics['variance_ratio']))\n                count += 2\n            \n            if count > 0:\n                method_scores[method] = score / count\n        \n        best_method = max(method_scores, key=method_scores.get)\n        return best_method, method_scores\n    \n    best_method, method_scores = select_best_imputation(imputation_comparison)\n    \n    print(f\"\\nImputation Method Scores:\")\n    for method, score in method_scores.items():\n        print(f\"  {method}: {score:.3f}\")\n    \n    print(f\"\\nBest imputation method: {best_method.upper()}\")\n    \n    # Apply best imputation\n    final_imputed_data = imputation_comparison[best_method]['data']\n    \n    return (\n        best_method,\n        evaluate_imputation_quality,\n        final_imputed_data,\n        imputation_comparison,\n        imputation_methods,\n        impute_missing_data,\n        method_scores,\n        quality_metrics,\n        select_best_imputation,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Robust Statistical Methods\n    \"\"\"\n    # Implement robust alternatives for noisy data\n    def robust_statistics(series):\n        \"\"\"Calculate robust statistical measures\"\"\"\n        cleaned_series = series.dropna()\n        \n        if len(cleaned_series) == 0:\n            return {}\n        \n        robust_stats = {\n            # Central tendency\n            'mean': cleaned_series.mean(),\n            'median': cleaned_series.median(),\n            'trimmed_mean_10': stats.trim_mean(cleaned_series, 0.1),\n            'trimmed_mean_20': stats.trim_mean(cleaned_series, 0.2),\n            \n            # Variability\n            'std': cleaned_series.std(),\n            'mad': np.median(np.abs(cleaned_series - cleaned_series.median())),  # Median Absolute Deviation\n            'iqr': cleaned_series.quantile(0.75) - cleaned_series.quantile(0.25),\n            \n            # Percentiles\n            'q25': cleaned_series.quantile(0.25),\n            'q75': cleaned_series.quantile(0.75),\n            \n            # Robust correlation (would need pairs of variables)\n            'n_observations': len(cleaned_series)\n        }\n        \n        return robust_stats\n    \n    # Calculate robust statistics for key variables\n    print(\"Robust Statistics Comparison:\")\n    print(\"=\" * 40)\n    \n    for column in ['temperature', 'species_richness', 'soil_pH']:\n        if column in quality_data.columns:\n            original_stats = robust_statistics(quality_data[column])\n            imputed_stats = robust_statistics(final_imputed_data[column])\n            \n            print(f\"\\n{column.upper()}:\")\n            print(f\"                    | Original | Imputed \")\n            print(\"-\" * 40)\n            \n            for stat_name in ['mean', 'median', 'trimmed_mean_10', 'std', 'mad']:\n                if stat_name in original_stats and stat_name in imputed_stats:\n                    orig_val = original_stats[stat_name]\n                    imp_val = imputed_stats[stat_name]\n                    print(f\"{stat_name:18} | {orig_val:7.2f}  | {imp_val:7.2f}\")\n    \n    # Robust correlation analysis\n    def robust_correlation(df, method='spearman'):\n        \"\"\"Calculate robust correlations\"\"\"\n        numeric_cols = df.select_dtypes(include=[np.number]).columns\n        \n        if method == 'spearman':\n            corr_matrix = df[numeric_cols].corr(method='spearman')\n        elif method == 'kendall':\n            corr_matrix = df[numeric_cols].corr(method='kendall')\n        else:\n            corr_matrix = df[numeric_cols].corr(method='pearson')\n        \n        return corr_matrix\n    \n    # Compare correlation matrices\n    pearson_corr = robust_correlation(final_imputed_data, 'pearson')\n    spearman_corr = robust_correlation(final_imputed_data, 'spearman')\n    \n    print(f\"\\nCorrelation Comparison (Species Richness vs Temperature):\")\n    if 'species_richness' in pearson_corr.columns and 'temperature' in pearson_corr.columns:\n        pearson_val = pearson_corr.loc['species_richness', 'temperature']\n        spearman_val = spearman_corr.loc['species_richness', 'temperature']\n        \n        print(f\"Pearson correlation:  {pearson_val:.3f}\")\n        print(f\"Spearman correlation: {spearman_val:.3f}\")\n        \n        if abs(pearson_val - spearman_val) > 0.1:\n            print(\"Large difference suggests non-linear relationship or outliers\")\n    \n    return (\n        imputed_stats,\n        original_stats,\n        pearson_corr,\n        pearson_val,\n        robust_correlation,\n        robust_statistics,\n        spearman_corr,\n        spearman_val,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Data Cleaning Workflow\n    \"\"\"\n    def comprehensive_data_cleaning(df, outlier_method='consensus', imputation_method='knn'):\n        \"\"\"Comprehensive data cleaning workflow\"\"\"\n        \n        cleaning_log = []\n        df_cleaned = df.copy()\n        \n        # Step 1: Initial data assessment\n        initial_shape = df_cleaned.shape\n        initial_missing = df_cleaned.isnull().sum().sum()\n        cleaning_log.append(f\"Initial data: {initial_shape[0]} rows, {initial_shape[1]} columns\")\n        cleaning_log.append(f\"Initial missing values: {initial_missing}\")\n        \n        # Step 2: Remove completely empty rows/columns\n        # Remove rows that are completely empty\n        empty_rows = df_cleaned.isnull().all(axis=1)\n        if empty_rows.any():\n            df_cleaned = df_cleaned[~empty_rows]\n            cleaning_log.append(f\"Removed {empty_rows.sum()} completely empty rows\")\n        \n        # Remove columns that are completely empty\n        empty_cols = df_cleaned.isnull().all(axis=0)\n        if empty_cols.any():\n            df_cleaned = df_cleaned.loc[:, ~empty_cols]\n            cleaning_log.append(f\"Removed {empty_cols.sum()} completely empty columns\")\n        \n        # Step 3: Handle extreme outliers (likely data entry errors)\n        outliers_removed = 0\n        \n        if outlier_method == 'consensus':\n            # Use consensus from multiple methods\n            multivariate_outliers_new = detect_multivariate_outliers(df_cleaned, contamination=0.05)\n            \n            if multivariate_outliers_new:\n                outlier_counts_new = pd.DataFrame(multivariate_outliers_new).sum(axis=1)\n                extreme_outliers = outlier_counts_new >= 2\n                \n                if extreme_outliers.any():\n                    outliers_removed = extreme_outliers.sum()\n                    df_cleaned = df_cleaned[~extreme_outliers]\n                    cleaning_log.append(f\"Removed {outliers_removed} extreme outliers\")\n        \n        # Step 4: Fix obvious data entry errors\n        # Example: Fix precipitation unit errors (values > 3000 likely in wrong units)\n        if 'precipitation' in df_cleaned.columns:\n            precip_errors = df_cleaned['precipitation'] > 3000\n            if precip_errors.any():\n                df_cleaned.loc[precip_errors, 'precipitation'] /= 10  # Convert mm to cm\n                cleaning_log.append(f\"Fixed {precip_errors.sum()} precipitation unit errors\")\n        \n        # Step 5: Handle impossible values\n        impossible_fixed = 0\n        \n        # Zero species with positive abundance\n        if 'species_richness' in df_cleaned.columns and 'total_abundance' in df_cleaned.columns:\n            impossible_cases = (df_cleaned['species_richness'] == 0) & (df_cleaned['total_abundance'] > 0)\n            if impossible_cases.any():\n                # Set species richness to 1 (minimum possible)\n                df_cleaned.loc[impossible_cases, 'species_richness'] = 1\n                impossible_fixed += impossible_cases.sum()\n        \n        if impossible_fixed > 0:\n            cleaning_log.append(f\"Fixed {impossible_fixed} impossible value combinations\")\n        \n        # Step 6: Impute missing values\n        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns\n        missing_before = df_cleaned[numeric_cols].isnull().sum().sum()\n        \n        if missing_before > 0:\n            if imputation_method == 'knn':\n                imputer = KNNImputer(n_neighbors=5)\n            elif imputation_method == 'median':\n                imputer = SimpleImputer(strategy='median')\n            else:\n                imputer = SimpleImputer(strategy='mean')\n            \n            df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])\n            cleaning_log.append(f\"Imputed {missing_before} missing values using {imputation_method}\")\n        \n        # Step 7: Final validation\n        final_validation = validate_ecological_data(df_cleaned)\n        remaining_issues = len(final_validation)\n        cleaning_log.append(f\"Final validation: {remaining_issues} issues remaining\")\n        \n        # Step 8: Calculate final quality score\n        final_quality = calculate_quality_score(df_cleaned, final_validation)\n        cleaning_log.append(f\"Final quality score: {final_quality['overall_score']:.1f}/100\")\n        \n        return df_cleaned, cleaning_log, final_validation\n    \n    # Apply comprehensive cleaning\n    cleaned_data, cleaning_log, final_validation = comprehensive_data_cleaning(quality_data)\n    \n    print(\"Data Cleaning Workflow:\")\n    print(\"=\" * 30)\n    \n    for step in cleaning_log:\n        print(f\"• {step}\")\n    \n    # Compare before and after\n    print(f\"\\nBefore vs After Cleaning:\")\n    print(f\"Shape: {quality_data.shape} → {cleaned_data.shape}\")\n    print(f\"Missing values: {quality_data.isnull().sum().sum()} → {cleaned_data.isnull().sum().sum()}\")\n    print(f\"Validation issues: {len(validation_results)} → {len(final_validation)}\")\n    \n    # Final data quality assessment\n    original_quality = calculate_quality_score(quality_data, validation_results)\n    final_quality_score = calculate_quality_score(cleaned_data, final_validation)\n    \n    print(f\"\\nQuality Score Improvement:\")\n    print(f\"Original: {original_quality['overall_score']:.1f}/100\")\n    print(f\"Cleaned:  {final_quality_score['overall_score']:.1f}/100\")\n    print(f\"Improvement: {final_quality_score['overall_score'] - original_quality['overall_score']:+.1f} points\")\n    \n    return (\n        cleaned_data,\n        cleaning_log,\n        comprehensive_data_cleaning,\n        final_quality_score,\n        final_validation,\n        impossible_fixed,\n        outliers_removed,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Summary and Best Practices\n\n    In this chapter, we covered comprehensive data quality assessment and improvement:\n\n    ✓ **Missing data analysis**: Patterns, correlations, and MCAR testing\n    ✓ **Outlier detection**: Univariate and multivariate methods\n    ✓ **Data validation**: Range checks and logical constraints\n    ✓ **Missing data imputation**: Multiple methods and quality evaluation\n    ✓ **Robust statistics**: Methods for noisy ecological data\n    ✓ **Comprehensive cleaning**: End-to-end workflow\n\n    ### Key Python Packages for Data Quality:\n    - **pandas**: Data manipulation and missing value handling\n    - **numpy**: Numerical operations and array processing\n    - **scipy.stats**: Statistical tests and robust methods\n    - **sklearn**: Machine learning-based outlier detection and imputation\n    - **holoviews**: Data quality visualization\n\n    ### Best Practices for Ecological Data Quality:\n    1. **Document everything**: Keep detailed logs of all cleaning steps\n    2. **Validate domain knowledge**: Use ecological constraints to check data\n    3. **Multiple methods**: Use several outlier detection approaches\n    4. **Preserve originals**: Always keep raw data unchanged\n    5. **Iterative process**: Clean, validate, and repeat as needed\n    6. **Quality metrics**: Quantify and track data quality improvements\n    7. **Bias awareness**: Consider how cleaning might introduce bias\n    8. **Expert review**: Have domain experts review flagged issues\n\n    ### Common Data Quality Issues in Ecology:\n    - **Measurement errors**: Equipment malfunction, human error\n    - **Unit confusion**: Mixed units (mm vs cm, °C vs °F)\n    - **Temporal misalignment**: Wrong dates or time zones\n    - **Spatial errors**: Incorrect coordinates or projections\n    - **Species misidentification**: Taxonomic errors\n    - **Sampling bias**: Non-representative data collection\n    - **Transcription errors**: Manual data entry mistakes\n    \"\"\"\n    \n    quality_summary = {\n        'Data Quality Assessment': {\n            'Original Quality Score': f\"{original_quality['overall_score']:.1f}/100\",\n            'Final Quality Score': f\"{final_quality_score['overall_score']:.1f}/100\",\n            'Improvement': f\"{final_quality_score['overall_score'] - original_quality['overall_score']:+.1f} points\"\n        },\n        'Issues Detected': {\n            'Missing Values': f\"{quality_data.isnull().sum().sum()}\",\n            'Validation Issues': f\"{len(validation_results)}\",\n            'Consensus Outliers': f\"{consensus_outliers.sum()}\"\n        },\n        'Cleaning Actions': {\n            'Outliers Removed': f\"{outliers_removed}\",\n            'Values Imputed': f\"{quality_data.isnull().sum().sum()}\",\n            'Issues Fixed': f\"{impossible_fixed}\"\n        },\n        'Final Dataset': {\n            'Shape': f\"{cleaned_data.shape[0]} × {cleaned_data.shape[1]}\",\n            'Missing Values': f\"{cleaned_data.isnull().sum().sum()}\",\n            'Remaining Issues': f\"{len(final_validation)}\"\n        }\n    }\n    \n    print(\"Data Quality Assessment Summary:\")\n    print(\"=\" * 40)\n    \n    for category, details in quality_summary.items():\n        print(f\"\\n{category}:\")\n        for key, value in details.items():\n            print(f\"  {key}: {value}\")\n    \n    print(\"\\n✓ Chapter 9 complete! Ready for machine learning applications.\")\n    \n    return quality_summary,\n\n\nif __name__ == \"__main__\":\n    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
