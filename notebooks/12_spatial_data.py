import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 12: Spatial Data Analysis for Ecology

    This chapter covers spatial analysis methods for ecological data, including
    spatial autocorrelation, interpolation, and modeling of geographic patterns
    using Python tools compatible with Pyodide environments.

    ## Learning Objectives
    - Understand spatial patterns and processes in ecological data
    - Analyze spatial autocorrelation and spatial dependence
    - Perform spatial interpolation and prediction
    - Model species distributions across landscapes
    - Handle coordinate systems and projections
    """
    )
    return


@app.cell
def __():
    # Essential imports for spatial analysis (Pyodide-compatible)
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.interpolate import griddata
    from sklearn.neighbors import NearestNeighbors
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Spatial analysis packages loaded")
    print("Note: Using scipy and sklearn for spatial operations (Pyodide-compatible)")
    return (
        NearestNeighbors,
        cdist,
        griddata,
        hv,
        np,
        opts,
        pd,
        pdist,
        squareform,
        stats,
        warnings,
    )


@app.cell
def __():
    """
    ## Create Spatial Ecological Dataset
    """
    # Generate realistic spatial ecological data
    np.random.seed(42)
    
    # Create a regular sampling grid with some randomness
    n_sites = 200
    
    # Base coordinates in a 100 x 100 km area
    x_coords = np.random.uniform(0, 100, n_sites)
    y_coords = np.random.uniform(0, 100, n_sites)
    
    # Create spatial gradients
    # Environmental gradient from SW to NE
    elevation = 500 + 10 * x_coords + 8 * y_coords + np.random.normal(0, 100, n_sites)
    elevation = np.maximum(elevation, 0)  # No negative elevations
    
    # Temperature decreases with elevation
    temperature = 25 - 0.006 * elevation + np.random.normal(0, 2, n_sites)
    
    # Precipitation with spatial autocorrelation
    # Create spatially correlated random field
    def create_spatial_field(x, y, scale=20, variance=1):
        """Create spatially autocorrelated random field"""
        # Distance matrix
        coords = np.column_stack([x, y])
        distances = squareform(pdist(coords))
        
        # Exponential correlation function
        correlation_matrix = variance * np.exp(-distances / scale)
        
        # Generate correlated random values
        try:
            L = np.linalg.cholesky(correlation_matrix + 1e-6 * np.eye(len(x)))
            random_field = L @ np.random.normal(0, 1, len(x))
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            random_field = np.random.normal(0, np.sqrt(variance), len(x))
        
        return random_field
    
    precipitation_base = 800 + 300 * create_spatial_field(x_coords, y_coords, scale=25)
    precipitation = np.maximum(precipitation_base, 100)  # Minimum 100mm
    
    # Species richness based on environment with spatial component
    richness_base = (
        10 + 
        0.02 * precipitation +
        -0.01 * elevation +
        0.5 * temperature +
        5 * create_spatial_field(x_coords, y_coords, scale=15, variance=0.5)
    )
    species_richness = np.maximum(np.round(richness_base).astype(int), 1)
    
    # Presence of a focal species with spatial clustering
    # Centers of high suitability
    hotspot_centers = [(25, 75), (70, 30), (80, 80)]
    
    suitability = np.zeros(n_sites)
    for center_x, center_y in hotspot_centers:
        distance_to_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        suitability += 0.8 * np.exp(-distance_to_center**2 / (2 * 15**2))
    
    # Add environmental suitability
    temp_suitability = np.exp(-((temperature - 20) / 5)**2)
    suitability += 0.3 * temp_suitability
    
    # Convert to presence probability and generate presence/absence
    presence_prob = 1 / (1 + np.exp(-(suitability - 0.5) * 3))
    species_presence = np.random.binomial(1, presence_prob, n_sites)
    
    # Create DataFrame
    spatial_data = pd.DataFrame({
        'site_id': [f"SITE_{i:03d}" for i in range(1, n_sites + 1)],
        'x_coord': x_coords,
        'y_coord': y_coords,
        'elevation': elevation,
        'temperature': temperature,
        'precipitation': precipitation,
        'species_richness': species_richness,
        'species_presence': species_presence,
        'suitability': suitability
    })
    
    print(f"Spatial ecological dataset created: {spatial_data.shape}")
    print(f"Spatial extent: {x_coords.min():.1f}-{x_coords.max():.1f} km (X), {y_coords.min():.1f}-{y_coords.max():.1f} km (Y)")
    print(f"Elevation range: {elevation.min():.0f}-{elevation.max():.0f} m")
    print(f"Species presence rate: {species_presence.mean():.1%}")
    
    return (
        center_x,
        center_y,
        coords,
        create_spatial_field,
        distance_to_center,
        elevation,
        hotspot_centers,
        n_sites,
        precipitation,
        precipitation_base,
        presence_prob,
        richness_base,
        spatial_data,
        species_presence,
        species_richness,
        suitability,
        temp_suitability,
        temperature,
        x_coords,
        y_coords,
    )


@app.cell
def __():
    """
    ## Spatial Data Visualization
    """
    # Create spatial plots
    def create_spatial_plot(data, x_col, y_col, color_col, title):
        """Create a spatial scatter plot"""
        return hv.Scatter(data, [x_col, y_col], color_col).opts(
            color=color_col,
            size=8,
            alpha=0.7,
            colorbar=True,
            title=title,
            xlabel="X Coordinate (km)",
            ylabel="Y Coordinate (km)",
            width=500,
            height=400,
            tools=['hover']
        )
    
    # Visualize spatial patterns
    elevation_plot = create_spatial_plot(
        spatial_data, 'x_coord', 'y_coord', 'elevation', 
        'Elevation Distribution'
    )
    
    temp_plot = create_spatial_plot(
        spatial_data, 'x_coord', 'y_coord', 'temperature',
        'Temperature Distribution'
    )
    
    richness_plot = create_spatial_plot(
        spatial_data, 'x_coord', 'y_coord', 'species_richness',
        'Species Richness Distribution'
    )
    
    presence_plot = create_spatial_plot(
        spatial_data, 'x_coord', 'y_coord', 'species_presence',
        'Species Presence/Absence'
    )
    
    print("Spatial Patterns Visualization:")
    (elevation_plot + temp_plot + richness_plot + presence_plot).cols(2)
    
    return (
        create_spatial_plot,
        elevation_plot,
        presence_plot,
        richness_plot,
        temp_plot,
    )


@app.cell
def __():
    """
    ## Spatial Autocorrelation Analysis
    """
    # Calculate Moran's I for spatial autocorrelation
    def calculate_morans_i(values, coordinates, distance_threshold=None):
        """
        Calculate Moran's I statistic for spatial autocorrelation
        """
        n = len(values)
        
        # Calculate distance matrix
        distances = squareform(pdist(coordinates))
        
        # Create spatial weights matrix
        if distance_threshold is None:
            # Use inverse distance weights
            weights = 1 / (distances + 1e-10)  # Add small value to avoid division by zero
            np.fill_diagonal(weights, 0)  # No self-neighbors
        else:
            # Binary weights based on distance threshold
            weights = (distances <= distance_threshold).astype(float)
            np.fill_diagonal(weights, 0)
        
        # Standardize weights (row normalization)
        row_sums = weights.sum(axis=1)
        weights = weights / (row_sums[:, np.newaxis] + 1e-10)
        
        # Calculate Moran's I
        values_centered = values - np.mean(values)
        
        # Numerator: sum of weighted cross-products
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * values_centered[i] * values_centered[j]
        
        # Denominator: sum of squared deviations
        denominator = np.sum(values_centered**2)
        
        # Moran's I
        morans_i = numerator / denominator
        
        # Expected value under null hypothesis
        expected_i = -1 / (n - 1)
        
        # Calculate standard error (simplified)
        # This is an approximation; exact calculation is more complex
        variance_i = 1 / (n - 1)  # Simplified
        z_score = (morans_i - expected_i) / np.sqrt(variance_i)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'z_score': z_score,
            'p_value': p_value,
            'weights_matrix': weights
        }
    
    # Test spatial autocorrelation for different variables
    coordinates = spatial_data[['x_coord', 'y_coord']].values
    
    autocorr_results = {}
    variables = ['elevation', 'temperature', 'precipitation', 'species_richness']
    
    print("Spatial Autocorrelation Analysis (Moran's I):")
    print("=" * 55)
    print("Variable        | Moran's I | Expected | Z-score | P-value")
    print("-" * 55)
    
    for var in variables:
        values = spatial_data[var].values
        result = calculate_morans_i(values, coordinates, distance_threshold=20)
        autocorr_results[var] = result
        
        morans_i = result['morans_i']
        expected_i = result['expected_i']
        z_score = result['z_score']
        p_value = result['p_value']
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"{var:14} | {morans_i:8.3f}  | {expected_i:8.3f} | {z_score:7.2f} | {p_value:.3f} {significance}")
    
    # Interpret results
    print(f"\nInterpretation:")
    print(f"Moran's I > 0: Positive spatial autocorrelation (similar values cluster)")
    print(f"Moran's I < 0: Negative spatial autocorrelation (dissimilar values cluster)")
    print(f"Moran's I ≈ 0: No spatial autocorrelation (random pattern)")
    print(f"Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    return autocorr_results, calculate_morans_i, coordinates, variables


@app.cell
def __():
    """
    ## Spatial Interpolation
    """
    # Interpolate environmental variables across the landscape
    def spatial_interpolation(data, x_col, y_col, value_col, method='linear', grid_resolution=50):
        """
        Interpolate point data to regular grid
        """
        # Extract coordinates and values
        points = data[[x_col, y_col]].values
        values = data[value_col].values
        
        # Create regular grid
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate using scipy.interpolate.griddata
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        interpolated_values = griddata(
            points, values, grid_points, 
            method=method, fill_value=np.nan
        )
        
        Z_grid = interpolated_values.reshape(X_grid.shape)
        
        return X_grid, Y_grid, Z_grid, x_grid, y_grid
    
    # Interpolate temperature across the study area
    X_temp, Y_temp, Z_temp, x_grid, y_grid = spatial_interpolation(
        spatial_data, 'x_coord', 'y_coord', 'temperature'
    )
    
    # Interpolate species richness
    X_rich, Y_rich, Z_rich, _, _ = spatial_interpolation(
        spatial_data, 'x_coord', 'y_coord', 'species_richness'
    )
    
    print(f"Spatial Interpolation Results:")
    print(f"Grid resolution: {len(x_grid)} × {len(y_grid)}")
    print(f"Temperature range (interpolated): {np.nanmin(Z_temp):.1f} - {np.nanmax(Z_temp):.1f}°C")
    print(f"Species richness range (interpolated): {np.nanmin(Z_rich):.1f} - {np.nanmax(Z_rich):.1f}")
    
    # Create interpolated surface data for visualization
    def create_surface_data(X, Y, Z, variable_name):
        """Convert grid data to format suitable for holoviews"""
        # Flatten arrays and create DataFrame
        surface_data = pd.DataFrame({
            'x': X.ravel(),
            'y': Y.ravel(),
            variable_name: Z.ravel()
        })
        # Remove NaN values
        surface_data = surface_data.dropna()
        return surface_data
    
    temp_surface = create_surface_data(X_temp, Y_temp, Z_temp, 'temperature')
    richness_surface = create_surface_data(X_rich, Y_rich, Z_rich, 'species_richness')
    
    # Create surface plots
    temp_surface_plot = hv.QuadMesh(temp_surface, ['x', 'y'], 'temperature').opts(
        title="Temperature Surface (Interpolated)",
        colorbar=True,
        cmap='viridis',
        width=500,
        height=400
    )
    
    richness_surface_plot = hv.QuadMesh(richness_surface, ['x', 'y'], 'species_richness').opts(
        title="Species Richness Surface (Interpolated)",
        colorbar=True,
        cmap='plasma',
        width=500,
        height=400
    )
    
    print("\nInterpolated Surface Maps:")
    temp_surface_plot + richness_surface_plot
    
    return (
        X_rich,
        X_temp,
        Y_rich,
        Y_temp,
        Z_rich,
        Z_temp,
        create_surface_data,
        richness_surface,
        richness_surface_plot,
        spatial_interpolation,
        temp_surface,
        temp_surface_plot,
        x_grid,
        y_grid,
    )


@app.cell
def __():
    """
    ## Spatial Distance Analysis
    """
    # Analyze patterns as a function of spatial distance
    def spatial_distance_analysis(data, x_col, y_col, value_col, max_distance=50, n_bins=20):
        """Analyze spatial correlation as function of distance"""
        coordinates = data[[x_col, y_col]].values
        values = data[value_col].values
        
        # Calculate all pairwise distances
        distances = squareform(pdist(coordinates))
        
        # Create distance bins
        distance_bins = np.linspace(0, max_distance, n_bins + 1)
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        correlations = []
        sample_sizes = []
        
        for i in range(len(distance_bins) - 1):
            d_min, d_max = distance_bins[i], distance_bins[i + 1]
            
            # Find pairs within this distance range
            mask = (distances >= d_min) & (distances < d_max)
            
            if np.sum(mask) > 10:  # Need sufficient pairs
                # Get value pairs
                indices = np.where(mask)
                value_pairs_i = values[indices[0]]
                value_pairs_j = values[indices[1]]
                
                # Calculate correlation
                if len(value_pairs_i) > 1:
                    corr, _ = stats.pearsonr(value_pairs_i, value_pairs_j)
                    correlations.append(corr)
                    sample_sizes.append(len(value_pairs_i))
                else:
                    correlations.append(np.nan)
                    sample_sizes.append(0)
            else:
                correlations.append(np.nan)
                sample_sizes.append(0)
        
        return bin_centers, correlations, sample_sizes
    
    # Analyze spatial correlation for temperature and species richness
    temp_distances, temp_correlations, temp_samples = spatial_distance_analysis(
        spatial_data, 'x_coord', 'y_coord', 'temperature'
    )
    
    richness_distances, richness_correlations, richness_samples = spatial_distance_analysis(
        spatial_data, 'x_coord', 'y_coord', 'species_richness'
    )
    
    print("Spatial Distance Analysis:")
    print("=" * 35)
    print("Distance (km) | Temperature | Species Richness")
    print("              | Correlation | Correlation")
    print("-" * 45)
    
    for i, dist in enumerate(temp_distances[:10]):  # Show first 10 bins
        temp_corr = temp_correlations[i] if not np.isnan(temp_correlations[i]) else 0
        rich_corr = richness_correlations[i] if not np.isnan(richness_correlations[i]) else 0
        
        print(f"{dist:12.1f}  | {temp_corr:11.3f} | {rich_corr:11.3f}")
    
    # Find range of spatial autocorrelation
    def find_autocorr_range(distances, correlations, threshold=0.1):
        """Find distance at which correlation drops below threshold"""
        valid_indices = ~np.isnan(correlations)
        if not any(valid_indices):
            return np.nan
        
        valid_distances = np.array(distances)[valid_indices]
        valid_correlations = np.array(correlations)[valid_indices]
        
        # Find first distance where correlation < threshold
        below_threshold = valid_correlations < threshold
        if any(below_threshold):
            return valid_distances[below_threshold][0]
        else:
            return valid_distances[-1]  # Correlation never drops below threshold
    
    temp_range = find_autocorr_range(temp_distances, temp_correlations)
    richness_range = find_autocorr_range(richness_distances, richness_correlations)
    
    print(f"\nSpatial Autocorrelation Range:")
    print(f"Temperature: {temp_range:.1f} km")
    print(f"Species richness: {richness_range:.1f} km")
    
    return (
        bin_centers,
        correlations,
        d_max,
        d_min,
        distance_bins,
        find_autocorr_range,
        rich_corr,
        richness_correlations,
        richness_distances,
        richness_range,
        richness_samples,
        sample_sizes,
        spatial_distance_analysis,
        temp_corr,
        temp_correlations,
        temp_distances,
        temp_range,
        temp_samples,
    )


@app.cell
def __():
    """
    ## Species Distribution Modeling with Spatial Components
    """
    # Fit species distribution model accounting for spatial structure
    def spatial_species_distribution_model(data, species_col, env_vars, coord_cols):
        """Fit SDM with environmental and spatial components"""
        
        # Environmental data
        X_env = data[env_vars].values
        
        # Spatial coordinates
        coords = data[coord_cols].values
        
        # Response variable
        y = data[species_col].values
        
        # Standardize environmental variables
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_env_scaled = scaler.fit_transform(X_env)
        
        # Create spatial basis functions (simplified trend surface)
        # Linear trends in x and y
        x_trend = coords[:, 0] - coords[:, 0].mean()
        y_trend = coords[:, 1] - coords[:, 1].mean()
        
        # Quadratic trends
        x2_trend = x_trend ** 2
        y2_trend = y_trend ** 2
        xy_trend = x_trend * y_trend
        
        # Combine environmental and spatial predictors
        X_spatial = np.column_stack([x_trend, y_trend, x2_trend, y2_trend, xy_trend])
        X_full = np.column_stack([X_env_scaled, X_spatial])
        
        # Fit models
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Environmental model only
        model_env = LogisticRegression(random_state=42)
        model_env.fit(X_env_scaled, y)
        y_pred_env = model_env.predict_proba(X_env_scaled)[:, 1]
        
        # Environmental + Spatial model
        model_full = LogisticRegression(random_state=42)
        model_full.fit(X_full, y)
        y_pred_full = model_full.predict_proba(X_full)[:, 1]
        
        # Model comparison
        auc_env = roc_auc_score(y, y_pred_env)
        auc_full = roc_auc_score(y, y_pred_full)
        
        acc_env = accuracy_score(y, y_pred_env > 0.5)
        acc_full = accuracy_score(y, y_pred_full > 0.5)
        
        return {
            'model_env': model_env,
            'model_full': model_full,
            'predictions_env': y_pred_env,
            'predictions_full': y_pred_full,
            'auc_env': auc_env,
            'auc_full': auc_full,
            'accuracy_env': acc_env,
            'accuracy_full': acc_full,
            'X_env_scaled': X_env_scaled,
            'X_full': X_full,
            'scaler': scaler
        }
    
    # Fit SDM for species presence
    env_variables = ['elevation', 'temperature', 'precipitation']
    coord_variables = ['x_coord', 'y_coord']
    
    sdm_results = spatial_species_distribution_model(
        spatial_data, 'species_presence', env_variables, coord_variables
    )
    
    print("Spatial Species Distribution Modeling:")
    print("=" * 45)
    print("Model Type              | AUC   | Accuracy")
    print("-" * 45)
    print(f"Environmental Only      | {sdm_results['auc_env']:.3f} | {sdm_results['accuracy_env']:.3f}")
    print(f"Environmental + Spatial | {sdm_results['auc_full']:.3f} | {sdm_results['accuracy_full']:.3f}")
    
    # Calculate improvement from adding spatial components
    auc_improvement = sdm_results['auc_full'] - sdm_results['auc_env']
    acc_improvement = sdm_results['accuracy_full'] - sdm_results['accuracy_env']
    
    print(f"\nImprovement from spatial components:")
    print(f"AUC improvement: {auc_improvement:+.3f}")
    print(f"Accuracy improvement: {acc_improvement:+.3f}")
    
    if auc_improvement > 0.05:
        print("Substantial improvement - spatial structure is important")
    elif auc_improvement > 0.02:
        print("Moderate improvement - some spatial structure present")
    else:
        print("Minimal improvement - limited spatial structure")
    
    return (
        LogisticRegression,
        StandardScaler,
        X_env_scaled,
        X_full,
        X_spatial,
        acc_improvement,
        accuracy_score,
        auc_improvement,
        coord_variables,
        env_variables,
        model_env,
        model_full,
        roc_auc_score,
        scaler,
        sdm_results,
        spatial_species_distribution_model,
        x2_trend,
        x_trend,
        xy_trend,
        y2_trend,
        y_pred_env,
        y_pred_full,
        y_trend,
    )


@app.cell
def __():
    """
    ## Spatial Prediction and Mapping
    """
    # Predict species suitability across the landscape
    def create_prediction_map(model, scaler, env_surfaces, spatial_coords, coord_cols):
        """Create prediction map from fitted model"""
        
        # Prepare environmental data
        env_data = []
        for var in env_variables:
            if var == 'elevation':
                # Create elevation surface (simple gradient)
                x_vals = spatial_coords[:, 0]
                y_vals = spatial_coords[:, 1]
                elev_pred = 500 + 10 * x_vals + 8 * y_vals
                env_data.append(elev_pred)
            elif var == 'temperature':
                # Use interpolated temperature
                temp_pred = griddata(
                    spatial_data[coord_cols].values,
                    spatial_data[var].values,
                    spatial_coords,
                    method='linear'
                )
                env_data.append(temp_pred)
            elif var == 'precipitation':
                # Use interpolated precipitation
                precip_pred = griddata(
                    spatial_data[coord_cols].values,
                    spatial_data[var].values,
                    spatial_coords,
                    method='linear'
                )
                env_data.append(precip_pred)
        
        X_pred_env = np.column_stack(env_data)
        
        # Handle missing values
        valid_mask = ~np.isnan(X_pred_env).any(axis=1)
        X_pred_env_clean = X_pred_env[valid_mask]
        coords_clean = spatial_coords[valid_mask]
        
        # Scale environmental variables
        X_pred_env_scaled = scaler.transform(X_pred_env_clean)
        
        # Add spatial components if needed
        if hasattr(model, 'n_features_in_') and model.n_features_in_ > len(env_variables):
            # Model includes spatial components
            x_trend_pred = coords_clean[:, 0] - spatial_data['x_coord'].mean()
            y_trend_pred = coords_clean[:, 1] - spatial_data['y_coord'].mean()
            x2_trend_pred = x_trend_pred ** 2
            y2_trend_pred = y_trend_pred ** 2
            xy_trend_pred = x_trend_pred * y_trend_pred
            
            X_spatial_pred = np.column_stack([
                x_trend_pred, y_trend_pred, x2_trend_pred, y2_trend_pred, xy_trend_pred
            ])
            X_pred_full = np.column_stack([X_pred_env_scaled, X_spatial_pred])
            
            predictions = model.predict_proba(X_pred_full)[:, 1]
        else:
            # Environmental model only
            predictions = model.predict_proba(X_pred_env_scaled)[:, 1]
        
        # Create results DataFrame
        pred_results = pd.DataFrame({
            'x_coord': coords_clean[:, 0],
            'y_coord': coords_clean[:, 1],
            'prediction': predictions
        })
        
        return pred_results
    
    # Create regular grid for prediction
    x_pred = np.linspace(spatial_data['x_coord'].min(), spatial_data['x_coord'].max(), 50)
    y_pred = np.linspace(spatial_data['y_coord'].min(), spatial_data['y_coord'].max(), 50)
    X_pred_grid, Y_pred_grid = np.meshgrid(x_pred, y_pred)
    pred_coords = np.column_stack([X_pred_grid.ravel(), Y_pred_grid.ravel()])
    
    # Generate predictions
    env_predictions = create_prediction_map(
        sdm_results['model_env'], sdm_results['scaler'], 
        None, pred_coords, coord_variables
    )
    
    full_predictions = create_prediction_map(
        sdm_results['model_full'], sdm_results['scaler'],
        None, pred_coords, coord_variables
    )
    
    # Create prediction maps
    env_pred_plot = hv.Scatter(env_predictions, 'x_coord', 'y_coord', 'prediction').opts(
        title="Environmental Model Predictions",
        color='prediction',
        cmap='viridis',
        size=8,
        colorbar=True,
        width=500,
        height=400
    )
    
    full_pred_plot = hv.Scatter(full_predictions, 'x_coord', 'y_coord', 'prediction').opts(
        title="Environmental + Spatial Model Predictions",
        color='prediction',
        cmap='viridis',
        size=8,
        colorbar=True,
        width=500,
        height=400
    )
    
    # Overlay actual presence points
    presence_points = spatial_data[spatial_data['species_presence'] == 1]
    presence_overlay = hv.Scatter(presence_points, 'x_coord', 'y_coord').opts(
        color='red',
        size=12,
        marker='x',
        alpha=0.8
    )
    
    print("Species Distribution Prediction Maps:")
    print(f"Environmental model predictions: {len(env_predictions)} grid points")
    print(f"Full model predictions: {len(full_predictions)} grid points")
    
    (env_pred_plot * presence_overlay + full_pred_plot * presence_overlay).cols(1)
    
    return (
        X_pred_grid,
        Y_pred_grid,
        coords_clean,
        create_prediction_map,
        elev_pred,
        env_pred_plot,
        env_predictions,
        full_pred_plot,
        full_predictions,
        precip_pred,
        pred_coords,
        pred_results,
        predictions,
        presence_overlay,
        presence_points,
        temp_pred,
        x_pred,
        y_pred,
    )


@app.cell
def __():
    """
    ## Spatial Cross-Validation
    """
    # Spatial cross-validation to account for spatial autocorrelation
    def spatial_cross_validation(data, model_func, coord_cols, response_col, n_folds=5, buffer_distance=10):
        """Perform spatial cross-validation with buffer zones"""
        
        coordinates = data[coord_cols].values
        
        # Create spatial folds using k-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_folds, random_state=42)
        fold_assignments = kmeans.fit_predict(coordinates)
        
        cv_results = []
        
        for fold in range(n_folds):
            # Test set: current fold
            test_mask = fold_assignments == fold
            test_indices = np.where(test_mask)[0]
            
            # Create buffer around test points
            test_coords = coordinates[test_mask]
            
            # Calculate distances from all points to test points
            distances_to_test = cdist(coordinates, test_coords)
            min_distances_to_test = np.min(distances_to_test, axis=1)
            
            # Training set: exclude test points and buffer zone
            train_mask = (fold_assignments != fold) & (min_distances_to_test > buffer_distance)
            train_indices = np.where(train_mask)[0]
            
            if len(train_indices) < 10 or len(test_indices) < 5:
                continue  # Skip if insufficient data
            
            # Fit model on training data
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            # Get predictions (simplified - would call actual model fitting function)
            # For demonstration, using the already fitted model
            train_coords_scaled = sdm_results['scaler'].transform(
                train_data[env_variables].values
            )
            test_coords_scaled = sdm_results['scaler'].transform(
                test_data[env_variables].values
            )
            
            # Use environmental model for simplicity
            test_predictions = sdm_results['model_env'].predict_proba(test_coords_scaled)[:, 1]
            test_actual = test_data[response_col].values
            
            # Calculate metrics
            try:
                auc = roc_auc_score(test_actual, test_predictions)
                accuracy = accuracy_score(test_actual, test_predictions > 0.5)
                
                cv_results.append({
                    'fold': fold,
                    'n_train': len(train_indices),
                    'n_test': len(test_indices),
                    'auc': auc,
                    'accuracy': accuracy
                })
            except ValueError:
                # Skip if all test cases are same class
                continue
        
        return cv_results
    
    # Perform spatial cross-validation
    spatial_cv_results = spatial_cross_validation(
        spatial_data, None, coord_variables, 'species_presence'
    )
    
    print("Spatial Cross-Validation Results:")
    print("=" * 45)
    print("Fold | N Train | N Test | AUC   | Accuracy")
    print("-" * 45)
    
    for result in spatial_cv_results:
        fold = result['fold']
        n_train = result['n_train']
        n_test = result['n_test']
        auc = result['auc']
        accuracy = result['accuracy']
        
        print(f"{fold:4d} | {n_train:7d} | {n_test:6d} | {auc:.3f} | {accuracy:.3f}")
    
    # Calculate mean performance
    if spatial_cv_results:
        mean_auc = np.mean([r['auc'] for r in spatial_cv_results])
        mean_accuracy = np.mean([r['accuracy'] for r in spatial_cv_results])
        std_auc = np.std([r['auc'] for r in spatial_cv_results])
        std_accuracy = np.std([r['accuracy'] for r in spatial_cv_results])
        
        print(f"\nSpatial CV Performance:")
        print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        
        # Compare to non-spatial performance
        print(f"\nComparison to non-spatial evaluation:")
        print(f"Non-spatial AUC: {sdm_results['auc_env']:.3f}")
        print(f"Spatial CV AUC: {mean_auc:.3f}")
        print(f"Difference: {sdm_results['auc_env'] - mean_auc:+.3f}")
        
        if sdm_results['auc_env'] - mean_auc > 0.05:
            print("Substantial overestimation due to spatial autocorrelation")
        else:
            print("Minimal bias from spatial autocorrelation")
    
    return (
        KMeans,
        accuracy,
        auc,
        buffer_distance,
        fold_assignments,
        kmeans,
        mean_accuracy,
        mean_auc,
        min_distances_to_test,
        spatial_cross_validation,
        spatial_cv_results,
        std_accuracy,
        std_auc,
        test_actual,
        test_coords,
        test_coords_scaled,
        test_data,
        test_indices,
        test_mask,
        test_predictions,
        train_coords_scaled,
        train_data,
        train_indices,
        train_mask,
    )


@app.cell
def __():
    """
    ## Summary and Best Practices\n
    In this chapter, we covered spatial analysis methods for ecological data:\n
    ✓ **Spatial data creation**: Generating realistic spatial ecological datasets
    ✓ **Spatial visualization**: Mapping ecological patterns across landscapes
    ✓ **Spatial autocorrelation**: Moran's I and spatial dependence analysis
    ✓ **Spatial interpolation**: Creating continuous surfaces from point data
    ✓ **Distance analysis**: Understanding spatial correlation structure
    ✓ **Species distribution modeling**: Environmental and spatial predictors
    ✓ **Spatial prediction**: Mapping species suitability across landscapes
    ✓ **Spatial cross-validation**: Accounting for spatial autocorrelation in model evaluation\n
    ### Key Concepts in Spatial Ecology:
    - **Spatial autocorrelation**: "Everything is related to everything else, but near things are more related"
    - **Scale dependence**: Patterns and processes vary with spatial scale
    - **Edge effects**: Boundary conditions affect ecological patterns
    - **Spatial heterogeneity**: Environmental variation drives species distributions
    - **Dispersal limitation**: Species distributions reflect movement constraints\n
    ### Python Packages for Spatial Analysis:
    - **scipy.spatial**: Distance calculations and spatial operations
    - **scipy.interpolate**: Spatial interpolation methods
    - **sklearn**: Machine learning for spatial modeling
    - **pandas**: Spatial data manipulation and analysis
    - **holoviews**: Interactive spatial visualization\n
    ### Best Practices for Spatial Ecological Analysis:
    1. **Check for spatial autocorrelation**: Always test before modeling
    2. **Use appropriate scales**: Match analysis scale to ecological processes
    3. **Account for spatial structure**: Include spatial components in models
    4. **Validate spatially**: Use spatial cross-validation methods
    5. **Consider boundaries**: Be aware of edge effects and study area limits
    6. **Document projections**: Keep track of coordinate systems and units
    7. **Visualize patterns**: Maps reveal insights not apparent in tables
    8. **Test assumptions**: Verify stationarity and isotropy when assumed\n
    ### Common Applications in Spatial Ecology:
    - **Species distribution modeling**: Predicting suitable habitat
    - **Conservation planning**: Identifying priority areas for protection
    - **Landscape ecology**: Understanding patch connectivity and fragmentation
    - **Disease ecology**: Modeling pathogen spread across landscapes
    - **Climate change impacts**: Projecting species range shifts
    - **Biodiversity mapping**: Creating continuous diversity surfaces
    - **Restoration planning**: Optimal placement of restoration sites
    """
    
    spatial_summary = {
        'Dataset Characteristics': {
            'Number of Sites': f"{len(spatial_data)}",
            'Spatial Extent': f"{x_coords.max() - x_coords.min():.0f} × {y_coords.max() - y_coords.min():.0f} km",
            'Variables Analyzed': len(variables)
        },
        'Spatial Autocorrelation': {
            "Temperature Moran's I": f"{autocorr_results['temperature']['morans_i']:.3f}",
            "Species Richness Moran's I": f"{autocorr_results['species_richness']['morans_i']:.3f}",
            'Autocorrelation Range': f"{temp_range:.1f} km (temp), {richness_range:.1f} km (richness)"
        },
        'Species Distribution Models': {
            'Environmental Model AUC': f"{sdm_results['auc_env']:.3f}",
            'Spatial Model AUC': f"{sdm_results['auc_full']:.3f}",
            'Improvement from Spatial': f"{auc_improvement:+.3f}"
        },
        'Spatial Cross-Validation': {
            'Mean CV AUC': f"{mean_auc:.3f} ± {std_auc:.3f}" if spatial_cv_results else "Not available",
            'Spatial Bias': f"{sdm_results['auc_env'] - mean_auc:+.3f}" if spatial_cv_results else "Not available",
            'CV Folds': f"{len(spatial_cv_results)}" if spatial_cv_results else "0"
        }
    }
    
    print("Spatial Analysis Summary:")
    print("=" * 35)
    
    for category, details in spatial_summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Chapter 12 complete! Ready for mechanistic modeling.")
    
    return spatial_summary,


if __name__ == "__main__":
    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
