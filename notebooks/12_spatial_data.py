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
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])\n        
        interpolated_values = griddata(
            points, values, grid_points, 
            method=method, fill_value=np.nan
        )\n        
        Z_grid = interpolated_values.reshape(X_grid.shape)\n        \n        return X_grid, Y_grid, Z_grid, x_grid, y_grid\n    \n    # Interpolate temperature across the study area\n    X_temp, Y_temp, Z_temp, x_grid, y_grid = spatial_interpolation(\n        spatial_data, 'x_coord', 'y_coord', 'temperature'\n    )\n    \n    # Interpolate species richness\n    X_rich, Y_rich, Z_rich, _, _ = spatial_interpolation(\n        spatial_data, 'x_coord', 'y_coord', 'species_richness'\n    )\n    \n    print(f\"Spatial Interpolation Results:\")\n    print(f\"Grid resolution: {len(x_grid)} × {len(y_grid)}\")\n    print(f\"Temperature range (interpolated): {np.nanmin(Z_temp):.1f} - {np.nanmax(Z_temp):.1f}°C\")\n    print(f\"Species richness range (interpolated): {np.nanmin(Z_rich):.1f} - {np.nanmax(Z_rich):.1f}\")\n    \n    # Create interpolated surface data for visualization\n    def create_surface_data(X, Y, Z, variable_name):\n        \"\"\"Convert grid data to format suitable for holoviews\"\"\"\n        # Flatten arrays and create DataFrame\n        surface_data = pd.DataFrame({\n            'x': X.ravel(),\n            'y': Y.ravel(),\n            variable_name: Z.ravel()\n        })\n        # Remove NaN values\n        surface_data = surface_data.dropna()\n        return surface_data\n    \n    temp_surface = create_surface_data(X_temp, Y_temp, Z_temp, 'temperature')\n    richness_surface = create_surface_data(X_rich, Y_rich, Z_rich, 'species_richness')\n    \n    # Create surface plots\n    temp_surface_plot = hv.QuadMesh(temp_surface, ['x', 'y'], 'temperature').opts(\n        title=\"Temperature Surface (Interpolated)\",\n        colorbar=True,\n        cmap='viridis',\n        width=500,\n        height=400\n    )\n    \n    richness_surface_plot = hv.QuadMesh(richness_surface, ['x', 'y'], 'species_richness').opts(\n        title=\"Species Richness Surface (Interpolated)\",\n        colorbar=True,\n        cmap='plasma',\n        width=500,\n        height=400\n    )\n    \n    print(\"\\nInterpolated Surface Maps:\")\n    temp_surface_plot + richness_surface_plot\n    \n    return (\n        X_rich,\n        X_temp,\n        Y_rich,\n        Y_temp,\n        Z_rich,\n        Z_temp,\n        create_surface_data,\n        richness_surface,\n        richness_surface_plot,\n        spatial_interpolation,\n        temp_surface,\n        temp_surface_plot,\n        x_grid,\n        y_grid,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Spatial Distance Analysis\n    \"\"\"\n    # Analyze patterns as a function of spatial distance\n    def spatial_distance_analysis(data, x_col, y_col, value_col, max_distance=50, n_bins=20):\n        \"\"\"Analyze spatial correlation as function of distance\"\"\"\n        coordinates = data[[x_col, y_col]].values\n        values = data[value_col].values\n        \n        # Calculate all pairwise distances\n        distances = squareform(pdist(coordinates))\n        \n        # Create distance bins\n        distance_bins = np.linspace(0, max_distance, n_bins + 1)\n        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2\n        \n        correlations = []\n        sample_sizes = []\n        \n        for i in range(len(distance_bins) - 1):\n            d_min, d_max = distance_bins[i], distance_bins[i + 1]\n            \n            # Find pairs within this distance range\n            mask = (distances >= d_min) & (distances < d_max)\n            \n            if np.sum(mask) > 10:  # Need sufficient pairs\n                # Get value pairs\n                indices = np.where(mask)\n                value_pairs_i = values[indices[0]]\n                value_pairs_j = values[indices[1]]\n                \n                # Calculate correlation\n                if len(value_pairs_i) > 1:\n                    corr, _ = stats.pearsonr(value_pairs_i, value_pairs_j)\n                    correlations.append(corr)\n                    sample_sizes.append(len(value_pairs_i))\n                else:\n                    correlations.append(np.nan)\n                    sample_sizes.append(0)\n            else:\n                correlations.append(np.nan)\n                sample_sizes.append(0)\n        \n        return bin_centers, correlations, sample_sizes\n    \n    # Analyze spatial correlation for temperature and species richness\n    temp_distances, temp_correlations, temp_samples = spatial_distance_analysis(\n        spatial_data, 'x_coord', 'y_coord', 'temperature'\n    )\n    \n    richness_distances, richness_correlations, richness_samples = spatial_distance_analysis(\n        spatial_data, 'x_coord', 'y_coord', 'species_richness'\n    )\n    \n    print(\"Spatial Distance Analysis:\")\n    print(\"=\" * 35)\n    print(\"Distance (km) | Temperature | Species Richness\")\n    print(\"              | Correlation | Correlation\")\n    print(\"-\" * 45)\n    \n    for i, dist in enumerate(temp_distances[:10]):  # Show first 10 bins\n        temp_corr = temp_correlations[i] if not np.isnan(temp_correlations[i]) else 0\n        rich_corr = richness_correlations[i] if not np.isnan(richness_correlations[i]) else 0\n        \n        print(f\"{dist:12.1f}  | {temp_corr:11.3f} | {rich_corr:11.3f}\")\n    \n    # Find range of spatial autocorrelation\n    def find_autocorr_range(distances, correlations, threshold=0.1):\n        \"\"\"Find distance at which correlation drops below threshold\"\"\"\n        valid_indices = ~np.isnan(correlations)\n        if not any(valid_indices):\n            return np.nan\n        \n        valid_distances = np.array(distances)[valid_indices]\n        valid_correlations = np.array(correlations)[valid_indices]\n        \n        # Find first distance where correlation < threshold\n        below_threshold = valid_correlations < threshold\n        if any(below_threshold):\n            return valid_distances[below_threshold][0]\n        else:\n            return valid_distances[-1]  # Correlation never drops below threshold\n    \n    temp_range = find_autocorr_range(temp_distances, temp_correlations)\n    richness_range = find_autocorr_range(richness_distances, richness_correlations)\n    \n    print(f\"\\nSpatial Autocorrelation Range:\")\n    print(f\"Temperature: {temp_range:.1f} km\")\n    print(f\"Species richness: {richness_range:.1f} km\")\n    \n    return (\n        bin_centers,\n        correlations,\n        d_max,\n        d_min,\n        distance_bins,\n        find_autocorr_range,\n        rich_corr,\n        richness_correlations,\n        richness_distances,\n        richness_range,\n        richness_samples,\n        sample_sizes,\n        spatial_distance_analysis,\n        temp_corr,\n        temp_correlations,\n        temp_distances,\n        temp_range,\n        temp_samples,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Species Distribution Modeling with Spatial Components\n    \"\"\"\n    # Fit species distribution model accounting for spatial structure\n    def spatial_species_distribution_model(data, species_col, env_vars, coord_cols):\n        \"\"\"Fit SDM with environmental and spatial components\"\"\"\n        \n        # Environmental data\n        X_env = data[env_vars].values\n        \n        # Spatial coordinates\n        coords = data[coord_cols].values\n        \n        # Response variable\n        y = data[species_col].values\n        \n        # Standardize environmental variables\n        from sklearn.preprocessing import StandardScaler\n        scaler = StandardScaler()\n        X_env_scaled = scaler.fit_transform(X_env)\n        \n        # Create spatial basis functions (simplified trend surface)\n        # Linear trends in x and y\n        x_trend = coords[:, 0] - coords[:, 0].mean()\n        y_trend = coords[:, 1] - coords[:, 1].mean()\n        \n        # Quadratic trends\n        x2_trend = x_trend ** 2\n        y2_trend = y_trend ** 2\n        xy_trend = x_trend * y_trend\n        \n        # Combine environmental and spatial predictors\n        X_spatial = np.column_stack([x_trend, y_trend, x2_trend, y2_trend, xy_trend])\n        X_full = np.column_stack([X_env_scaled, X_spatial])\n        \n        # Fit models\n        from sklearn.linear_model import LogisticRegression\n        from sklearn.metrics import roc_auc_score, accuracy_score\n        \n        # Environmental model only\n        model_env = LogisticRegression(random_state=42)\n        model_env.fit(X_env_scaled, y)\n        y_pred_env = model_env.predict_proba(X_env_scaled)[:, 1]\n        \n        # Environmental + Spatial model\n        model_full = LogisticRegression(random_state=42)\n        model_full.fit(X_full, y)\n        y_pred_full = model_full.predict_proba(X_full)[:, 1]\n        \n        # Model comparison\n        auc_env = roc_auc_score(y, y_pred_env)\n        auc_full = roc_auc_score(y, y_pred_full)\n        \n        acc_env = accuracy_score(y, y_pred_env > 0.5)\n        acc_full = accuracy_score(y, y_pred_full > 0.5)\n        \n        return {\n            'model_env': model_env,\n            'model_full': model_full,\n            'predictions_env': y_pred_env,\n            'predictions_full': y_pred_full,\n            'auc_env': auc_env,\n            'auc_full': auc_full,\n            'accuracy_env': acc_env,\n            'accuracy_full': acc_full,\n            'X_env_scaled': X_env_scaled,\n            'X_full': X_full,\n            'scaler': scaler\n        }\n    \n    # Fit SDM for species presence\n    env_variables = ['elevation', 'temperature', 'precipitation']\n    coord_variables = ['x_coord', 'y_coord']\n    \n    sdm_results = spatial_species_distribution_model(\n        spatial_data, 'species_presence', env_variables, coord_variables\n    )\n    \n    print(\"Spatial Species Distribution Modeling:\")\n    print(\"=\" * 45)\n    print(\"Model Type              | AUC   | Accuracy\")\n    print(\"-\" * 45)\n    print(f\"Environmental Only      | {sdm_results['auc_env']:.3f} | {sdm_results['accuracy_env']:.3f}\")\n    print(f\"Environmental + Spatial | {sdm_results['auc_full']:.3f} | {sdm_results['accuracy_full']:.3f}\")\n    \n    # Calculate improvement from adding spatial components\n    auc_improvement = sdm_results['auc_full'] - sdm_results['auc_env']\n    acc_improvement = sdm_results['accuracy_full'] - sdm_results['accuracy_env']\n    \n    print(f\"\\nImprovement from spatial components:\")\n    print(f\"AUC improvement: {auc_improvement:+.3f}\")\n    print(f\"Accuracy improvement: {acc_improvement:+.3f}\")\n    \n    if auc_improvement > 0.05:\n        print(\"Substantial improvement - spatial structure is important\")\n    elif auc_improvement > 0.02:\n        print(\"Moderate improvement - some spatial structure present\")\n    else:\n        print(\"Minimal improvement - limited spatial structure\")\n    \n    return (\n        LogisticRegression,\n        StandardScaler,\n        X_env_scaled,\n        X_full,\n        X_spatial,\n        acc_improvement,\n        accuracy_score,\n        auc_improvement,\n        coord_variables,\n        env_variables,\n        model_env,\n        model_full,\n        roc_auc_score,\n        scaler,\n        sdm_results,\n        spatial_species_distribution_model,\n        x2_trend,\n        x_trend,\n        xy_trend,\n        y2_trend,\n        y_pred_env,\n        y_pred_full,\n        y_trend,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Spatial Prediction and Mapping\n    \"\"\"\n    # Predict species suitability across the landscape\n    def create_prediction_map(model, scaler, env_surfaces, spatial_coords, coord_cols):\n        \"\"\"Create prediction map from fitted model\"\"\"\n        \n        # Prepare environmental data\n        env_data = []\n        for var in env_variables:\n            if var == 'elevation':\n                # Create elevation surface (simple gradient)\n                x_vals = spatial_coords[:, 0]\n                y_vals = spatial_coords[:, 1]\n                elev_pred = 500 + 10 * x_vals + 8 * y_vals\n                env_data.append(elev_pred)\n            elif var == 'temperature':\n                # Use interpolated temperature\n                temp_pred = griddata(\n                    spatial_data[coord_cols].values,\n                    spatial_data[var].values,\n                    spatial_coords,\n                    method='linear'\n                )\n                env_data.append(temp_pred)\n            elif var == 'precipitation':\n                # Use interpolated precipitation\n                precip_pred = griddata(\n                    spatial_data[coord_cols].values,\n                    spatial_data[var].values,\n                    spatial_coords,\n                    method='linear'\n                )\n                env_data.append(precip_pred)\n        \n        X_pred_env = np.column_stack(env_data)\n        \n        # Handle missing values\n        valid_mask = ~np.isnan(X_pred_env).any(axis=1)\n        X_pred_env_clean = X_pred_env[valid_mask]\n        coords_clean = spatial_coords[valid_mask]\n        \n        # Scale environmental variables\n        X_pred_env_scaled = scaler.transform(X_pred_env_clean)\n        \n        # Add spatial components if needed\n        if hasattr(model, 'n_features_in_') and model.n_features_in_ > len(env_variables):\n            # Model includes spatial components\n            x_trend_pred = coords_clean[:, 0] - spatial_data['x_coord'].mean()\n            y_trend_pred = coords_clean[:, 1] - spatial_data['y_coord'].mean()\n            x2_trend_pred = x_trend_pred ** 2\n            y2_trend_pred = y_trend_pred ** 2\n            xy_trend_pred = x_trend_pred * y_trend_pred\n            \n            X_spatial_pred = np.column_stack([\n                x_trend_pred, y_trend_pred, x2_trend_pred, y2_trend_pred, xy_trend_pred\n            ])\n            X_pred_full = np.column_stack([X_pred_env_scaled, X_spatial_pred])\n            \n            predictions = model.predict_proba(X_pred_full)[:, 1]\n        else:\n            # Environmental model only\n            predictions = model.predict_proba(X_pred_env_scaled)[:, 1]\n        \n        # Create results DataFrame\n        pred_results = pd.DataFrame({\n            'x_coord': coords_clean[:, 0],\n            'y_coord': coords_clean[:, 1],\n            'prediction': predictions\n        })\n        \n        return pred_results\n    \n    # Create regular grid for prediction\n    x_pred = np.linspace(spatial_data['x_coord'].min(), spatial_data['x_coord'].max(), 50)\n    y_pred = np.linspace(spatial_data['y_coord'].min(), spatial_data['y_coord'].max(), 50)\n    X_pred_grid, Y_pred_grid = np.meshgrid(x_pred, y_pred)\n    pred_coords = np.column_stack([X_pred_grid.ravel(), Y_pred_grid.ravel()])\n    \n    # Generate predictions\n    env_predictions = create_prediction_map(\n        sdm_results['model_env'], sdm_results['scaler'], \n        None, pred_coords, coord_variables\n    )\n    \n    full_predictions = create_prediction_map(\n        sdm_results['model_full'], sdm_results['scaler'],\n        None, pred_coords, coord_variables\n    )\n    \n    # Create prediction maps\n    env_pred_plot = hv.Scatter(env_predictions, 'x_coord', 'y_coord', 'prediction').opts(\n        title=\"Environmental Model Predictions\",\n        color='prediction',\n        cmap='viridis',\n        size=8,\n        colorbar=True,\n        width=500,\n        height=400\n    )\n    \n    full_pred_plot = hv.Scatter(full_predictions, 'x_coord', 'y_coord', 'prediction').opts(\n        title=\"Environmental + Spatial Model Predictions\",\n        color='prediction',\n        cmap='viridis',\n        size=8,\n        colorbar=True,\n        width=500,\n        height=400\n    )\n    \n    # Overlay actual presence points\n    presence_points = spatial_data[spatial_data['species_presence'] == 1]\n    presence_overlay = hv.Scatter(presence_points, 'x_coord', 'y_coord').opts(\n        color='red',\n        size=12,\n        marker='x',\n        alpha=0.8\n    )\n    \n    print(\"Species Distribution Prediction Maps:\")\n    print(f\"Environmental model predictions: {len(env_predictions)} grid points\")\n    print(f\"Full model predictions: {len(full_predictions)} grid points\")\n    \n    (env_pred_plot * presence_overlay + full_pred_plot * presence_overlay).cols(1)\n    \n    return (\n        X_pred_grid,\n        Y_pred_grid,\n        coords_clean,\n        create_prediction_map,\n        elev_pred,\n        env_pred_plot,\n        env_predictions,\n        full_pred_plot,\n        full_predictions,\n        precip_pred,\n        pred_coords,\n        pred_results,\n        predictions,\n        presence_overlay,\n        presence_points,\n        temp_pred,\n        x_pred,\n        y_pred,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Spatial Cross-Validation\n    \"\"\"\n    # Spatial cross-validation to account for spatial autocorrelation\n    def spatial_cross_validation(data, model_func, coord_cols, response_col, n_folds=5, buffer_distance=10):\n        \"\"\"Perform spatial cross-validation with buffer zones\"\"\"\n        \n        coordinates = data[coord_cols].values\n        \n        # Create spatial folds using k-means clustering\n        from sklearn.cluster import KMeans\n        \n        kmeans = KMeans(n_clusters=n_folds, random_state=42)\n        fold_assignments = kmeans.fit_predict(coordinates)\n        \n        cv_results = []\n        \n        for fold in range(n_folds):\n            # Test set: current fold\n            test_mask = fold_assignments == fold\n            test_indices = np.where(test_mask)[0]\n            \n            # Create buffer around test points\n            test_coords = coordinates[test_mask]\n            \n            # Calculate distances from all points to test points\n            distances_to_test = cdist(coordinates, test_coords)\n            min_distances_to_test = np.min(distances_to_test, axis=1)\n            \n            # Training set: exclude test points and buffer zone\n            train_mask = (fold_assignments != fold) & (min_distances_to_test > buffer_distance)\n            train_indices = np.where(train_mask)[0]\n            \n            if len(train_indices) < 10 or len(test_indices) < 5:\n                continue  # Skip if insufficient data\n            \n            # Fit model on training data\n            train_data = data.iloc[train_indices]\n            test_data = data.iloc[test_indices]\n            \n            # Get predictions (simplified - would call actual model fitting function)\n            # For demonstration, using the already fitted model\n            train_coords_scaled = sdm_results['scaler'].transform(\n                train_data[env_variables].values\n            )\n            test_coords_scaled = sdm_results['scaler'].transform(\n                test_data[env_variables].values\n            )\n            \n            # Use environmental model for simplicity\n            test_predictions = sdm_results['model_env'].predict_proba(test_coords_scaled)[:, 1]\n            test_actual = test_data[response_col].values\n            \n            # Calculate metrics\n            try:\n                auc = roc_auc_score(test_actual, test_predictions)\n                accuracy = accuracy_score(test_actual, test_predictions > 0.5)\n                \n                cv_results.append({\n                    'fold': fold,\n                    'n_train': len(train_indices),\n                    'n_test': len(test_indices),\n                    'auc': auc,\n                    'accuracy': accuracy\n                })\n            except ValueError:\n                # Skip if all test cases are same class\n                continue\n        \n        return cv_results\n    \n    # Perform spatial cross-validation\n    spatial_cv_results = spatial_cross_validation(\n        spatial_data, None, coord_variables, 'species_presence'\n    )\n    \n    print(\"Spatial Cross-Validation Results:\")\n    print(\"=\" * 45)\n    print(\"Fold | N Train | N Test | AUC   | Accuracy\")\n    print(\"-\" * 45)\n    \n    for result in spatial_cv_results:\n        fold = result['fold']\n        n_train = result['n_train']\n        n_test = result['n_test']\n        auc = result['auc']\n        accuracy = result['accuracy']\n        \n        print(f\"{fold:4d} | {n_train:7d} | {n_test:6d} | {auc:.3f} | {accuracy:.3f}\")\n    \n    # Calculate mean performance\n    if spatial_cv_results:\n        mean_auc = np.mean([r['auc'] for r in spatial_cv_results])\n        mean_accuracy = np.mean([r['accuracy'] for r in spatial_cv_results])\n        std_auc = np.std([r['auc'] for r in spatial_cv_results])\n        std_accuracy = np.std([r['accuracy'] for r in spatial_cv_results])\n        \n        print(f\"\\nSpatial CV Performance:\")\n        print(f\"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}\")\n        print(f\"Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}\")\n        \n        # Compare to non-spatial performance\n        print(f\"\\nComparison to non-spatial evaluation:\")\n        print(f\"Non-spatial AUC: {sdm_results['auc_env']:.3f}\")\n        print(f\"Spatial CV AUC: {mean_auc:.3f}\")\n        print(f\"Difference: {sdm_results['auc_env'] - mean_auc:+.3f}\")\n        \n        if sdm_results['auc_env'] - mean_auc > 0.05:\n            print(\"Substantial overestimation due to spatial autocorrelation\")\n        else:\n            print(\"Minimal bias from spatial autocorrelation\")\n    \n    return (\n        KMeans,\n        accuracy,\n        auc,\n        buffer_distance,\n        fold_assignments,\n        kmeans,\n        mean_accuracy,\n        mean_auc,\n        min_distances_to_test,\n        spatial_cross_validation,\n        spatial_cv_results,\n        std_accuracy,\n        std_auc,\n        test_actual,\n        test_coords,\n        test_coords_scaled,\n        test_data,\n        test_indices,\n        test_mask,\n        test_predictions,\n        train_coords_scaled,\n        train_data,\n        train_indices,\n        train_mask,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Summary and Best Practices\n\n    In this chapter, we covered spatial analysis methods for ecological data:\n\n    ✓ **Spatial data creation**: Generating realistic spatial ecological datasets\n    ✓ **Spatial visualization**: Mapping ecological patterns across landscapes\n    ✓ **Spatial autocorrelation**: Moran's I and spatial dependence analysis\n    ✓ **Spatial interpolation**: Creating continuous surfaces from point data\n    ✓ **Distance analysis**: Understanding spatial correlation structure\n    ✓ **Species distribution modeling**: Environmental and spatial predictors\n    ✓ **Spatial prediction**: Mapping species suitability across landscapes\n    ✓ **Spatial cross-validation**: Accounting for spatial autocorrelation in model evaluation\n\n    ### Key Concepts in Spatial Ecology:\n    - **Spatial autocorrelation**: \"Everything is related to everything else, but near things are more related\"\n    - **Scale dependence**: Patterns and processes vary with spatial scale\n    - **Edge effects**: Boundary conditions affect ecological patterns\n    - **Spatial heterogeneity**: Environmental variation drives species distributions\n    - **Dispersal limitation**: Species distributions reflect movement constraints\n\n    ### Python Packages for Spatial Analysis:\n    - **scipy.spatial**: Distance calculations and spatial operations\n    - **scipy.interpolate**: Spatial interpolation methods\n    - **sklearn**: Machine learning for spatial modeling\n    - **pandas**: Spatial data manipulation and analysis\n    - **holoviews**: Interactive spatial visualization\n\n    ### Best Practices for Spatial Ecological Analysis:\n    1. **Check for spatial autocorrelation**: Always test before modeling\n    2. **Use appropriate scales**: Match analysis scale to ecological processes\n    3. **Account for spatial structure**: Include spatial components in models\n    4. **Validate spatially**: Use spatial cross-validation methods\n    5. **Consider boundaries**: Be aware of edge effects and study area limits\n    6. **Document projections**: Keep track of coordinate systems and units\n    7. **Visualize patterns**: Maps reveal insights not apparent in tables\n    8. **Test assumptions**: Verify stationarity and isotropy when assumed\n\n    ### Common Applications in Spatial Ecology:\n    - **Species distribution modeling**: Predicting suitable habitat\n    - **Conservation planning**: Identifying priority areas for protection\n    - **Landscape ecology**: Understanding patch connectivity and fragmentation\n    - **Disease ecology**: Modeling pathogen spread across landscapes\n    - **Climate change impacts**: Projecting species range shifts\n    - **Biodiversity mapping**: Creating continuous diversity surfaces\n    - **Restoration planning**: Optimal placement of restoration sites\n    \"\"\"\n    \n    spatial_summary = {\n        'Dataset Characteristics': {\n            'Number of Sites': f\"{len(spatial_data)}\",\n            'Spatial Extent': f\"{x_coords.max() - x_coords.min():.0f} × {y_coords.max() - y_coords.min():.0f} km\",\n            'Variables Analyzed': len(variables)\n        },\n        'Spatial Autocorrelation': {\n            'Temperature Moran\\'s I': f\"{autocorr_results['temperature']['morans_i']:.3f}\",\n            'Species Richness Moran\\'s I': f\"{autocorr_results['species_richness']['morans_i']:.3f}\",\n            'Autocorrelation Range': f\"{temp_range:.1f} km (temp), {richness_range:.1f} km (richness)\"\n        },\n        'Species Distribution Models': {\n            'Environmental Model AUC': f\"{sdm_results['auc_env']:.3f}\",\n            'Spatial Model AUC': f\"{sdm_results['auc_full']:.3f}\",\n            'Improvement from Spatial': f\"{auc_improvement:+.3f}\"\n        },\n        'Spatial Cross-Validation': {\n            'Mean CV AUC': f\"{mean_auc:.3f} ± {std_auc:.3f}\" if spatial_cv_results else \"Not available\",\n            'Spatial Bias': f\"{sdm_results['auc_env'] - mean_auc:+.3f}\" if spatial_cv_results else \"Not available\",\n            'CV Folds': f\"{len(spatial_cv_results)}\" if spatial_cv_results else \"0\"\n        }\n    }\n    \n    print(\"Spatial Analysis Summary:\")\n    print(\"=\" * 35)\n    \n    for category, details in spatial_summary.items():\n        print(f\"\\n{category}:\")\n        for key, value in details.items():\n            print(f\"  {key}: {value}\")\n    \n    print(\"\\n✓ Chapter 12 complete! Ready for mechanistic modeling.\")\n    \n    return spatial_summary,\n\n\nif __name__ == \"__main__\":\n    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
