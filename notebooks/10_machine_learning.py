import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 10: Machine Learning for Ecological Modeling

    This chapter covers machine learning applications in ecological research,
    using Python's scikit-learn and related libraries to build predictive
    models for species distribution, abundance, and diversity.

    ## Learning Objectives
    - Apply supervised learning for species classification and prediction
    - Use unsupervised learning for community pattern discovery
    - Perform feature selection and model validation
    - Understand ensemble methods and their ecological applications
    - Implement species distribution modeling workflows
    """
    )
    return


@app.cell
def __():
    # Essential imports for machine learning
    import pandas as pd
    import numpy as np
    import holoviews as hv
    from holoviews import opts
    
    # Scikit-learn imports
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Machine learning packages loaded")
    return (
        DBSCAN,
        DecisionTreeClassifier,
        GridSearchCV,
        KMeans,
        KNeighborsClassifier,
        LabelEncoder,
        Lasso,
        LogisticRegression,
        PCA,
        RFE,
        RandomForestClassifier,
        RandomForestRegressor,
        Ridge,
        SVC,
        SVR,
        SelectKBest,
        StandardScaler,
        classification_report,
        confusion_matrix,
        cross_val_score,
        f_classif,
        hv,
        mean_squared_error,
        np,
        opts,
        pd,
        r2_score,
        train_test_split,
        warnings,
    )


@app.cell
def __():
    """
    ## Create Comprehensive Ecological Dataset
    """
    # Generate realistic ecological data with multiple variables
    np.random.seed(42)
    n_sites = 500
    
    # Environmental variables
    temperature = np.random.normal(15, 5, n_sites)
    precipitation = np.random.lognormal(6, 0.5, n_sites)
    elevation = np.random.uniform(0, 2000, n_sites)
    soil_pH = np.random.normal(6.5, 1.2, n_sites)
    humidity = np.random.normal(60, 15, n_sites)
    
    # Derived environmental variables
    temp_range = np.abs(np.random.normal(10, 3, n_sites))
    growing_season = 365 - elevation * 0.1 + temperature * 5 + np.random.normal(0, 20, n_sites)
    growing_season = np.clip(growing_season, 100, 300)
    
    # Habitat classification based on environmental conditions
    def assign_habitat(temp, precip, elev):
        if elev > 1500:
            return 'Alpine'
        elif temp < 5:
            return 'Boreal' 
        elif precip > 1000 and temp > 20:
            return 'Tropical'
        elif precip < 300:
            return 'Desert'
        elif temp > 15 and precip > 800:
            return 'Temperate_Forest'
        else:
            return 'Grassland'
    
    habitat = np.array([assign_habitat(t, p, e) for t, p, e in zip(temperature, precipitation, elevation)])
    
    # Species richness with complex environmental relationships
    species_richness = (
        25 + 
        0.2 * precipitation/100 +
        -0.003 * elevation +
        2 * np.exp(-(temperature - 20)**2 / 50) +  # Optimal temperature around 20°C
        np.random.normal(0, 3, n_sites)
    )
    species_richness = np.maximum(species_richness, 1).astype(int)
    
    # Shannon diversity
    shannon_diversity = np.log(species_richness) * np.random.uniform(0.8, 1.2, n_sites)
    
    # Total abundance
    total_abundance = np.random.lognormal(4 + species_richness/30, 0.5, n_sites)
    
    # Create target variables for different ML tasks
    # Binary: High diversity (above 75th percentile)
    high_diversity = (species_richness > np.percentile(species_richness, 75)).astype(int)
    
    # Multi-class: Diversity categories
    diversity_quartiles = pd.qcut(species_richness, q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # Create DataFrame
    ml_data = pd.DataFrame({
        'site_id': [f"SITE_{i:03d}" for i in range(1, n_sites + 1)],
        'temperature': temperature,
        'precipitation': precipitation,
        'elevation': elevation,
        'soil_pH': soil_pH,
        'humidity': humidity,
        'temp_range': temp_range,
        'growing_season': growing_season,
        'habitat': habitat,
        'species_richness': species_richness,
        'shannon_diversity': shannon_diversity,
        'total_abundance': total_abundance,
        'high_diversity': high_diversity,
        'diversity_category': diversity_quartiles
    })
    
    print(f"ML dataset created: {ml_data.shape}")
    print("\nHabitat distribution:")
    print(ml_data['habitat'].value_counts())
    print("\nDiversity category distribution:")
    print(ml_data['diversity_category'].value_counts())
    
    return (
        assign_habitat,
        diversity_quartiles,
        growing_season,
        habitat,
        high_diversity,
        humidity,
        ml_data,
        n_sites,
        shannon_diversity,
        soil_pH,
        species_richness,
        temp_range,
        temperature,
        total_abundance,
    )


@app.cell
def __():
    """
    ## Classification: Predicting Habitat Types
    """
    # Prepare data for habitat classification
    feature_cols = ['temperature', 'precipitation', 'elevation', 'soil_pH', 'humidity', 'temp_range']
    X = ml_data[feature_cols]
    y = ml_data['habitat']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Habitat Classification Setup:")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_encoded))}")
    
    # Train multiple classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train and evaluate classifiers
    classifier_results = {}
    
    for name, clf in classifiers.items():
        # Use scaled data for SVM, Logistic Regression, KNN
        if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        classifier_results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
    
    # Display results
    print("\nClassifier Performance:")
    print("Classifier              | Accuracy | CV Mean ± Std")
    print("-" * 50)
    
    for name, results in classifier_results.items():
        print(f"{name:22} | {results['accuracy']:.3f}    | {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    return (
        X,
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        classifier_results,
        classifiers,
        feature_cols,
        label_encoder,
        scaler,
        y,
        y_encoded,
        y_test,
        y_train,
    )


@app.cell
def __():
    """
    ## Detailed Analysis of Best Classifier
    """
    # Find best classifier
    best_classifier_name = max(classifier_results.keys(), 
                              key=lambda x: classifier_results[x]['accuracy'])
    best_classifier = classifier_results[best_classifier_name]
    
    print(f"Best Classifier: {best_classifier_name}")
    print(f"Accuracy: {best_classifier['accuracy']:.3f}")
    
    # Detailed classification report
    y_pred_best = best_classifier['predictions']
    class_names = label_encoder.classes_
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Feature importance (for tree-based models)
    if best_classifier_name in ['Random Forest', 'Decision Tree']:
        feature_importance = best_classifier['model'].feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
    
    return (
        best_classifier,
        best_classifier_name,
        class_names,
        cm,
        cm_df,
        feature_importance,
        importance_df,
        y_pred_best,
    )


@app.cell
def __():
    """
    ## Regression: Predicting Species Richness
    """
    # Prepare data for regression
    X_reg = ml_data[feature_cols]
    y_reg = ml_data['species_richness']
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    # Train multiple regressors
    regressors = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf')
    }
    
    # Train and evaluate regressors
    regressor_results = {}
    
    for name, reg in regressors.items():
        # Use scaled data for regularized methods and SVR
        if name in ['Ridge Regression', 'Lasso Regression', 'SVR']:
            reg.fit(X_train_reg_scaled, y_train_reg)
            y_pred_reg = reg.predict(X_test_reg_scaled)
            cv_scores_reg = cross_val_score(reg, X_train_reg_scaled, y_train_reg, cv=5, scoring='r2')
        else:
            reg.fit(X_train_reg, y_train_reg)
            y_pred_reg = reg.predict(X_test_reg)
            cv_scores_reg = cross_val_score(reg, X_train_reg, y_train_reg, cv=5, scoring='r2')
        
        # Calculate metrics
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        
        regressor_results[name] = {
            'model': reg,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores_reg.mean(),
            'cv_std': cv_scores_reg.std(),
            'predictions': y_pred_reg
        }
    
    # Display results
    print("Species Richness Regression Results:")
    print("Regressor               | RMSE  | R²    | CV R² ± Std")
    print("-" * 55)
    
    for name, results in regressor_results.items():
        print(f"{name:22} | {results['rmse']:.2f}  | {results['r2']:.3f} | {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    return (
        X_reg,
        X_test_reg,
        X_test_reg_scaled,
        X_train_reg,
        X_train_reg_scaled,
        regressor_results,
        regressors,
        scaler_reg,
        y_reg,
        y_test_reg,
        y_train_reg,
    )


@app.cell
def __():
    """
    ## Hyperparameter Tuning
    """
    # Grid search for Random Forest hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search for classification
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search_clf = GridSearchCV(
        rf_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    print("Performing Grid Search for Random Forest Classifier...")
    grid_search_clf.fit(X_train, y_train)
    
    print(f"Best Classification Parameters: {grid_search_clf.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search_clf.best_score_:.3f}")
    
    # Test the tuned model
    best_rf_clf = grid_search_clf.best_estimator_
    y_pred_tuned = best_rf_clf.predict(X_test)
    tuned_accuracy = np.mean(y_pred_tuned == y_test)
    
    print(f"Tuned Model Test Accuracy: {tuned_accuracy:.3f}")
    
    # Grid search for regression
    rf_regressor = RandomForestRegressor(random_state=42)
    grid_search_reg = GridSearchCV(
        rf_regressor, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    
    print("\nPerforming Grid Search for Random Forest Regressor...")
    grid_search_reg.fit(X_train_reg, y_train_reg)
    
    print(f"Best Regression Parameters: {grid_search_reg.best_params_}")
    print(f"Best Cross-Validation R²: {grid_search_reg.best_score_:.3f}")
    
    # Test the tuned regression model
    best_rf_reg = grid_search_reg.best_estimator_
    y_pred_reg_tuned = best_rf_reg.predict(X_test_reg)
    tuned_r2 = r2_score(y_test_reg, y_pred_reg_tuned)
    
    print(f"Tuned Regression Model Test R²: {tuned_r2:.3f}")
    
    return (
        best_rf_clf,
        best_rf_reg,
        grid_search_clf,
        grid_search_reg,
        param_grid,
        rf_classifier,
        rf_regressor,
        tuned_accuracy,
        tuned_r2,
        y_pred_reg_tuned,
        y_pred_tuned,
    )


@app.cell
def __():
    """
    ## Feature Selection
    """
    # Univariate feature selection
    selector = SelectKBest(score_func=f_classif, k=4)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected features
    selected_features = np.array(feature_cols)[selector.get_support()]
    feature_scores = selector.scores_
    
    print("Univariate Feature Selection (Classification):")
    print(f"Selected features: {list(selected_features)}")
    
    # Display all feature scores
    feature_score_df = pd.DataFrame({
        'feature': feature_cols,
        'score': feature_scores,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print("\nFeature Scores:")
    print(feature_score_df)
    
    # Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=4)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    
    rfe_features = np.array(feature_cols)[rfe.support_]
    print(f"\nRFE Selected features: {list(rfe_features)}")
    
    # Compare model performance with feature selection
    models_comparison = {}
    
    # Original features
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train, y_train)
    original_score = rf_original.score(X_test, y_test)
    models_comparison['All Features'] = original_score
    
    # Univariate selection
    rf_univariate = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_univariate.fit(X_train_selected, y_train)
    univariate_score = rf_univariate.score(X_test_selected, y_test)
    models_comparison['Univariate Selection'] = univariate_score
    
    # RFE selection
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rfe.fit(X_train_rfe, y_train)
    rfe_score = rf_rfe.score(X_test_rfe, y_test)
    models_comparison['RFE Selection'] = rfe_score
    
    print("\nModel Performance Comparison:")
    for method, score in models_comparison.items():
        print(f"{method:20}: {score:.3f}")
    
    return (
        X_test_rfe,
        X_test_selected,
        X_train_rfe,
        X_train_selected,
        feature_score_df,
        feature_scores,
        models_comparison,
        original_score,
        rfe,
        rfe_features,
        rfe_score,
        rf_original,
        rf_rfe,
        rf_univariate,
        selected_features,
        selector,
        univariate_score,
    )


@app.cell
def __():
    """
    ## Unsupervised Learning: Clustering Ecological Sites
    """
    # Prepare data for clustering
    X_cluster = ml_data[feature_cols]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    
    # K-Means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    
    # Add cluster labels to data
    ml_data['cluster'] = cluster_labels
    
    print(f"K-Means Clustering (k={n_clusters}):")
    print("Cluster distribution:")
    print(pd.Series(cluster_labels).value_counts().sort_index())
    
    # Analyze clusters
    cluster_summary = ml_data.groupby('cluster')[feature_cols + ['species_richness']].mean()
    print("\nCluster Characteristics (means):")
    print(cluster_summary.round(2))
    
    # DBSCAN clustering (density-based)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_cluster_scaled)
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"\nDBSCAN Clustering:")
    print(f"Number of clusters: {n_clusters_dbscan}")
    print(f"Number of noise points: {n_noise}")
    
    # Principal Component Analysis for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"\nPCA Explained Variance:")
    print(f"PC1: {explained_variance[0]:.3f}")
    print(f"PC2: {explained_variance[1]:.3f}")
    print(f"Total: {explained_variance.sum():.3f}")
    
    # Create visualization data
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'habitat': ml_data['habitat'],
        'cluster': cluster_labels,
        'species_richness': ml_data['species_richness']
    })
    
    return (
        X_cluster,
        X_cluster_scaled,
        X_pca,
        cluster_labels,
        cluster_summary,
        dbscan,
        dbscan_labels,
        explained_variance,
        kmeans,
        n_clusters,
        n_clusters_dbscan,
        n_noise,
        pca,
        pca_df,
    )


@app.cell
def __():
    """
    ## Species Distribution Modeling (SDM)
    """
    # Create binary presence/absence for a "focal species"
    # Simulate species with specific environmental preferences
    def simulate_species_presence(temp, precip, elev, ph):
        # Species prefers moderate temperatures, high precipitation, low elevation, neutral pH
        temp_suitability = np.exp(-(temp - 18)**2 / 50)
        precip_suitability = np.exp(-(precip - 1200)**2 / 200000)
        elev_suitability = np.exp(-elev / 1000)
        ph_suitability = np.exp(-(ph - 7)**2 / 2)
        
        # Combined suitability
        suitability = temp_suitability * precip_suitability * elev_suitability * ph_suitability
        
        # Convert to presence probability
        probability = suitability / (1 + suitability)
        
        # Generate presence/absence with some randomness
        presence = np.random.binomial(1, probability)
        return presence, probability
    
    species_presence, habitat_suitability = simulate_species_presence(
        ml_data['temperature'], 
        ml_data['precipitation'],
        ml_data['elevation'],
        ml_data['soil_pH']
    )
    
    ml_data['species_presence'] = species_presence
    ml_data['habitat_suitability'] = habitat_suitability
    
    print(f"Species Distribution Modeling:")
    print(f"Species prevalence: {species_presence.mean():.3f}")
    print(f"Number of presence records: {species_presence.sum()}")
    print(f"Number of absence records: {len(species_presence) - species_presence.sum()}")
    
    # SDM using different algorithms
    X_sdm = ml_data[feature_cols]
    y_sdm = ml_data['species_presence']
    
    # Split data
    X_train_sdm, X_test_sdm, y_train_sdm, y_test_sdm = train_test_split(
        X_sdm, y_sdm, test_size=0.3, random_state=42, stratify=y_sdm
    )
    
    # Scale features
    scaler_sdm = StandardScaler()
    X_train_sdm_scaled = scaler_sdm.fit_transform(X_train_sdm)
    X_test_sdm_scaled = scaler_sdm.transform(X_test_sdm)
    
    # Train SDM models
    sdm_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    sdm_results = {}
    
    for name, model in sdm_models.items():
        if name == 'Random Forest':
            model.fit(X_train_sdm, y_train_sdm)
            y_pred_sdm = model.predict(X_test_sdm)
            y_prob_sdm = model.predict_proba(X_test_sdm)[:, 1]
        else:
            model.fit(X_train_sdm_scaled, y_train_sdm)
            y_pred_sdm = model.predict(X_test_sdm_scaled) 
            y_prob_sdm = model.predict_proba(X_test_sdm_scaled)[:, 1]
        
        accuracy = np.mean(y_pred_sdm == y_test_sdm)
        
        sdm_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred_sdm,
            'probabilities': y_prob_sdm
        }
    
    # Display SDM results
    print("\nSpecies Distribution Model Performance:")
    for name, results in sdm_results.items():
        print(f"{name}: Accuracy = {results['accuracy']:.3f}")
    
    return (
        X_sdm,
        X_test_sdm,
        X_test_sdm_scaled,
        X_train_sdm,
        X_train_sdm_scaled,
        habitat_suitability,
        scaler_sdm,
        sdm_models,
        sdm_results,
        simulate_species_presence,
        species_presence,
        y_sdm,
        y_test_sdm,
        y_train_sdm,
    )


@app.cell
def __():
    """
    ## Model Interpretation and Variable Importance
    """
    # Feature importance from Random Forest models
    rf_clf_importance = best_rf_clf.feature_importances_
    rf_reg_importance = regressor_results['Random Forest']['model'].feature_importances_
    
    importance_comparison = pd.DataFrame({
        'feature': feature_cols,
        'classification_importance': rf_clf_importance,
        'regression_importance': rf_reg_importance
    })
    
    print("Feature Importance Comparison:")
    print(importance_comparison.round(3))
    
    # Partial dependence analysis (simplified)
    def calculate_partial_dependence(model, X, feature_idx, feature_name):
        """Calculate partial dependence for a single feature"""
        X_temp = X.copy()
        feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)
        
        predictions = []
        for value in feature_values:
            X_temp[:, feature_idx] = value
            pred = model.predict(X_temp).mean()
            predictions.append(pred)
        
        return feature_values, predictions
    
    # Calculate partial dependence for key features
    rf_model = regressor_results['Random Forest']['model']
    
    print("\nPartial Dependence Analysis (Species Richness):")
    
    for i, feature in enumerate(feature_cols[:3]):  # Top 3 features
        feature_values, predictions = calculate_partial_dependence(
            rf_model, X_train_reg.values, i, feature
        )
        
        print(f"\n{feature}:")
        print(f"  Range: {feature_values.min():.1f} - {feature_values.max():.1f}")
        print(f"  Prediction range: {np.min(predictions):.1f} - {np.max(predictions):.1f}")
    
    return (
        calculate_partial_dependence,
        feature_values,
        importance_comparison,
        predictions,
        rf_clf_importance,
        rf_model,
        rf_reg_importance,
    )


@app.cell
def __():
    """
    ## Cross-Validation and Model Validation
    """
    from sklearn.model_selection import StratifiedKFold, validation_curve
    
    # Detailed cross-validation for best models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Classification CV
    clf_cv_scores = cross_val_score(best_rf_clf, X, y_encoded, cv=cv, scoring='accuracy')
    
    print("Detailed Cross-Validation Results:")
    print(f"Classification (Random Forest):")
    print(f"  CV Scores: {clf_cv_scores}")
    print(f"  Mean: {clf_cv_scores.mean():.3f} ± {clf_cv_scores.std():.3f}")
    
    # Regression CV with different metrics
    reg_cv_scores_r2 = cross_val_score(best_rf_reg, X_reg, y_reg, cv=5, scoring='r2')
    reg_cv_scores_neg_mse = cross_val_score(best_rf_reg, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error')
    
    print(f"\nRegression (Random Forest):")
    print(f"  R² CV Scores: {reg_cv_scores_r2}")
    print(f"  R² Mean: {reg_cv_scores_r2.mean():.3f} ± {reg_cv_scores_r2.std():.3f}")
    print(f"  RMSE Mean: {np.sqrt(-reg_cv_scores_neg_mse.mean()):.2f}")
    
    # Validation curves for hyperparameter sensitivity
    param_range = [10, 50, 100, 200, 300]
    train_scores, val_scores = validation_curve(
        RandomForestRegressor(random_state=42), X_reg, y_reg,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='r2'
    )
    
    print(f"\nValidation Curve (n_estimators):")
    for i, n_est in enumerate(param_range):
        print(f"  {n_est:3d} trees: Train R² = {train_scores[i].mean():.3f}, Val R² = {val_scores[i].mean():.3f}")
    
    return (
        StratifiedKFold,
        clf_cv_scores,
        cv,
        param_range,
        reg_cv_scores_neg_mse,
        reg_cv_scores_r2,
        train_scores,
        val_scores,
        validation_curve,
    )


@app.cell
def __():
    """
    ## Summary and Best Practices

    In this chapter, we covered machine learning applications for ecological modeling:

    ✓ **Classification**: Habitat type prediction using environmental variables
    ✓ **Regression**: Species richness prediction with multiple algorithms
    ✓ **Hyperparameter tuning**: Grid search for optimal model parameters
    ✓ **Feature selection**: Univariate and recursive feature elimination
    ✓ **Unsupervised learning**: Site clustering and dimensionality reduction
    ✓ **Species Distribution Modeling**: Presence/absence prediction
    ✓ **Model interpretation**: Feature importance and partial dependence
    ✓ **Cross-validation**: Robust model evaluation and validation

    ### Key Machine Learning Packages:
    - **scikit-learn**: Comprehensive ML algorithms and tools
    - **pandas**: Data manipulation and preparation
    - **numpy**: Numerical computing and array operations
    - **scipy**: Statistical functions and optimization

    ### Best Practices for Ecological ML:
    1. **Data preparation**: Handle missing values, outliers, and scaling
    2. **Feature engineering**: Create biologically meaningful variables
    3. **Model selection**: Compare multiple algorithms and use CV
    4. **Hyperparameter tuning**: Optimize model performance systematically
    5. **Feature selection**: Remove irrelevant or redundant variables
    6. **Model interpretation**: Understand what drives predictions
    7. **Validation**: Use independent data when possible
    8. **Biological realism**: Ensure predictions make ecological sense

    ### Common Ecological ML Applications:
    - **Species Distribution Modeling (SDM)**: MaxEnt, Random Forest, GLM
    - **Community Classification**: Habitat type prediction
    - **Abundance Prediction**: Population modeling
    - **Diversity Estimation**: Alpha and beta diversity patterns
    - **Environmental Gradient Analysis**: Ordination and clustering
    - **Conservation Planning**: Priority area identification
    """
    
    ml_summary = {
        'Classification Models': {
            'Best': best_classifier_name,
            'Accuracy': f"{best_classifier['accuracy']:.3f}",
            'Applications': 'Habitat classification, species identification'
        },
        'Regression Models': {
            'Best': max(regressor_results.keys(), key=lambda x: regressor_results[x]['r2']),
            'R²': f"{max(regressor_results.values(), key=lambda x: x['r2'])['r2']:.3f}",
            'Applications': 'Species richness prediction, abundance modeling'
        },
        'Clustering': {
            'Methods': 'K-Means, DBSCAN',
            'Clusters': f"{n_clusters} (K-Means), {n_clusters_dbscan} (DBSCAN)",
            'Applications': 'Site classification, community patterns'
        },
        'Feature Selection': {
            'Methods': 'Univariate, RFE',
            'Top Features': list(selected_features[:3]),
            'Applications': 'Model simplification, variable importance'
        }
    }
    
    print("Machine Learning Summary for Ecological Data:")
    print("=" * 50)
    
    for category, details in ml_summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Chapter 10 complete! Ready for time series analysis.")
    
    return ml_summary,


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()