import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 7: Bayesian Biostatistics for Ecological Data

    This chapter introduces Bayesian statistical methods for ecological research,
    covering prior specification, posterior inference, and model comparison using
    Python's statistical computing tools.

    ## Learning Objectives
    - Understand Bayesian inference principles for ecology
    - Specify appropriate priors for ecological parameters
    - Perform Bayesian model fitting and diagnostics
    - Compare models using Bayesian information criteria
    - Apply hierarchical models for ecological data
    """
    )
    return


@app.cell
def __():
    # Essential imports for Bayesian analysis
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from scipy.optimize import minimize
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    
    # Note: In a full environment, you might use PyMC, Stan, or similar
    # Here we'll implement simplified Bayesian methods compatible with Pyodide
    
    hv.extension('bokeh')
    
    print("✓ Bayesian statistics packages loaded")
    print("Note: Using simplified Bayesian methods compatible with Pyodide")
    return hv, minimize, np, opts, pd, stats, warnings


@app.cell
def __():
    """
    ## Bayesian Fundamentals for Ecology

    ### Bayes' Theorem:
    P(θ|data) = P(data|θ) × P(θ) / P(data)

    Where:
    - P(θ|data) = Posterior (what we want)
    - P(data|θ) = Likelihood (how well model fits data)
    - P(θ) = Prior (our initial beliefs)
    - P(data) = Marginal likelihood (normalizing constant)

    ### Why Bayesian Methods in Ecology?
    - **Uncertainty quantification**: Natural way to express uncertainty
    - **Prior knowledge**: Incorporate expert knowledge and previous studies
    - **Small samples**: Better performance with limited data
    - **Hierarchical models**: Natural for nested ecological data
    - **Missing data**: Elegant handling of incomplete observations
    - **Model comparison**: Principled approach to model selection
    """
    return


@app.cell
def __():
    """
    ## Create Ecological Dataset for Bayesian Analysis
    """
    # Generate realistic ecological data
    np.random.seed(42)
    n_sites = 100
    
    # Environmental variables
    temperature = np.random.normal(15, 5, n_sites)
    precipitation = np.random.lognormal(6, 0.5, n_sites)
    habitat = np.random.choice(['Forest', 'Grassland', 'Wetland'], n_sites, p=[0.5, 0.3, 0.2])
    
    # Create habitat effects
    habitat_effects = {'Forest': 1.2, 'Grassland': 1.0, 'Wetland': 1.1}
    habitat_multiplier = np.array([habitat_effects[h] for h in habitat])
    
    # Species richness with environmental relationships
    log_richness = (
        2.5 +  # Intercept (log scale)
        0.05 * (temperature - 15) +  # Temperature effect
        0.3 * (np.log(precipitation) - 6) +  # Precipitation effect
        np.log(habitat_multiplier) +  # Habitat effect
        np.random.normal(0, 0.3, n_sites)  # Random error
    )
    
    species_richness = np.exp(log_richness).astype(int)
    species_richness = np.maximum(species_richness, 1)  # Minimum 1 species
    
    # Presence/absence data for a focal species
    # Probability depends on temperature (optimal around 18°C)
    temp_suitability = np.exp(-((temperature - 18) / 8)**2)
    presence_prob = 0.1 + 0.7 * temp_suitability  # Base + temperature effect
    species_presence = np.random.binomial(1, presence_prob, n_sites)
    
    # Create DataFrame
    bayesian_data = pd.DataFrame({
        'site_id': [f"SITE_{i:03d}" for i in range(1, n_sites + 1)],
        'temperature': temperature,
        'precipitation': precipitation,
        'habitat': habitat,
        'species_richness': species_richness,
        'species_presence': species_presence,
        'log_precipitation': np.log(precipitation)
    })
    
    print(f"Bayesian analysis dataset created: {bayesian_data.shape}")
    print(f"Species presence rate: {species_presence.mean():.2%}")
    print(f"Mean species richness: {species_richness.mean():.1f}")
    
    return (
        bayesian_data,
        habitat,
        habitat_effects,
        habitat_multiplier,
        log_richness,
        n_sites,
        precipitation,
        presence_prob,
        species_presence,
        species_richness,
        temp_suitability,
        temperature,
    )


@app.cell
def __():
    """
    ## Bayesian Linear Regression
    """
    # Bayesian linear regression for species richness
    def bayesian_linear_regression(X, y, prior_mean=0, prior_var=10, error_var=1):
        """
        Bayesian linear regression with conjugate priors
        """
        # Add intercept
        X_design = np.column_stack([np.ones(len(X)), X])
        
        # Prior parameters
        n_params = X_design.shape[1]
        prior_precision = np.eye(n_params) / prior_var
        prior_mean_vec = np.full(n_params, prior_mean)
        
        # Posterior parameters (conjugate normal-inverse-gamma)
        posterior_precision = prior_precision + (X_design.T @ X_design) / error_var
        posterior_cov = np.linalg.inv(posterior_precision)
        
        posterior_mean = posterior_cov @ (
            prior_precision @ prior_mean_vec + 
            (X_design.T @ y) / error_var
        )
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_cov': posterior_cov,
            'posterior_precision': posterior_precision,
            'X_design': X_design
        }
    
    # Fit Bayesian regression: Species richness ~ Temperature
    X_temp = bayesian_data['temperature'].values
    y_richness = bayesian_data['species_richness'].values
    
    # Standardize predictors
    X_temp_std = (X_temp - X_temp.mean()) / X_temp.std()
    
    # Fit model with different priors
    priors = {
        'uninformative': {'prior_var': 100},  # Vague prior
        'weakly_informative': {'prior_var': 10},  # Moderate prior  
        'informative': {'prior_var': 1}  # Strong prior
    }
    
    results = {}
    for prior_name, prior_params in priors.items():
        result = bayesian_linear_regression(
            X_temp_std, y_richness, 
            prior_var=prior_params['prior_var']
        )
        results[prior_name] = result
    
    # Display results
    print("Bayesian Linear Regression Results:")
    print("Species Richness ~ Temperature")
    print("=" * 50)
    
    print("Prior Type        | Intercept | Slope  | Slope Std")
    print("-" * 50)
    
    for prior_name, result in results.items():
        intercept = result['posterior_mean'][0]
        slope = result['posterior_mean'][1] 
        slope_std = np.sqrt(result['posterior_cov'][1, 1])
        
        print(f"{prior_name:16} | {intercept:8.2f}  | {slope:6.3f} | {slope_std:8.3f}")
    
    # Credible intervals
    def calculate_credible_interval(mean, var, alpha=0.05):
        """Calculate Bayesian credible interval"""
        std = np.sqrt(var)
        t_value = stats.t.ppf(1 - alpha/2, df=len(y_richness) - 2)
        lower = mean - t_value * std
        upper = mean + t_value * std
        return lower, upper
    
    print(f"\n95% Credible Intervals (Weakly Informative Prior):")
    result = results['weakly_informative']
    
    for i, param_name in enumerate(['Intercept', 'Slope']):
        mean = result['posterior_mean'][i]
        var = result['posterior_cov'][i, i]
        lower, upper = calculate_credible_interval(mean, var)
        print(f"{param_name}: [{lower:.3f}, {upper:.3f}]")
    
    return (
        X_temp,
        X_temp_std,
        bayesian_linear_regression,
        calculate_credible_interval,
        lower,
        priors,
        result,
        results,
        upper,
        y_richness,
    )


@app.cell
def __():
    """
    ## Bayesian Logistic Regression
    """
    # Bayesian logistic regression for species presence
    def bayesian_logistic_regression_mcmc(X, y, n_iter=1000, prior_var=10):
        """
        Simple Metropolis-Hastings for Bayesian logistic regression
        """
        # Add intercept
        X_design = np.column_stack([np.ones(len(X)), X])
        n_params = X_design.shape[1]
        
        # Initialize parameters
        beta = np.zeros(n_params)
        beta_samples = []
        
        # Proposal variance
        proposal_var = 0.1
        
        # Log likelihood function
        def log_likelihood(beta, X, y):
            linear_pred = X @ beta
            # Prevent overflow
            linear_pred = np.clip(linear_pred, -500, 500)
            prob = 1 / (1 + np.exp(-linear_pred))
            prob = np.clip(prob, 1e-15, 1 - 1e-15)  # Prevent log(0)
            return np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        
        # Log prior (normal)
        def log_prior(beta, var=prior_var):
            return -0.5 * np.sum(beta**2) / var
        
        # MCMC sampling
        accepted = 0
        
        for i in range(n_iter):
            # Propose new parameters
            beta_new = beta + np.random.normal(0, proposal_var, n_params)
            
            # Calculate acceptance probability
            log_p_new = log_likelihood(beta_new, X_design, y) + log_prior(beta_new)
            log_p_old = log_likelihood(beta, X_design, y) + log_prior(beta)
            
            log_alpha = min(0, log_p_new - log_p_old)
            
            # Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                beta = beta_new
                accepted += 1
            
            # Store sample (after burn-in)
            if i >= n_iter // 2:
                beta_samples.append(beta.copy())
        
        acceptance_rate = accepted / n_iter
        beta_samples = np.array(beta_samples)
        
        return {
            'samples': beta_samples,
            'acceptance_rate': acceptance_rate,
            'posterior_mean': np.mean(beta_samples, axis=0),
            'posterior_std': np.std(beta_samples, axis=0)
        }
    
    # Fit Bayesian logistic regression: Species presence ~ Temperature
    X_temp_presence = (bayesian_data['temperature'] - bayesian_data['temperature'].mean()) / bayesian_data['temperature'].std()
    y_presence = bayesian_data['species_presence'].values
    
    print("Fitting Bayesian Logistic Regression...")
    print("Species Presence ~ Temperature")
    
    logistic_result = bayesian_logistic_regression_mcmc(
        X_temp_presence, y_presence, n_iter=2000
    )
    
    print(f"MCMC completed. Acceptance rate: {logistic_result['acceptance_rate']:.2%}")
    
    # Display results
    print(f"\nPosterior Summary:")
    print("Parameter | Mean   | Std    | 95% CI")
    print("-" * 40)
    
    param_names = ['Intercept', 'Temperature']
    for i, name in enumerate(param_names):
        mean = logistic_result['posterior_mean'][i]
        std = logistic_result['posterior_std'][i]
        
        # Calculate credible interval from samples
        samples = logistic_result['samples'][:, i]
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)
        
        print(f"{name:9} | {mean:6.3f} | {std:6.3f} | [{ci_lower:6.3f}, {ci_upper:6.3f}]")
    
    # Convert to probability scale
    temp_effect = logistic_result['posterior_mean'][1]
    print(f"\nTemperature effect (log-odds): {temp_effect:.3f}")
    print(f"Odds ratio per 1 SD increase: {np.exp(temp_effect):.3f}")
    
    return (
        X_temp_presence,
        bayesian_logistic_regression_mcmc,
        ci_lower,
        ci_upper,
        logistic_result,
        param_names,
        samples,
        temp_effect,
        y_presence,
    )


@app.cell
def __():
    """
    ## Hierarchical Bayesian Models
    """
    # Hierarchical model for species richness by habitat
    def hierarchical_model_simple(data, group_col, response_col):
        """
        Simple hierarchical model: response varies by group
        """
        groups = data[group_col].unique()
        n_groups = len(groups)
        
        # Group-level data
        group_data = {}
        for group in groups:
            group_subset = data[data[group_col] == group]
            group_data[group] = {
                'values': group_subset[response_col].values,
                'n': len(group_subset),
                'mean': group_subset[response_col].mean(),
                'var': group_subset[response_col].var()
            }
        
        # Hyperpriors (population level)
        # Grand mean and variance
        grand_mean = data[response_col].mean()
        grand_var = data[response_col].var()
        
        # Empirical Bayes estimates
        # Shrinkage towards grand mean based on sample size and variance
        group_estimates = {}\n        
        for group in groups:\n            gdata = group_data[group]\n            \n            # James-Stein type shrinkage\n            # More shrinkage for smaller samples and higher variance\n            shrinkage_factor = gdata['n'] / (gdata['n'] + grand_var / gdata['var'])\n            \n            posterior_mean = (\n                shrinkage_factor * gdata['mean'] + \n                (1 - shrinkage_factor) * grand_mean\n            )\n            \n            # Posterior variance (simplified)\n            posterior_var = gdata['var'] / gdata['n']\n            \n            group_estimates[group] = {\n                'posterior_mean': posterior_mean,\n                'posterior_var': posterior_var,\n                'shrinkage': 1 - shrinkage_factor,\n                'sample_mean': gdata['mean'],\n                'sample_size': gdata['n']\n            }\n        \n        return group_estimates, group_data\n    \n    # Apply hierarchical model to species richness by habitat\n    habitat_model, habitat_data = hierarchical_model_simple(\n        bayesian_data, 'habitat', 'species_richness'\n    )\n    \n    print(\"Hierarchical Model: Species Richness by Habitat\")\n    print(\"=\" * 55)\n    print(\"Habitat   | Sample | Sample | Posterior | Posterior | Shrinkage\")\n    print(\"          |   N    |  Mean  |    Mean   |    Std    |    %\")\n    print(\"-\" * 55)\n    \n    for habitat in ['Forest', 'Grassland', 'Wetland']:\n        if habitat in habitat_model:\n            est = habitat_model[habitat]\n            sample_mean = est['sample_mean']\n            posterior_mean = est['posterior_mean']\n            posterior_std = np.sqrt(est['posterior_var'])\n            shrinkage_pct = est['shrinkage'] * 100\n            sample_size = est['sample_size']\n            \n            print(f\"{habitat:9} | {sample_size:6d} | {sample_mean:6.1f} | {posterior_mean:9.1f} | {posterior_std:9.2f} | {shrinkage_pct:8.1f}\")\n    \n    # Compare with classical (non-hierarchical) estimates\n    print(f\"\\nClassical vs Hierarchical Estimates:\")\n    for habitat in habitat_model.keys():\n        classical = habitat_model[habitat]['sample_mean']\n        hierarchical = habitat_model[habitat]['posterior_mean']\n        diff = hierarchical - classical\n        \n        print(f\"{habitat}: Classical = {classical:.1f}, Hierarchical = {hierarchical:.1f}, Diff = {diff:+.1f}\")\n    \n    return (\n        habitat_data,\n        habitat_model,\n        hierarchical_model_simple,\n        posterior_mean,\n        posterior_std,\n        sample_mean,\n        shrinkage_pct,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Model Comparison with Bayes Factors\n    \"\"\"\n    # Compare models using approximate Bayes factors\n    def calculate_bic_bayes_factor(model1_bic, model2_bic):\n        \"\"\"Calculate approximate Bayes Factor from BIC values\"\"\"\n        # BF_12 = exp((BIC_2 - BIC_1) / 2)\n        log_bf = (model2_bic - model1_bic) / 2\n        bf = np.exp(log_bf)\n        return bf\n    \n    def interpret_bayes_factor(bf):\n        \"\"\"Interpret Bayes Factor strength of evidence\"\"\"\n        if bf < 1:\n            bf_inv = 1 / bf\n            if bf_inv < 3:\n                return f\"Weak evidence for Model 1 (BF = 1/{bf_inv:.1f})\"\n            elif bf_inv < 10:\n                return f\"Moderate evidence for Model 1 (BF = 1/{bf_inv:.1f})\"\n            else:\n                return f\"Strong evidence for Model 1 (BF = 1/{bf_inv:.1f})\"\n        else:\n            if bf < 3:\n                return f\"Weak evidence for Model 2 (BF = {bf:.1f})\"\n            elif bf < 10:\n                return f\"Moderate evidence for Model 2 (BF = {bf:.1f})\"\n            else:\n                return f\"Strong evidence for Model 2 (BF = {bf:.1f})\"\n    \n    # Fit multiple models for species richness\n    from sklearn.linear_model import LinearRegression\n    from sklearn.metrics import mean_squared_error\n    \n    # Prepare data\n    X_full = bayesian_data[['temperature', 'log_precipitation']].values\n    X_temp_only = bayesian_data[['temperature']].values\n    X_precip_only = bayesian_data[['log_precipitation']].values\n    y = bayesian_data['species_richness'].values\n    \n    models = {\n        'Null Model': np.ones((len(y), 1)),  # Intercept only\n        'Temperature Only': X_temp_only,\n        'Precipitation Only': X_precip_only, \n        'Temperature + Precipitation': X_full\n    }\n    \n    model_results = {}\n    \n    for model_name, X in models.items():\n        if model_name == 'Null Model':\n            # Null model: just the mean\n            y_pred = np.full_like(y, y.mean())\n            n_params = 1\n        else:\n            # Fit linear regression\n            reg = LinearRegression()\n            reg.fit(X, y)\n            y_pred = reg.predict(X)\n            n_params = X.shape[1] + 1  # Parameters + intercept\n        \n        # Calculate model fit statistics\n        mse = mean_squared_error(y, y_pred)\n        n = len(y)\n        \n        # BIC = n * log(MSE) + k * log(n)\n        bic = n * np.log(mse) + n_params * np.log(n)\n        \n        # AIC = n * log(MSE) + 2 * k\n        aic = n * np.log(mse) + 2 * n_params\n        \n        model_results[model_name] = {\n            'mse': mse,\n            'aic': aic,\n            'bic': bic,\n            'n_params': n_params,\n            'predictions': y_pred\n        }\n    \n    # Display model comparison\n    print(\"Bayesian Model Comparison:\")\n    print(\"=\" * 50)\n    print(\"Model                    | MSE   | AIC   | BIC   | Params\")\n    print(\"-\" * 50)\n    \n    for model_name, results in model_results.items():\n        mse = results['mse']\n        aic = results['aic']\n        bic = results['bic']\n        n_params = results['n_params']\n        \n        print(f\"{model_name:23} | {mse:5.1f} | {aic:5.1f} | {bic:5.1f} | {n_params:6d}\")\n    \n    # Calculate Bayes Factors (using BIC approximation)\n    print(f\"\\nBayes Factors (relative to Temperature Only):\")\n    reference_bic = model_results['Temperature Only']['bic']\n    \n    for model_name, results in model_results.items():\n        if model_name != 'Temperature Only':\n            bf = calculate_bic_bayes_factor(reference_bic, results['bic'])\n            interpretation = interpret_bayes_factor(bf)\n            print(f\"{model_name}: {interpretation}\")\n    \n    # Find best model by BIC\n    best_model = min(model_results.keys(), key=lambda x: model_results[x]['bic'])\n    print(f\"\\nBest model by BIC: {best_model}\")\n    \n    return (\n        LinearRegression,\n        X_full,\n        X_precip_only,\n        X_temp_only,\n        aic,\n        best_model,\n        bic,\n        calculate_bic_bayes_factor,\n        interpret_bayes_factor,\n        mean_squared_error,\n        model_results,\n        models,\n        mse,\n        n_params,\n        reference_bic,\n        reg,\n        y,\n        y_pred,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Prior Sensitivity Analysis\n    \"\"\"\n    # Examine how results change with different priors\n    def prior_sensitivity_analysis(X, y, prior_vars=[0.1, 1, 10, 100]):\n        \"\"\"Analyze sensitivity to prior specification\"\"\"\n        sensitivity_results = {}\n        \n        for prior_var in prior_vars:\n            result = bayesian_linear_regression(X, y, prior_var=prior_var)\n            \n            sensitivity_results[prior_var] = {\n                'posterior_mean': result['posterior_mean'],\n                'posterior_std': np.sqrt(np.diag(result['posterior_cov'])),\n                'prior_var': prior_var\n            }\n        \n        return sensitivity_results\n    \n    # Run sensitivity analysis\n    sensitivity_results = prior_sensitivity_analysis(X_temp_std, y_richness)\n    \n    print(\"Prior Sensitivity Analysis:\")\n    print(\"Temperature Effect on Species Richness\")\n    print(\"=\" * 50)\n    print(\"Prior Var | Posterior Mean | Posterior Std | 95% CI\")\n    print(\"-\" * 50)\n    \n    for prior_var, result in sensitivity_results.items():\n        post_mean = result['posterior_mean'][1]  # Slope\n        post_std = result['posterior_std'][1]\n        \n        # 95% credible interval\n        ci_lower = post_mean - 1.96 * post_std\n        ci_upper = post_mean + 1.96 * post_std\n        \n        print(f\"{prior_var:8.1f}  | {post_mean:13.3f}  | {post_std:12.3f}  | [{ci_lower:.3f}, {ci_upper:.3f}]\")\n    \n    # Assess prior influence\n    print(f\"\\nPrior Influence Assessment:\")\n    \n    # Compare most informative vs least informative prior\n    strong_prior = sensitivity_results[0.1]\n    weak_prior = sensitivity_results[100]\n    \n    mean_diff = abs(strong_prior['posterior_mean'][1] - weak_prior['posterior_mean'][1])\n    std_ratio = strong_prior['posterior_std'][1] / weak_prior['posterior_std'][1]\n    \n    print(f\"Difference in posterior means: {mean_diff:.4f}\")\n    print(f\"Ratio of posterior std devs: {std_ratio:.3f}\")\n    \n    if mean_diff < 0.1 and std_ratio > 0.8:\n        print(\"Result: Data strongly dominates prior (robust conclusion)\")\n    elif mean_diff > 0.2:\n        print(\"Result: Prior has substantial influence (need more data)\")\n    else:\n        print(\"Result: Moderate prior influence (reasonable conclusion)\")\n    \n    return (\n        ci_lower,\n        ci_upper,\n        mean_diff,\n        post_mean,\n        post_std,\n        prior_sensitivity_analysis,\n        prior_var,\n        sensitivity_results,\n        std_ratio,\n        strong_prior,\n        weak_prior,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Bayesian Model Diagnostics\n    \"\"\"\n    # Check MCMC convergence and model adequacy\n    def mcmc_diagnostics(samples, parameter_names=None):\n        \"\"\"Basic MCMC diagnostics\"\"\"\n        if parameter_names is None:\n            parameter_names = [f\"Parameter {i}\" for i in range(samples.shape[1])]\n        \n        diagnostics = {}\n        \n        for i, param_name in enumerate(parameter_names):\n            param_samples = samples[:, i]\n            \n            # Effective sample size (simplified)\n            # Autocorrelation at lag 1\n            autocorr_1 = np.corrcoef(param_samples[:-1], param_samples[1:])[0, 1]\n            eff_sample_size = len(param_samples) / (1 + 2 * autocorr_1)\n            \n            # Monte Carlo standard error\n            mc_se = np.std(param_samples) / np.sqrt(eff_sample_size)\n            \n            # Geweke diagnostic (simplified)\n            # Compare first 10% to last 50% of chain\n            n_samples = len(param_samples)\n            first_part = param_samples[:n_samples//10]\n            last_part = param_samples[n_samples//2:]\n            \n            geweke_z = (np.mean(first_part) - np.mean(last_part)) / np.sqrt(\n                np.var(first_part)/len(first_part) + np.var(last_part)/len(last_part)\n            )\n            \n            diagnostics[param_name] = {\n                'mean': np.mean(param_samples),\n                'std': np.std(param_samples),\n                'autocorr_1': autocorr_1,\n                'eff_sample_size': eff_sample_size,\n                'mc_se': mc_se,\n                'geweke_z': geweke_z\n            }\n        \n        return diagnostics\n    \n    # Diagnose logistic regression MCMC\n    mcmc_diag = mcmc_diagnostics(logistic_result['samples'], ['Intercept', 'Temperature'])\n    \n    print(\"MCMC Diagnostics (Logistic Regression):\")\n    print(\"=\" * 60)\n    print(\"Parameter   | Mean   | Std    | AutoCorr | Eff.Size | MC SE\")\n    print(\"-\" * 60)\n    \n    for param_name, diag in mcmc_diag.items():\n        mean = diag['mean']\n        std = diag['std']\n        autocorr = diag['autocorr_1']\n        eff_size = diag['eff_sample_size']\n        mc_se = diag['mc_se']\n        \n        print(f\"{param_name:10} | {mean:6.3f} | {std:6.3f} | {autocorr:8.3f} | {eff_size:8.0f} | {mc_se:.4f}\")\n    \n    # Check convergence\n    print(f\"\\nConvergence Assessment:\")\n    \n    for param_name, diag in mcmc_diag.items():\n        issues = []\n        \n        if diag['eff_sample_size'] < 100:\n            issues.append(\"Low effective sample size\")\n        \n        if abs(diag['geweke_z']) > 2:\n            issues.append(\"Potential non-convergence (Geweke test)\")\n        \n        if diag['autocorr_1'] > 0.5:\n            issues.append(\"High autocorrelation\")\n        \n        if issues:\n            print(f\"{param_name}: {', '.join(issues)}\")\n        else:\n            print(f\"{param_name}: Looks good\")\n    \n    return (\n        autocorr,\n        diag,\n        eff_size,\n        mc_se,\n        mcmc_diag,\n        mcmc_diagnostics,\n        param_name,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Posterior Predictive Checks\n    \"\"\"\n    # Check if model generates realistic data\n    def posterior_predictive_check(model_result, X, y, n_sim=100):\n        \"\"\"Posterior predictive checking\"\"\"\n        # For linear regression\n        if 'posterior_mean' in model_result:\n            posterior_mean = model_result['posterior_mean']\n            posterior_cov = model_result['posterior_cov']\n            X_design = model_result['X_design']\n            \n            # Generate posterior predictive samples\n            pred_samples = []\n            \n            for _ in range(n_sim):\n                # Sample parameters from posterior\n                beta_sample = np.random.multivariate_normal(posterior_mean, posterior_cov)\n                \n                # Generate predictions\n                y_pred_mean = X_design @ beta_sample\n                \n                # Add observation error (assuming normal)\n                residual_var = np.var(y - X_design @ posterior_mean)\n                y_pred = np.random.normal(y_pred_mean, np.sqrt(residual_var))\n                \n                pred_samples.append(y_pred)\n            \n            pred_samples = np.array(pred_samples)\n            \n            # Calculate summary statistics\n            pred_mean = np.mean(pred_samples, axis=0)\n            pred_lower = np.percentile(pred_samples, 2.5, axis=0)\n            pred_upper = np.percentile(pred_samples, 97.5, axis=0)\n            \n            # Test statistics for comparison\n            obs_stats = {\n                'mean': np.mean(y),\n                'std': np.std(y),\n                'min': np.min(y),\n                'max': np.max(y),\n                'median': np.median(y)\n            }\n            \n            pred_stats = {}\n            for stat_name in obs_stats.keys():\n                if stat_name == 'mean':\n                    stat_values = [np.mean(pred_samples[i]) for i in range(n_sim)]\n                elif stat_name == 'std':\n                    stat_values = [np.std(pred_samples[i]) for i in range(n_sim)]\n                elif stat_name == 'min':\n                    stat_values = [np.min(pred_samples[i]) for i in range(n_sim)]\n                elif stat_name == 'max':\n                    stat_values = [np.max(pred_samples[i]) for i in range(n_sim)]\n                elif stat_name == 'median':\n                    stat_values = [np.median(pred_samples[i]) for i in range(n_sim)]\n                \n                pred_stats[stat_name] = {\n                    'mean': np.mean(stat_values),\n                    'p_value': np.mean(np.array(stat_values) > obs_stats[stat_name])\n                }\n            \n            return {\n                'predictions': pred_samples,\n                'pred_mean': pred_mean,\n                'pred_lower': pred_lower,\n                'pred_upper': pred_upper,\n                'observed_stats': obs_stats,\n                'predicted_stats': pred_stats\n            }\n    \n    # Perform posterior predictive check\n    ppc_result = posterior_predictive_check(\n        results['weakly_informative'], X_temp_std, y_richness\n    )\n    \n    print(\"Posterior Predictive Check:\")\n    print(\"Species Richness Model\")\n    print(\"=\" * 40)\n    print(\"Statistic | Observed | Predicted | P-value\")\n    print(\"-\" * 40)\n    \n    for stat_name in ['mean', 'std', 'min', 'max']:\n        obs_val = ppc_result['observed_stats'][stat_name]\n        pred_val = ppc_result['predicted_stats'][stat_name]['mean']\n        p_val = ppc_result['predicted_stats'][stat_name]['p_value']\n        \n        print(f\"{stat_name:8} | {obs_val:8.1f}  | {pred_val:9.1f} | {p_val:7.3f}\")\n    \n    # Interpretation\n    print(f\"\\nInterpretation:\")\n    extreme_p_values = sum(1 for stat in ppc_result['predicted_stats'].values() \n                          if stat['p_value'] < 0.05 or stat['p_value'] > 0.95)\n    \n    if extreme_p_values == 0:\n        print(\"Model fits the data well (no extreme p-values)\")\n    elif extreme_p_values <= 1:\n        print(\"Model fit is reasonable (only 1 extreme p-value)\")\n    else:\n        print(f\"Model may not fit well ({extreme_p_values} extreme p-values)\")\n    \n    return (\n        extreme_p_values,\n        obs_val,\n        p_val,\n        posterior_predictive_check,\n        ppc_result,\n        pred_val,\n        stat_name,\n    )\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Practical Bayesian Workflow\n    \"\"\"\n    # Summary of Bayesian workflow for ecological data\n    \n    workflow_steps = {\n        \"1. Model Specification\": [\n            \"Choose appropriate likelihood (normal, Poisson, binomial)\",\n            \"Specify prior distributions for parameters\",\n            \"Consider hierarchical structure if applicable\",\n            \"Include relevant covariates and interactions\"\n        ],\n        \"2. Prior Specification\": [\n            \"Use weakly informative priors when possible\",\n            \"Incorporate expert knowledge cautiously\",\n            \"Check prior predictive distributions\",\n            \"Consider prior sensitivity analysis\"\n        ],\n        \"3. Model Fitting\": [\n            \"Use appropriate computational method (MCMC, VI, etc.)\",\n            \"Check for convergence and mixing\",\n            \"Ensure adequate effective sample size\",\n            \"Monitor acceptance rates and diagnostics\"\n        ],\n        \"4. Model Checking\": [\n            \"Perform posterior predictive checks\",\n            \"Check residuals and model assumptions\",\n            \"Compare observed vs predicted patterns\",\n            \"Validate with out-of-sample data when possible\"\n        ],\n        \"5. Model Comparison\": [\n            \"Use information criteria (WAIC, LOO-CV)\",\n            \"Calculate Bayes factors when appropriate\",\n            \"Consider biological plausibility\",\n            \"Account for model uncertainty\"\n        ],\n        \"6. Interpretation\": [\n            \"Report credible intervals, not just point estimates\",\n            \"Quantify uncertainty in predictions\",\n            \"Consider practical significance\",\n            \"Communicate results clearly to stakeholders\"\n        ]\n    }\n    \n    print(\"Bayesian Workflow for Ecological Data Analysis:\")\n    print(\"=\" * 55)\n    \n    for step, guidelines in workflow_steps.items():\n        print(f\"\\n{step}:\")\n        for guideline in guidelines:\n            print(f\"  • {guideline}\")\n    \n    return workflow_steps,\n\n\n@app.cell\ndef __():\n    \"\"\"\n    ## Summary and Best Practices\n\n    In this chapter, we covered Bayesian methods for ecological data analysis:\n\n    ✓ **Bayesian fundamentals**: Prior, likelihood, and posterior concepts\n    ✓ **Bayesian regression**: Linear and logistic models with uncertainty\n    ✓ **Hierarchical models**: Accounting for group structure in data\n    ✓ **Model comparison**: Bayes factors and information criteria\n    ✓ **Prior sensitivity**: Testing robustness to prior assumptions\n    ✓ **MCMC diagnostics**: Convergence and mixing assessment\n    ✓ **Posterior predictive checks**: Model validation and adequacy\n    ✓ **Practical workflow**: Step-by-step Bayesian analysis guide\n\n    ### Key Advantages of Bayesian Methods:\n    - **Natural uncertainty quantification**: Credible intervals\n    - **Prior knowledge incorporation**: Expert knowledge and previous studies\n    - **Hierarchical modeling**: Natural for nested ecological data\n    - **Missing data handling**: Elegant treatment of incomplete observations\n    - **Small sample performance**: Better with limited data\n    - **Decision theory**: Principled approach to decision making\n\n    ### Python Packages for Bayesian Analysis:\n    - **PyMC**: Probabilistic programming and MCMC\n    - **Stan (PyStan)**: High-performance Bayesian modeling\n    - **scipy.stats**: Basic Bayesian computations\n    - **scikit-learn**: Model comparison and validation\n    - **arviz**: Exploratory analysis of Bayesian models\n\n    ### Best Practices for Ecological Bayesian Analysis:\n    1. **Start simple**: Begin with basic models, add complexity gradually\n    2. **Informative priors**: Use biological knowledge, but test sensitivity\n    3. **Model checking**: Always validate with posterior predictive checks\n    4. **Convergence**: Ensure MCMC chains have converged properly\n    5. **Communication**: Explain uncertainty clearly to stakeholders\n    6. **Reproducibility**: Document all modeling choices and assumptions\n    7. **Collaboration**: Work with domain experts on prior specification\n    8. **Validation**: Use out-of-sample data when available\n\n    ### Common Applications in Ecology:\n    - **Population modeling**: Abundance estimation with uncertainty\n    - **Species distribution modeling**: Habitat suitability with priors\n    - **Community ecology**: Hierarchical models for multiple species\n    - **Conservation planning**: Decision making under uncertainty\n    - **Climate change impacts**: Incorporating projection uncertainty\n    - **Meta-analysis**: Combining studies with varying quality\n    \"\"\"\n    \n    bayesian_summary = {\n        'Analysis Results': {\n            'Linear Model (Temperature Effect)': f\"{results['weakly_informative']['posterior_mean'][1]:.3f} ± {np.sqrt(results['weakly_informative']['posterior_cov'][1,1]):.3f}\",\n            'Logistic Model (Presence)': f\"{logistic_result['posterior_mean'][1]:.3f} ± {logistic_result['posterior_std'][1]:.3f}\",\n            'Best Model (BIC)': best_model\n        },\n        'Model Diagnostics': {\n            'MCMC Acceptance Rate': f\"{logistic_result['acceptance_rate']:.1%}\",\n            'Effective Sample Size': f\"{int(mcmc_diag['Temperature']['eff_sample_size'])}\",\n            'Convergence Issues': 'None detected' if extreme_p_values == 0 else f'{extreme_p_values} potential issues'\n        },\n        'Prior Sensitivity': {\n            'Mean Difference': f\"{mean_diff:.4f}\",\n            'Std Ratio': f\"{std_ratio:.3f}\",\n            'Conclusion': 'Robust to priors' if mean_diff < 0.1 else 'Prior sensitive'\n        },\n        'Hierarchical Effects': {\n            'Habitat Groups': len(habitat_model),\n            'Shrinkage Range': f\"{min(est['shrinkage'] for est in habitat_model.values()):.1%} - {max(est['shrinkage'] for est in habitat_model.values()):.1%}\",\n            'Grand Mean': f\"{bayesian_data['species_richness'].mean():.1f}\"\n        }\n    }\n    \n    print(\"Bayesian Analysis Summary:\")\n    print(\"=\" * 35)\n    \n    for category, details in bayesian_summary.items():\n        print(f\"\\n{category}:\")\n        for key, value in details.items():\n            print(f\"  {key}: {value}\")\n    \n    print(\"\\n✓ Chapter 7 complete! Ready for ordination and classification analysis.\")\n    \n    return bayesian_summary,\n\n\nif __name__ == \"__main__\":\n    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
