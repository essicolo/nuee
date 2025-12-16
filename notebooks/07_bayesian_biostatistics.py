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
        group_estimates = {}
        
        for group in groups:
            gdata = group_data[group]
            
            # James-Stein type shrinkage
            # More shrinkage for smaller samples and higher variance
            shrinkage_factor = gdata['n'] / (gdata['n'] + grand_var / gdata['var'])
            
            posterior_mean = (
                shrinkage_factor * gdata['mean'] + 
                (1 - shrinkage_factor) * grand_mean
            )
            
            # Posterior variance (simplified)
            posterior_var = gdata['var'] / gdata['n']
            
            group_estimates[group] = {
                'posterior_mean': posterior_mean,
                'posterior_var': posterior_var,
                'shrinkage': 1 - shrinkage_factor,
                'sample_mean': gdata['mean'],
                'sample_size': gdata['n']
            }
        
        return group_estimates, group_data
    
    # Apply hierarchical model to species richness by habitat
    habitat_model, habitat_data = hierarchical_model_simple(
        bayesian_data, 'habitat', 'species_richness'
    )
    
    print("Hierarchical Model: Species Richness by Habitat")
    print("=" * 55)
    print("Habitat   | Sample | Sample | Posterior | Posterior | Shrinkage")
    print("          |   N    |  Mean  |    Mean   |    Std    |    %")
    print("-" * 55)
    
    for habitat in ['Forest', 'Grassland', 'Wetland']:
        if habitat in habitat_model:
            est = habitat_model[habitat]
            sample_mean = est['sample_mean']
            posterior_mean = est['posterior_mean']
            posterior_std = np.sqrt(est['posterior_var'])
            shrinkage_pct = est['shrinkage'] * 100
            sample_size = est['sample_size']
            
            print(f"{habitat:9} | {sample_size:6d} | {sample_mean:6.1f} | {posterior_mean:9.1f} | {posterior_std:9.2f} | {shrinkage_pct:8.1f}")
    
    # Compare with classical (non-hierarchical) estimates
    print(f"\nClassical vs Hierarchical Estimates:")
    for habitat in habitat_model.keys():
        classical = habitat_model[habitat]['sample_mean']
        hierarchical = habitat_model[habitat]['posterior_mean']
        diff = hierarchical - classical
        
        print(f"{habitat}: Classical = {classical:.1f}, Hierarchical = {hierarchical:.1f}, Diff = {diff:+.1f}")
    
    return (
        habitat_data,
        habitat_model,
        hierarchical_model_simple,
        posterior_mean,
        posterior_std,
        sample_mean,
        shrinkage_pct,
    )


@app.cell
def __():
    """
    ## Model Comparison with Bayes Factors
    """
    # Compare models using approximate Bayes factors
    def calculate_bic_bayes_factor(model1_bic, model2_bic):
        """Calculate approximate Bayes Factor from BIC values"""
        # BF_12 = exp((BIC_2 - BIC_1) / 2)
        log_bf = (model2_bic - model1_bic) / 2
        bf = np.exp(log_bf)
        return bf
    
    def interpret_bayes_factor(bf):
        """Interpret Bayes Factor strength of evidence"""
        if bf < 1:
            bf_inv = 1 / bf
            if bf_inv < 3:
                return f"Weak evidence for Model 1 (BF = 1/{bf_inv:.1f})"
            elif bf_inv < 10:
                return f"Moderate evidence for Model 1 (BF = 1/{bf_inv:.1f})"
            else:
                return f"Strong evidence for Model 1 (BF = 1/{bf_inv:.1f})"
        else:
            if bf < 3:
                return f"Weak evidence for Model 2 (BF = {bf:.1f})"
            elif bf < 10:
                return f"Moderate evidence for Model 2 (BF = {bf:.1f})"
            else:
                return f"Strong evidence for Model 2 (BF = {bf:.1f})"
    
    # Fit multiple models for species richness
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    # Prepare data
    X_full = bayesian_data[['temperature', 'log_precipitation']].values
    X_temp_only = bayesian_data[['temperature']].values
    X_precip_only = bayesian_data[['log_precipitation']].values
    y = bayesian_data['species_richness'].values
    
    models = {
        'Null Model': np.ones((len(y), 1)),  # Intercept only
        'Temperature Only': X_temp_only,
        'Precipitation Only': X_precip_only, 
        'Temperature + Precipitation': X_full
    }
    
    model_results = {}
    
    for model_name, X in models.items():
        if model_name == 'Null Model':
            # Null model: just the mean
            y_pred = np.full_like(y, y.mean())
            n_params = 1
        else:
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            y_pred = reg.predict(X)
            n_params = X.shape[1] + 1  # Parameters + intercept
        
        # Calculate model fit statistics
        mse = mean_squared_error(y, y_pred)
        n = len(y)
        
        # BIC = n * log(MSE) + k * log(n)
        bic = n * np.log(mse) + n_params * np.log(n)
        
        # AIC = n * log(MSE) + 2 * k
        aic = n * np.log(mse) + 2 * n_params
        
        model_results[model_name] = {
            'mse': mse,
            'aic': aic,
            'bic': bic,
            'n_params': n_params,
            'predictions': y_pred
        }
    
    # Display model comparison
    print("Bayesian Model Comparison:")
    print("=" * 50)
    print("Model                    | MSE   | AIC   | BIC   | Params")
    print("-" * 50)
    
    for model_name, results in model_results.items():
        mse = results['mse']
        aic = results['aic']
        bic = results['bic']
        n_params = results['n_params']
        
        print(f"{model_name:23} | {mse:5.1f} | {aic:5.1f} | {bic:5.1f} | {n_params:6d}")
    
    # Calculate Bayes Factors (using BIC approximation)
    print(f"\nBayes Factors (relative to Temperature Only):")
    reference_bic = model_results['Temperature Only']['bic']
    
    for model_name, results in model_results.items():
        if model_name != 'Temperature Only':
            bf = calculate_bic_bayes_factor(reference_bic, results['bic'])
            interpretation = interpret_bayes_factor(bf)
            print(f"{model_name}: {interpretation}")
    
    # Find best model by BIC
    best_model = min(model_results.keys(), key=lambda x: model_results[x]['bic'])
    print(f"\nBest model by BIC: {best_model}")
    
    return (
        LinearRegression,
        X_full,
        X_precip_only,
        X_temp_only,
        aic,
        best_model,
        bic,
        calculate_bic_bayes_factor,
        interpret_bayes_factor,
        mean_squared_error,
        model_results,
        models,
        mse,
        n_params,
        reference_bic,
        reg,
        y,
        y_pred,
    )


@app.cell
def __():
    """
    ## Prior Sensitivity Analysis
    """
    # Examine how results change with different priors
    def prior_sensitivity_analysis(X, y, prior_vars=[0.1, 1, 10, 100]):
        """Analyze sensitivity to prior specification"""
        sensitivity_results = {}
        
        for prior_var in prior_vars:
            result = bayesian_linear_regression(X, y, prior_var=prior_var)
            
            sensitivity_results[prior_var] = {
                'posterior_mean': result['posterior_mean'],
                'posterior_std': np.sqrt(np.diag(result['posterior_cov'])),
                'prior_var': prior_var
            }
        
        return sensitivity_results
    
    # Run sensitivity analysis
    sensitivity_results = prior_sensitivity_analysis(X_temp_std, y_richness)
    
    print("Prior Sensitivity Analysis:")
    print("Temperature Effect on Species Richness")
    print("=" * 50)
    print("Prior Var | Posterior Mean | Posterior Std | 95% CI")
    print("-" * 50)
    
    for prior_var, result in sensitivity_results.items():
        post_mean = result['posterior_mean'][1]  # Slope
        post_std = result['posterior_std'][1]
        
        # 95% credible interval
        ci_lower = post_mean - 1.96 * post_std
        ci_upper = post_mean + 1.96 * post_std
        
        print(f"{prior_var:8.1f}  | {post_mean:13.3f}  | {post_std:12.3f}  | [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Assess prior influence
    print(f"\nPrior Influence Assessment:")
    
    # Compare most informative vs least informative prior
    strong_prior = sensitivity_results[0.1]
    weak_prior = sensitivity_results[100]
    
    mean_diff = abs(strong_prior['posterior_mean'][1] - weak_prior['posterior_mean'][1])
    std_ratio = strong_prior['posterior_std'][1] / weak_prior['posterior_std'][1]
    
    print(f"Difference in posterior means: {mean_diff:.4f}")
    print(f"Ratio of posterior std devs: {std_ratio:.3f}")
    
    if mean_diff < 0.1 and std_ratio > 0.8:
        print("Result: Data strongly dominates prior (robust conclusion)")
    elif mean_diff > 0.2:
        print("Result: Prior has substantial influence (need more data)")
    else:
        print("Result: Moderate prior influence (reasonable conclusion)")
    
    return (
        ci_lower,
        ci_upper,
        mean_diff,
        post_mean,
        post_std,
        prior_sensitivity_analysis,
        prior_var,
        sensitivity_results,
        std_ratio,
        strong_prior,
        weak_prior,
    )


@app.cell
def __():
    """
    ## Bayesian Model Diagnostics
    """
    # Check MCMC convergence and model adequacy
    def mcmc_diagnostics(samples, parameter_names=None):
        """Basic MCMC diagnostics"""
        if parameter_names is None:
            parameter_names = [f"Parameter {i}" for i in range(samples.shape[1])]
        
        diagnostics = {}
        
        for i, param_name in enumerate(parameter_names):
            param_samples = samples[:, i]
            
            # Effective sample size (simplified)
            # Autocorrelation at lag 1
            autocorr_1 = np.corrcoef(param_samples[:-1], param_samples[1:])[0, 1]
            eff_sample_size = len(param_samples) / (1 + 2 * autocorr_1)
            
            # Monte Carlo standard error
            mc_se = np.std(param_samples) / np.sqrt(eff_sample_size)
            
            # Geweke diagnostic (simplified)
            # Compare first 10% to last 50% of chain
            n_samples = len(param_samples)
            first_part = param_samples[:n_samples//10]
            last_part = param_samples[n_samples//2:]
            
            geweke_z = (np.mean(first_part) - np.mean(last_part)) / np.sqrt(
                np.var(first_part)/len(first_part) + np.var(last_part)/len(last_part)
            )
            
            diagnostics[param_name] = {
                'mean': np.mean(param_samples),
                'std': np.std(param_samples),
                'autocorr_1': autocorr_1,
                'eff_sample_size': eff_sample_size,
                'mc_se': mc_se,
                'geweke_z': geweke_z
            }
        
        return diagnostics
    
    # Diagnose logistic regression MCMC
    mcmc_diag = mcmc_diagnostics(logistic_result['samples'], ['Intercept', 'Temperature'])
    
    print("MCMC Diagnostics (Logistic Regression):")
    print("=" * 60)
    print("Parameter   | Mean   | Std    | AutoCorr | Eff.Size | MC SE")
    print("-" * 60)
    
    for param_name, diag in mcmc_diag.items():
        mean = diag['mean']
        std = diag['std']
        autocorr = diag['autocorr_1']
        eff_size = diag['eff_sample_size']
        mc_se = diag['mc_se']
        
        print(f"{param_name:10} | {mean:6.3f} | {std:6.3f} | {autocorr:8.3f} | {eff_size:8.0f} | {mc_se:.4f}")
    
    # Check convergence
    print(f"\nConvergence Assessment:")
    
    for param_name, diag in mcmc_diag.items():
        issues = []
        
        if diag['eff_sample_size'] < 100:
            issues.append("Low effective sample size")
        
        if abs(diag['geweke_z']) > 2:
            issues.append("Potential non-convergence (Geweke test)")
        
        if diag['autocorr_1'] > 0.5:
            issues.append("High autocorrelation")
        
        if issues:
            print(f"{param_name}: {', '.join(issues)}")
        else:
            print(f"{param_name}: Looks good")
    
    return (
        autocorr,
        diag,
        eff_size,
        mc_se,
        mcmc_diag,
        mcmc_diagnostics,
        param_name,
    )


@app.cell
def __():
    """
    ## Posterior Predictive Checks
    """
    # Check if model generates realistic data
    def posterior_predictive_check(model_result, X, y, n_sim=100):
        """Posterior predictive checking"""
        # For linear regression
        if 'posterior_mean' in model_result:
            posterior_mean = model_result['posterior_mean']
            posterior_cov = model_result['posterior_cov']
            X_design = model_result['X_design']
            
            # Generate posterior predictive samples
            pred_samples = []
            
            for _ in range(n_sim):
                # Sample parameters from posterior
                beta_sample = np.random.multivariate_normal(posterior_mean, posterior_cov)
                
                # Generate predictions
                y_pred_mean = X_design @ beta_sample
                
                # Add observation error (assuming normal)
                residual_var = np.var(y - X_design @ posterior_mean)
                y_pred = np.random.normal(y_pred_mean, np.sqrt(residual_var))
                
                pred_samples.append(y_pred)
            
            pred_samples = np.array(pred_samples)
            
            # Calculate summary statistics
            pred_mean = np.mean(pred_samples, axis=0)
            pred_lower = np.percentile(pred_samples, 2.5, axis=0)
            pred_upper = np.percentile(pred_samples, 97.5, axis=0)
            
            # Test statistics for comparison
            obs_stats = {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y),
                'median': np.median(y)
            }
            
            pred_stats = {}
            for stat_name in obs_stats.keys():
                if stat_name == 'mean':
                    stat_values = [np.mean(pred_samples[i]) for i in range(n_sim)]
                elif stat_name == 'std':
                    stat_values = [np.std(pred_samples[i]) for i in range(n_sim)]
                elif stat_name == 'min':
                    stat_values = [np.min(pred_samples[i]) for i in range(n_sim)]
                elif stat_name == 'max':
                    stat_values = [np.max(pred_samples[i]) for i in range(n_sim)]
                elif stat_name == 'median':
                    stat_values = [np.median(pred_samples[i]) for i in range(n_sim)]
                
                pred_stats[stat_name] = {
                    'mean': np.mean(stat_values),
                    'p_value': np.mean(np.array(stat_values) > obs_stats[stat_name])
                }
            
            return {
                'predictions': pred_samples,
                'pred_mean': pred_mean,
                'pred_lower': pred_lower,
                'pred_upper': pred_upper,
                'observed_stats': obs_stats,
                'predicted_stats': pred_stats
            }
    
    # Perform posterior predictive check
    ppc_result = posterior_predictive_check(
        results['weakly_informative'], X_temp_std, y_richness
    )
    
    print("Posterior Predictive Check:")
    print("Species Richness Model")
    print("=" * 40)
    print("Statistic | Observed | Predicted | P-value")
    print("-" * 40)
    
    for stat_name in ['mean', 'std', 'min', 'max']:
        obs_val = ppc_result['observed_stats'][stat_name]
        pred_val = ppc_result['predicted_stats'][stat_name]['mean']
        p_val = ppc_result['predicted_stats'][stat_name]['p_value']
        
        print(f"{stat_name:8} | {obs_val:8.1f}  | {pred_val:9.1f} | {p_val:7.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    extreme_p_values = sum(1 for stat in ppc_result['predicted_stats'].values() 
                          if stat['p_value'] < 0.05 or stat['p_value'] > 0.95)
    
    if extreme_p_values == 0:
        print("Model fits the data well (no extreme p-values)")
    elif extreme_p_values <= 1:
        print("Model fit is reasonable (only 1 extreme p-value)")
    else:
        print(f"Model may not fit well ({extreme_p_values} extreme p-values)")
    
    return (
        extreme_p_values,
        obs_val,
        p_val,
        posterior_predictive_check,
        ppc_result,
        pred_val,
        stat_name,
    )


@app.cell
def __():
    """
    ## Practical Bayesian Workflow
    """
    # Summary of Bayesian workflow for ecological data
    
    workflow_steps = {
        "1. Model Specification": [
            "Choose appropriate likelihood (normal, Poisson, binomial)",
            "Specify prior distributions for parameters",
            "Consider hierarchical structure if applicable",
            "Include relevant covariates and interactions"
        ],
        "2. Prior Specification": [
            "Use weakly informative priors when possible",
            "Incorporate expert knowledge cautiously",
            "Check prior predictive distributions",
            "Consider prior sensitivity analysis"
        ],
        "3. Model Fitting": [
            "Use appropriate computational method (MCMC, VI, etc.)",
            "Check for convergence and mixing",
            "Ensure adequate effective sample size",
            "Monitor acceptance rates and diagnostics"
        ],
        "4. Model Checking": [
            "Perform posterior predictive checks",
            "Check residuals and model assumptions",
            "Compare observed vs predicted patterns",
            "Validate with out-of-sample data when possible"
        ],
        "5. Model Comparison": [
            "Use information criteria (WAIC, LOO-CV)",
            "Calculate Bayes factors when appropriate",
            "Consider biological plausibility",
            "Account for model uncertainty"
        ],
        "6. Interpretation": [
            "Report credible intervals, not just point estimates",
            "Quantify uncertainty in predictions",
            "Consider practical significance",
            "Communicate results clearly to stakeholders"
        ]
    }
    
    print("Bayesian Workflow for Ecological Data Analysis:")
    print("=" * 55)
    
    for step, guidelines in workflow_steps.items():
        print(f"\n{step}:")
        for guideline in guidelines:
            print(f"  • {guideline}")
    
    return workflow_steps,


@app.cell
def __():
    """
    ## Summary and Best Practices\n
    In this chapter, we covered Bayesian methods for ecological data analysis:\n
    ✓ **Bayesian fundamentals**: Prior, likelihood, and posterior concepts
    ✓ **Bayesian regression**: Linear and logistic models with uncertainty
    ✓ **Hierarchical models**: Accounting for group structure in data
    ✓ **Model comparison**: Bayes factors and information criteria
    ✓ **Prior sensitivity**: Testing robustness to prior assumptions
    ✓ **MCMC diagnostics**: Convergence and mixing assessment
    ✓ **Posterior predictive checks**: Model validation and adequacy
    ✓ **Practical workflow**: Step-by-step Bayesian analysis guide\n
    ### Key Advantages of Bayesian Methods:
    - **Natural uncertainty quantification**: Credible intervals
    - **Prior knowledge incorporation**: Expert knowledge and previous studies
    - **Hierarchical modeling**: Natural for nested ecological data
    - **Missing data handling**: Elegant treatment of incomplete observations
    - **Small sample performance**: Better with limited data
    - **Decision theory**: Principled approach to decision making\n
    ### Python Packages for Bayesian Analysis:
    - **PyMC**: Probabilistic programming and MCMC
    - **Stan (PyStan)**: High-performance Bayesian modeling
    - **scipy.stats**: Basic Bayesian computations
    - **scikit-learn**: Model comparison and validation
    - **arviz**: Exploratory analysis of Bayesian models\n
    ### Best Practices for Ecological Bayesian Analysis:
    1. **Start simple**: Begin with basic models, add complexity gradually
    2. **Informative priors**: Use biological knowledge, but test sensitivity
    3. **Model checking**: Always validate with posterior predictive checks
    4. **Convergence**: Ensure MCMC chains have converged properly
    5. **Communication**: Explain uncertainty clearly to stakeholders
    6. **Reproducibility**: Document all modeling choices and assumptions
    7. **Collaboration**: Work with domain experts on prior specification
    8. **Validation**: Use out-of-sample data when available\n
    ### Common Applications in Ecology:
    - **Population modeling**: Abundance estimation with uncertainty
    - **Species distribution modeling**: Habitat suitability with priors
    - **Community ecology**: Hierarchical models for multiple species
    - **Conservation planning**: Decision making under uncertainty
    - **Climate change impacts**: Incorporating projection uncertainty
    - **Meta-analysis**: Combining studies with varying quality
    """
    
    bayesian_summary = {
        'Analysis Results': {
            'Linear Model (Temperature Effect)': f"{results['weakly_informative']['posterior_mean'][1]:.3f} ± {np.sqrt(results['weakly_informative']['posterior_cov'][1,1]):.3f}",
            'Logistic Model (Presence)': f"{logistic_result['posterior_mean'][1]:.3f} ± {logistic_result['posterior_std'][1]:.3f}",
            'Best Model (BIC)': best_model
        },
        'Model Diagnostics': {
            'MCMC Acceptance Rate': f"{logistic_result['acceptance_rate']:.1%}",
            'Effective Sample Size': f"{int(mcmc_diag['Temperature']['eff_sample_size'])}",
            'Convergence Issues': 'None detected' if extreme_p_values == 0 else f'{extreme_p_values} potential issues'
        },
        'Prior Sensitivity': {
            'Mean Difference': f"{mean_diff:.4f}",
            'Std Ratio': f"{std_ratio:.3f}",
            'Conclusion': 'Robust to priors' if mean_diff < 0.1 else 'Prior sensitive'
        },
        'Hierarchical Effects': {
            'Habitat Groups': len(habitat_model),
            'Shrinkage Range': f"{min(est['shrinkage'] for est in habitat_model.values()):.1%} - {max(est['shrinkage'] for est in habitat_model.values()):.1%}",
            'Grand Mean': f"{bayesian_data['species_richness'].mean():.1f}"
        }
    }
    
    print("Bayesian Analysis Summary:")
    print("=" * 35)
    
    for category, details in bayesian_summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Chapter 7 complete! Ready for ordination and classification analysis.")
    
    return bayesian_summary,


if __name__ == "__main__":
    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
