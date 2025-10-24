import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 13: Mechanistic Modeling of Ecological Systems

    This chapter covers mechanistic (process-based) modeling approaches for
    understanding ecological dynamics, including population models, predator-prey
    systems, and community dynamics using differential equations and simulation.

    ## Learning Objectives
    - Understand mechanistic vs empirical modeling approaches
    - Implement population growth and dynamics models
    - Model predator-prey and competitive interactions
    - Simulate community dynamics and ecosystem processes
    - Fit mechanistic models to ecological data
    - Understand model behavior and stability analysis
    """
    )
    return


@app.cell
def __():
    # Essential imports for mechanistic modeling
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from scipy.integrate import odeint, solve_ivp
    from scipy.optimize import minimize, differential_evolution
    import holoviews as hv
    from holoviews import opts
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Mechanistic modeling packages loaded")
    return (
        differential_evolution,
        hv,
        minimize,
        np,
        odeint,
        opts,
        pd,
        solve_ivp,
        stats,
        warnings,
    )


@app.cell
def __():
    """
    ## Introduction to Mechanistic Modeling

    **Mechanistic models** describe ecological processes using mathematical
    representations of the underlying biological mechanisms.

    ### Key Characteristics:
    - **Process-based**: Model the mechanisms, not just patterns
    - **Predictive**: Can extrapolate beyond observed conditions
    - **Mechanistic insight**: Reveal causal relationships
    - **Parameter interpretation**: Parameters have biological meaning
    - **Testable hypotheses**: Generate specific predictions

    ### Types of Mechanistic Models:
    1. **Population models**: Growth, survival, reproduction
    2. **Interaction models**: Competition, predation, mutualism
    3. **Metapopulation models**: Spatial population dynamics
    4. **Food web models**: Multi-species community dynamics
    5. **Ecosystem models**: Nutrient cycling, energy flow
    6. **Individual-based models**: Emergent population patterns

    ### Mathematical Frameworks:
    - **Differential equations**: Continuous-time dynamics
    - **Difference equations**: Discrete-time dynamics
    - **Stochastic models**: Random processes and uncertainty
    - **Spatial models**: Explicit spatial structure
    """
    return


@app.cell
def __():
    """
    ## Single Species Population Models
    """
    
    # Exponential growth model
    def exponential_growth(N, t, r):
        """
        Exponential growth: dN/dt = r * N
        N: population size
        r: intrinsic growth rate
        """
        return r * N
    
    # Logistic growth model
    def logistic_growth(N, t, r, K):
        """
        Logistic growth: dN/dt = r * N * (1 - N/K)
        N: population size
        r: intrinsic growth rate
        K: carrying capacity
        """
        return r * N * (1 - N / K)
    
    # Simulate population dynamics
    def simulate_population(model_func, N0, t_span, params):
        """Simulate population model over time"""
        t = np.linspace(0, t_span, 100)
        
        if model_func == exponential_growth:
            r = params[0]
            solution = odeint(model_func, N0, t, args=(r,))
        elif model_func == logistic_growth:
            r, K = params
            solution = odeint(model_func, N0, t, args=(r, K))
        
        return t, solution.flatten()
    
    # Example simulations
    N0 = 10  # Initial population
    t_span = 20  # Time span
    
    # Exponential growth
    r = 0.1  # Growth rate
    t_exp, N_exp = simulate_population(exponential_growth, N0, t_span, [r])
    
    # Logistic growth
    r = 0.2  # Growth rate
    K = 1000  # Carrying capacity
    t_log, N_log = simulate_population(logistic_growth, N0, t_span, [r, K])
    
    # Create comparison DataFrame
    pop_comparison = pd.DataFrame({
        'time': t_exp,
        'exponential': N_exp,
        'logistic': N_log
    })
    
    print("Population Growth Models:")
    print("=" * 30)
    print(f"Initial population: {N0}")
    print(f"Time span: {t_span} years")
    print(f"Exponential growth rate: {r}")
    print(f"Logistic growth rate: {r}, Carrying capacity: {K}")
    
    print(f"\nFinal population sizes:")
    print(f"Exponential: {N_exp[-1]:.0f}")
    print(f"Logistic: {N_log[-1]:.0f}")
    
    # Visualize population dynamics
    exp_curve = hv.Curve(pop_comparison, 'time', 'exponential', label='Exponential').opts(
        color='red', line_width=2
    )
    log_curve = hv.Curve(pop_comparison, 'time', 'logistic', label='Logistic').opts(
        color='blue', line_width=2
    )
    
    carrying_capacity_line = hv.HLine(K).opts(
        color='green', line_dash='dashed', line_width=2, alpha=0.7
    )
    
    population_plot = (exp_curve * log_curve * carrying_capacity_line).opts(
        title="Population Growth Models",
        xlabel="Time (years)",
        ylabel="Population Size",
        legend_position='top_left',
        width=600,
        height=400
    )
    
    print("\nPopulation Growth Comparison:")
    population_plot
    
    return (
        K,
        N0,
        N_exp,
        N_log,
        carrying_capacity_line,
        exp_curve,
        exponential_growth,
        log_curve,
        logistic_growth,
        pop_comparison,
        population_plot,
        r,
        simulate_population,
        t_exp,
        t_log,
        t_span,
    )


@app.cell
def __():
    """
    ## Predator-Prey Models
    """
    
    # Lotka-Volterra predator-prey model
    def lotka_volterra(y, t, alpha, beta, gamma, delta):
        """
        Lotka-Volterra predator-prey model
        y[0] = prey population (N)
        y[1] = predator population (P)
        
        dN/dt = alpha * N - beta * N * P
        dP/dt = gamma * N * P - delta * P
        
        alpha: prey growth rate
        beta: predation rate
        gamma: predator efficiency
        delta: predator death rate
        """
        N, P = y
        dNdt = alpha * N - beta * N * P
        dPdt = gamma * N * P - delta * P
        return [dNdt, dPdt]
    
    # Rosenzweig-MacArthur model (with carrying capacity)
    def rosenzweig_macarthur(y, t, r, K, a, b, c, d):
        """
        Rosenzweig-MacArthur predator-prey model
        y[0] = prey population (N)
        y[1] = predator population (P)
        
        dN/dt = r * N * (1 - N/K) - (a * N * P) / (1 + b * N)
        dP/dt = (c * a * N * P) / (1 + b * N) - d * P
        
        r: prey growth rate
        K: prey carrying capacity
        a: attack rate
        b: handling time
        c: conversion efficiency
        d: predator death rate
        """
        N, P = y
        dNdt = r * N * (1 - N/K) - (a * N * P) / (1 + b * N)
        dPdt = (c * a * N * P) / (1 + b * N) - d * P
        return [dNdt, dPdt]
    
    # Simulate predator-prey dynamics
    def simulate_predator_prey(model_func, y0, t_span, params):
        """Simulate predator-prey system"""
        t = np.linspace(0, t_span, 1000)
        
        if model_func == lotka_volterra:
            alpha, beta, gamma, delta = params
            solution = odeint(model_func, y0, t, args=(alpha, beta, gamma, delta))
        elif model_func == rosenzweig_macarthur:
            r, K, a, b, c, d = params
            solution = odeint(model_func, y0, t, args=(r, K, a, b, c, d))
        
        return t, solution[:, 0], solution[:, 1]  # time, prey, predator
    
    # Example: Lotka-Volterra system
    y0_lv = [100, 20]  # Initial prey and predator populations
    params_lv = [0.5, 0.01, 0.005, 0.2]  # alpha, beta, gamma, delta
    t_lv, prey_lv, pred_lv = simulate_predator_prey(lotka_volterra, y0_lv, 30, params_lv)
    
    # Example: Rosenzweig-MacArthur system
    y0_rm = [100, 20]  # Initial populations
    params_rm = [1.0, 500, 0.1, 0.01, 0.5, 0.1]  # r, K, a, b, c, d
    t_rm, prey_rm, pred_rm = simulate_predator_prey(rosenzweig_macarthur, y0_rm, 30, params_rm)
    
    print("Predator-Prey Models:")
    print("=" * 25)
    print("Lotka-Volterra Parameters:")
    print(f"  Prey growth rate (α): {params_lv[0]}")
    print(f"  Predation rate (β): {params_lv[1]}")
    print(f"  Predator efficiency (γ): {params_lv[2]}")
    print(f"  Predator death rate (δ): {params_lv[3]}")
    
    print("\\nRosenzweig-MacArthur Parameters:")
    print(f"  Prey growth rate (r): {params_rm[0]}")
    print(f"  Carrying capacity (K): {params_rm[1]}")
    print(f"  Attack rate (a): {params_rm[2]}")
    print(f"  Handling time (b): {params_rm[3]}")
    print(f"  Conversion efficiency (c): {params_rm[4]}")
    print(f"  Predator death rate (d): {params_rm[5]}")
    
    # Create predator-prey plots
    lv_data = pd.DataFrame({'time': t_lv, 'prey': prey_lv, 'predator': pred_lv})
    rm_data = pd.DataFrame({'time': t_rm, 'prey': prey_rm, 'predator': pred_rm})
    
    # Time series plots
    lv_prey_curve = hv.Curve(lv_data, 'time', 'prey', label='Prey').opts(color='green', line_width=2)
    lv_pred_curve = hv.Curve(lv_data, 'time', 'predator', label='Predator').opts(color='red', line_width=2)
    lv_plot = (lv_prey_curve * lv_pred_curve).opts(
        title="Lotka-Volterra Dynamics",
        xlabel="Time",
        ylabel="Population Size",
        legend_position='top_right'
    )
    
    rm_prey_curve = hv.Curve(rm_data, 'time', 'prey', label='Prey').opts(color='green', line_width=2)
    rm_pred_curve = hv.Curve(rm_data, 'time', 'predator', label='Predator').opts(color='red', line_width=2)
    rm_plot = (rm_prey_curve * rm_pred_curve).opts(
        title="Rosenzweig-MacArthur Dynamics",
        xlabel="Time",
        ylabel="Population Size",
        legend_position='top_right'
    )
    
    # Phase plots (predator vs prey)
    lv_phase = hv.Curve(lv_data, 'prey', 'predator').opts(
        title="Lotka-Volterra Phase Plot",
        xlabel="Prey Population",
        ylabel="Predator Population",
        color='blue'
    )
    
    rm_phase = hv.Curve(rm_data, 'prey', 'predator').opts(
        title="Rosenzweig-MacArthur Phase Plot",
        xlabel="Prey Population", 
        ylabel="Predator Population",
        color='purple'
    )
    
    print("\\nPredator-Prey Dynamics:")
    (lv_plot + rm_plot + lv_phase + rm_phase).cols(2)
    
    return (
        lv_data,
        lv_phase,
        lv_plot,
        lv_pred_curve,
        lv_prey_curve,
        lotka_volterra,
        params_lv,
        params_rm,
        pred_lv,
        pred_rm,
        prey_lv,
        prey_rm,
        rm_data,
        rm_phase,
        rm_plot,
        rm_pred_curve,
        rm_prey_curve,
        rosenzweig_macarthur,
        simulate_predator_prey,
        t_lv,
        t_rm,
        y0_lv,
        y0_rm,
    )


@app.cell
def __():
    """
    ## Competition Models
    """
    
    # Lotka-Volterra competition model
    def lotka_volterra_competition(y, t, r1, r2, K1, K2, alpha12, alpha21):
        """
        Two-species competition model
        y[0] = species 1 population (N1)
        y[1] = species 2 population (N2)
        
        dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        
        r1, r2: growth rates
        K1, K2: carrying capacities
        alpha12: effect of species 2 on species 1
        alpha21: effect of species 1 on species 2
        """
        N1, N2 = y
        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return [dN1dt, dN2dt]
    
    # Simulate competition scenarios
    def simulate_competition(params, y0, t_span, scenario_name):
        """Simulate competition with given parameters"""
        t = np.linspace(0, t_span, 500)
        r1, r2, K1, K2, alpha12, alpha21 = params
        
        solution = odeint(lotka_volterra_competition, y0, t, 
                         args=(r1, r2, K1, K2, alpha12, alpha21))
        
        return {
            'scenario': scenario_name,
            'time': t,
            'species1': solution[:, 0],
            'species2': solution[:, 1],
            'params': params
        }
    
    # Different competition scenarios
    y0_comp = [50, 50]  # Initial populations
    t_comp = 50
    
    scenarios = {
        'Coexistence': [0.5, 0.4, 1000, 800, 0.3, 0.2],  # Weak competition
        'Species 1 Wins': [0.6, 0.4, 1000, 800, 0.2, 0.8],  # Species 1 superior
        'Species 2 Wins': [0.4, 0.6, 800, 1000, 0.8, 0.2],  # Species 2 superior
        'Competitive Exclusion': [0.5, 0.5, 1000, 1000, 1.2, 1.1]  # Strong competition
    }
    
    competition_results = {}
    for scenario_name, params in scenarios.items():
        competition_results[scenario_name] = simulate_competition(
            params, y0_comp, t_comp, scenario_name
        )
    
    print("Competition Model Scenarios:")
    print("=" * 35)
    
    for scenario, result in competition_results.items():
        params = result['params']
        final_n1 = result['species1'][-1]
        final_n2 = result['species2'][-1]
        
        print(f"\\n{scenario}:")
        print(f"  α₁₂ = {params[4]:.1f}, α₂₁ = {params[5]:.1f}")
        print(f"  Final N₁: {final_n1:.0f}, Final N₂: {final_n2:.0f}")
        
        # Determine outcome
        if final_n1 > 10 and final_n2 > 10:
            outcome = "Coexistence"
        elif final_n1 > 10:
            outcome = "Species 1 dominates"
        elif final_n2 > 10:
            outcome = "Species 2 dominates"
        else:
            outcome = "Both extinct"
        
        print(f"  Outcome: {outcome}")
    
    # Visualize competition scenarios
    competition_plots = []
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (scenario, result) in enumerate(competition_results.items()):
        comp_data = pd.DataFrame({
            'time': result['time'],
            'species1': result['species1'],
            'species2': result['species2']
        })
        
        sp1_curve = hv.Curve(comp_data, 'time', 'species1', label='Species 1').opts(
            color='blue', line_width=2
        )
        sp2_curve = hv.Curve(comp_data, 'time', 'species2', label='Species 2').opts(
            color='red', line_width=2, line_dash='dashed'
        )
        
        comp_plot = (sp1_curve * sp2_curve).opts(
            title=scenario,
            xlabel="Time",
            ylabel="Population Size",
            legend_position='top_right',
            width=400,
            height=300
        )
        
        competition_plots.append(comp_plot)
    
    print("\\nCompetition Dynamics:")
    (competition_plots[0] + competition_plots[1] + 
     competition_plots[2] + competition_plots[3]).cols(2)
    
    return (
        colors,
        comp_data,
        comp_plot,
        competition_plots,
        competition_results,
        final_n1,
        final_n2,
        lotka_volterra_competition,
        outcome,
        params,
        result,
        scenario,
        scenarios,
        simulate_competition,
        sp1_curve,
        sp2_curve,
        t_comp,
        y0_comp,
    )


@app.cell
def __():
    """
    ## Metapopulation Models
    """
    
    # Classic metapopulation model (Levins model)
    def levins_metapopulation(p, t, c, e):
        """
        Levins metapopulation model
        dp/dt = c * p * (1 - p) - e * p
        
        p: proportion of patches occupied
        c: colonization rate
        e: extinction rate
        """
        return c * p * (1 - p) - e * p
    
    # Spatially explicit metapopulation model (simplified)
    def simulate_spatial_metapopulation(n_patches=20, colonization_prob=0.1, 
                                      extinction_prob=0.05, t_steps=100):
        """
        Simulate spatially explicit metapopulation
        """
        # Initialize patches (0 = empty, 1 = occupied)
        patches = np.random.binomial(1, 0.3, n_patches)  # Start with 30% occupied
        
        # Store time series
        time_series = []
        occupied_patches = []
        
        for t in range(t_steps):
            # Record current state
            time_series.append(t)
            occupied_patches.append(np.sum(patches))
            
            # Create new patch states
            new_patches = patches.copy()
            
            for i in range(n_patches):
                if patches[i] == 1:
                    # Occupied patch: may go extinct
                    if np.random.random() < extinction_prob:
                        new_patches[i] = 0
                else:
                    # Empty patch: may be colonized
                    # Colonization probability depends on nearby occupied patches
                    # Simplified: probability increases with total occupied patches
                    local_colonization_prob = colonization_prob * np.sum(patches) / n_patches
                    if np.random.random() < local_colonization_prob:
                        new_patches[i] = 1
            
            patches = new_patches
        
        return np.array(time_series), np.array(occupied_patches)
    
    # Simulate Levins model
    p0 = 0.3  # Initial proportion occupied
    t_meta = np.linspace(0, 50, 200)
    
    # Different parameter combinations
    meta_scenarios = {
        'Persistence': [0.2, 0.1],  # c > e
        'Decline': [0.1, 0.2],     # c < e
        'Marginal': [0.15, 0.14]   # c ≈ e
    }
    
    levins_results = {}
    for scenario, (c, e) in meta_scenarios.items():
        p_trajectory = odeint(levins_metapopulation, p0, t_meta, args=(c, e))
        levins_results[scenario] = {
            'time': t_meta,
            'proportion': p_trajectory.flatten(),
            'c': c,
            'e': e,
            'equilibrium': (c - e) / c if c > e else 0
        }
    
    # Simulate spatial metapopulation
    np.random.seed(42)
    spatial_time, spatial_occupied = simulate_spatial_metapopulation(
        n_patches=25, colonization_prob=0.15, extinction_prob=0.08, t_steps=100
    )
    
    print("Metapopulation Models:")
    print("=" * 25)
    print("Levins Model Results:")
    
    for scenario, result in levins_results.items():
        c, e = result['c'], result['e']
        final_p = result['proportion'][-1]
        equilibrium = result['equilibrium']
        
        print(f"\\n{scenario}:")
        print(f"  Colonization rate (c): {c}")
        print(f"  Extinction rate (e): {e}")
        print(f"  Predicted equilibrium: {equilibrium:.3f}")
        print(f"  Final proportion: {final_p:.3f}")
    
    print(f"\\nSpatial Metapopulation:")
    print(f"  Number of patches: 25")
    print(f"  Final occupied patches: {spatial_occupied[-1]}")
    print(f"  Final proportion: {spatial_occupied[-1]/25:.2f}")
    
    # Visualize metapopulation dynamics
    levins_plots = []
    
    for scenario, result in levins_results.items():
        meta_data = pd.DataFrame({
            'time': result['time'],
            'proportion': result['proportion']
        })
        
        levins_curve = hv.Curve(meta_data, 'time', 'proportion').opts(
            title=f"Levins Model: {scenario}",
            xlabel="Time",
            ylabel="Proportion Occupied",
            line_width=2
        )
        
        # Add equilibrium line
        eq_line = hv.HLine(result['equilibrium']).opts(
            color='red', line_dash='dashed', alpha=0.7
        )
        
        levins_plots.append(levins_curve * eq_line)
    
    # Spatial metapopulation plot
    spatial_data = pd.DataFrame({
        'time': spatial_time,
        'occupied': spatial_occupied,
        'proportion': spatial_occupied / 25
    })
    
    spatial_plot = hv.Curve(spatial_data, 'time', 'occupied').opts(
        title="Spatial Metapopulation",
        xlabel="Time Steps",
        ylabel="Occupied Patches",
        line_width=2,
        color='green'
    )
    
    print("\\nMetapopulation Dynamics:")
    (levins_plots[0] + levins_plots[1] + levins_plots[2] + spatial_plot).cols(2)
    
    return (
        c,
        e,
        eq_line,
        equilibrium,
        final_p,
        levins_curve,
        levins_metapopulation,
        levins_plots,
        levins_results,
        meta_data,
        meta_scenarios,
        p0,
        p_trajectory,
        scenario,
        simulate_spatial_metapopulation,
        spatial_data,
        spatial_occupied,
        spatial_plot,
        spatial_time,
        t_meta,
    )


@app.cell
def __():
    """
    ## Parameter Estimation and Model Fitting
    """
    
    # Generate synthetic data for parameter estimation
    def generate_population_data(model_type='logistic', true_params=None, noise_level=0.1, n_points=20):
        """Generate synthetic population data with noise"""
        np.random.seed(42)
        
        if model_type == 'logistic':
            r_true, K_true = true_params if true_params else [0.3, 1000]
            t_data = np.linspace(0, 20, n_points)
            N0 = 50
            
            # Generate true trajectory
            N_true = K_true / (1 + ((K_true - N0) / N0) * np.exp(-r_true * t_data))
            
            # Add noise
            noise = np.random.normal(0, noise_level * np.mean(N_true), len(N_true))
            N_observed = N_true + noise
            N_observed = np.maximum(N_observed, 1)  # Ensure positive
            
            return t_data, N_observed, N_true, [r_true, K_true]
    
    # Objective function for parameter estimation
    def logistic_objective(params, t_data, N_observed):
        """Objective function for logistic model fitting"""
        r, K = params
        
        # Prevent unrealistic parameters
        if r <= 0 or K <= 0 or K > 10000:
            return 1e10
        
        try:
            N0 = N_observed[0]
            N_predicted = K / (1 + ((K - N0) / N0) * np.exp(-r * t_data))
            
            # Sum of squared residuals
            ssr = np.sum((N_observed - N_predicted)**2)
            return ssr
            
        except (OverflowError, ZeroDivisionError):
            return 1e10
    
    # Fit model to data
    def fit_population_model(t_data, N_data, model_type='logistic'):
        """Fit population model to data using optimization"""
        
        if model_type == 'logistic':
            # Initial parameter guess
            K_guess = max(N_data) * 1.2  # Slightly above maximum observed
            r_guess = 0.1
            initial_guess = [r_guess, K_guess]
            
            # Parameter bounds
            bounds = [(0.001, 2), (max(N_data), 10000)]
            
            # Optimization
            result = minimize(logistic_objective, initial_guess, 
                            args=(t_data, N_data), bounds=bounds,
                            method='L-BFGS-B')
            
            if result.success:
                r_fit, K_fit = result.x
                
                # Calculate fitted trajectory
                N0 = N_data[0]
                t_fine = np.linspace(t_data.min(), t_data.max(), 100)
                N_fit = K_fit / (1 + ((K_fit - N0) / N0) * np.exp(-r_fit * t_fine))
                
                # Calculate R-squared
                N_pred_data = K_fit / (1 + ((K_fit - N0) / N0) * np.exp(-r_fit * t_data))
                ss_res = np.sum((N_data - N_pred_data)**2)
                ss_tot = np.sum((N_data - np.mean(N_data))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                return {
                    'success': True,
                    'params': [r_fit, K_fit],
                    'r_squared': r_squared,
                    't_fit': t_fine,
                    'N_fit': N_fit,
                    'objective_value': result.fun
                }
            else:
                return {'success': False, 'message': result.message}
    
    # Generate example data and fit model
    t_obs, N_obs, N_true_traj, true_params = generate_population_data(
        'logistic', [0.25, 800], noise_level=0.15
    )
    
    # Fit model
    fit_result = fit_population_model(t_obs, N_obs, 'logistic')
    
    print("Parameter Estimation Results:")
    print("=" * 35)
    print(f"True parameters: r = {true_params[0]}, K = {true_params[1]}")
    
    if fit_result['success']:
        r_est, K_est = fit_result['params']
        print(f"Estimated parameters: r = {r_est:.3f}, K = {K_est:.1f}")
        print(f"Parameter errors: r error = {abs(r_est - true_params[0]):.3f}, K error = {abs(K_est - true_params[1]):.1f}")
        print(f"R-squared: {fit_result['r_squared']:.3f}")
        print(f"Objective value (SSR): {fit_result['objective_value']:.1f}")
        
        # Create fitting visualization
        fitting_data = pd.DataFrame({
            'time_obs': t_obs,
            'N_observed': N_obs,
            'N_true': N_true_traj
        })
        
        fit_data = pd.DataFrame({
            'time_fit': fit_result['t_fit'],
            'N_fitted': fit_result['N_fit']
        })
        
        # Plot observed data, true trajectory, and fitted model
        observed_points = hv.Scatter(fitting_data, 'time_obs', 'N_observed', label='Observed').opts(
            color='red', size=8, alpha=0.8
        )
        
        true_curve = hv.Curve(fitting_data, 'time_obs', 'N_true', label='True Model').opts(
            color='blue', line_width=3, alpha=0.7
        )
        
        fitted_curve = hv.Curve(fit_data, 'time_fit', 'N_fitted', label='Fitted Model').opts(
            color='green', line_width=2, line_dash='dashed'
        )
        
        fitting_plot = (observed_points * true_curve * fitted_curve).opts(
            title="Population Model Fitting",
            xlabel="Time",
            ylabel="Population Size",
            legend_position='bottom_right',
            width=600,
            height=400
        )
        
        print("\\nModel Fitting Visualization:")
        fitting_plot
        
    else:
        print(f"Fitting failed: {fit_result['message']}")
    
    return (
        K_est,
        K_fit,
        K_guess,
        N0,
        N_fit,
        N_obs,
        N_pred_data,
        N_true_traj,
        bounds,
        fit_data,
        fit_population_model,
        fit_result,
        fitted_curve,
        fitting_data,
        fitting_plot,
        generate_population_data,
        initial_guess,
        logistic_objective,
        observed_points,
        r_est,
        r_fit,
        r_guess,
        r_squared,
        ss_res,
        ss_tot,
        t_fine,
        t_obs,
        true_curve,
        true_params,
    )


@app.cell
def __():
    """
    ## Stability Analysis
    """
    
    # Analyze equilibrium points and stability
    def analyze_equilibria(model_type, params):
        """Analyze equilibrium points and their stability"""
        
        if model_type == 'logistic':
            r, K = params
            # Logistic model has two equilibria: N* = 0 (unstable) and N* = K (stable)
            equilibria = [0, K]
            stability = ['Unstable', 'Stable'] if r > 0 else ['Stable', 'Unstable']
            
            return {
                'equilibria': equilibria,
                'stability': stability,
                'interpretation': f"Carrying capacity equilibrium at N* = {K}"
            }
            
        elif model_type == 'predator_prey':
            alpha, beta, gamma, delta = params
            # Lotka-Volterra equilibrium
            N_eq = delta / gamma if gamma > 0 else 0
            P_eq = alpha / beta if beta > 0 else 0
            
            return {
                'equilibria': [(0, 0), (N_eq, P_eq)],
                'stability': ['Saddle point', 'Neutrally stable (center)'],
                'interpretation': f"Coexistence equilibrium at N* = {N_eq:.1f}, P* = {P_eq:.1f}"
            }
    
    # Phase plane analysis for two-species systems
    def phase_plane_analysis(model_func, params, x_range, y_range, n_points=15):
        """Create phase plane plot with vector field"""
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate direction vectors
        if model_func == lotka_volterra:
            alpha, beta, gamma, delta = params
            DX = alpha * X - beta * X * Y
            DY = gamma * X * Y - delta * Y
        elif model_func == lotka_volterra_competition:
            r1, r2, K1, K2, alpha12, alpha21 = params
            DX = r1 * X * (1 - (X + alpha12 * Y) / K1)
            DY = r2 * Y * (1 - (Y + alpha21 * X) / K2)
        
        # Normalize vectors for better visualization
        M = np.sqrt(DX**2 + DY**2)
        M[M == 0] = 1  # Avoid division by zero
        DX_norm = DX / M
        DY_norm = DY / M
        
        return X, Y, DX_norm, DY_norm
    
    # Analyze different model types
    print("Stability Analysis:")
    print("=" * 20)
    
    # Logistic model
    logistic_analysis = analyze_equilibria('logistic', [0.2, 1000])
    print("\\nLogistic Model:")
    print(f"Equilibria: {logistic_analysis['equilibria']}")
    print(f"Stability: {logistic_analysis['stability']}")
    print(f"Interpretation: {logistic_analysis['interpretation']}")
    
    # Predator-prey model
    pp_analysis = analyze_equilibria('predator_prey', [0.5, 0.01, 0.005, 0.2])
    print("\\nPredator-Prey Model:")
    print(f"Equilibria: {pp_analysis['equilibria']}")
    print(f"Stability: {pp_analysis['stability']}")
    print(f"Interpretation: {pp_analysis['interpretation']}")
    
    # Create phase plane plot for competition model
    comp_params = [0.5, 0.4, 1000, 800, 0.3, 0.2]  # Coexistence scenario
    X_comp, Y_comp, DX_comp, DY_comp = phase_plane_analysis(
        lotka_volterra_competition, comp_params, [0, 1200], [0, 1000]
    )
    
    # Convert to format for holoviews
    vector_data = []
    for i in range(X_comp.shape[0]):
        for j in range(X_comp.shape[1]):
            vector_data.append({
                'x': X_comp[i, j],
                'y': Y_comp[i, j],
                'dx': DX_comp[i, j] * 50,  # Scale for visibility
                'dy': DY_comp[i, j] * 50
            })
    
    vector_df = pd.DataFrame(vector_data)
    
    # Create phase plane visualization
    # Note: This is a simplified representation - proper vector fields would need custom plotting
    vector_plot = hv.Scatter(vector_df, 'x', 'y').opts(
        title="Competition Model Phase Plane",
        xlabel="Species 1 Population",
        ylabel="Species 2 Population",
        size=2,
        alpha=0.6,
        width=500,
        height=400
    )
    
    # Add nullclines (simplified)
    # Species 1 nullcline: N1 = K1 - alpha12 * N2
    r1, r2, K1, K2, alpha12, alpha21 = comp_params
    y_null1 = np.linspace(0, 1000, 100)
    x_null1 = K1 - alpha12 * y_null1
    x_null1 = np.maximum(x_null1, 0)  # Keep positive
    
    # Species 2 nullcline: N2 = K2 - alpha21 * N1
    x_null2 = np.linspace(0, 1200, 100)
    y_null2 = K2 - alpha21 * x_null2
    y_null2 = np.maximum(y_null2, 0)  # Keep positive
    
    nullcline_data1 = pd.DataFrame({'x': x_null1, 'y': y_null1})
    nullcline_data2 = pd.DataFrame({'x': x_null2, 'y': y_null2})
    
    nullcline1 = hv.Curve(nullcline_data1, 'x', 'y', label='Species 1 nullcline').opts(
        color='blue', line_width=3
    )
    nullcline2 = hv.Curve(nullcline_data2, 'x', 'y', label='Species 2 nullcline').opts(
        color='red', line_width=3
    )
    
    phase_plane_plot = (vector_plot * nullcline1 * nullcline2).opts(
        legend_position='top_right'
    )
    
    print("\\nPhase Plane Analysis:")
    phase_plane_plot
    
    return (
        DX,
        DX_comp,
        DX_norm,
        DY,
        DY_comp,
        DY_norm,
        K1,
        K2,
        M,
        N_eq,
        P_eq,
        X,
        X_comp,
        Y,
        Y_comp,
        alpha12,
        alpha21,
        analyze_equilibria,
        comp_params,
        logistic_analysis,
        nullcline1,
        nullcline2,
        nullcline_data1,
        nullcline_data2,
        phase_plane_analysis,
        phase_plane_plot,
        pp_analysis,
        r1,
        r2,
        vector_data,
        vector_df,
        vector_plot,
        x_null1,
        x_null2,
        y_null1,
        y_null2,
    )


@app.cell
def __():
    """
    ## Model Selection and Comparison
    """
    
    # Compare different population models
    def compare_population_models(t_data, N_data):
        """Compare multiple population models"""
        
        models = {}
        
        # 1. Exponential model
        def exponential_objective(params, t_data, N_data):
            r = params[0]
            if r <= 0:
                return 1e10
            
            N0 = N_data[0]
            N_pred = N0 * np.exp(r * t_data)
            return np.sum((N_data - N_pred)**2)
        
        # Fit exponential
        result_exp = minimize(exponential_objective, [0.1], args=(t_data, N_data),
                             bounds=[(0.001, 2)], method='L-BFGS-B')
        
        if result_exp.success:
            r_exp = result_exp.x[0]
            N0 = N_data[0]
            N_pred_exp = N0 * np.exp(r_exp * t_data)
            
            # Calculate AIC
            ssr_exp = result_exp.fun
            n = len(N_data)
            k_exp = 1  # Number of parameters
            aic_exp = n * np.log(ssr_exp / n) + 2 * k_exp
            
            models['Exponential'] = {
                'params': [r_exp],
                'ssr': ssr_exp,
                'aic': aic_exp,
                'predictions': N_pred_exp,
                'param_names': ['r']
            }
        
        # 2. Logistic model (already implemented)
        logistic_fit = fit_population_model(t_data, N_data, 'logistic')
        
        if logistic_fit['success']:
            ssr_log = logistic_fit['objective_value']
            k_log = 2  # Number of parameters
            aic_log = n * np.log(ssr_log / n) + 2 * k_log
            
            r_log, K_log = logistic_fit['params']
            N0 = N_data[0]
            N_pred_log = K_log / (1 + ((K_log - N0) / N0) * np.exp(-r_log * t_data))
            
            models['Logistic'] = {
                'params': [r_log, K_log],
                'ssr': ssr_log,
                'aic': aic_log,
                'predictions': N_pred_log,
                'param_names': ['r', 'K']
            }
        
        # 3. Gompertz model
        def gompertz_objective(params, t_data, N_data):
            r, K = params
            if r <= 0 or K <= 0:
                return 1e10
            
            N0 = N_data[0]
            try:
                N_pred = K * np.exp(np.log(N0/K) * np.exp(-r * t_data))
                return np.sum((N_data - N_pred)**2)
            except (OverflowError, ZeroDivisionError):
                return 1e10
        
        # Fit Gompertz
        K_guess = max(N_data) * 1.2
        result_gomp = minimize(gompertz_objective, [0.1, K_guess], args=(t_data, N_data),
                              bounds=[(0.001, 2), (max(N_data), 10000)], method='L-BFGS-B')
        
        if result_gomp.success:
            r_gomp, K_gomp = result_gomp.x
            N0 = N_data[0]
            N_pred_gomp = K_gomp * np.exp(np.log(N0/K_gomp) * np.exp(-r_gomp * t_data))
            
            ssr_gomp = result_gomp.fun
            k_gomp = 2
            aic_gomp = n * np.log(ssr_gomp / n) + 2 * k_gomp
            
            models['Gompertz'] = {
                'params': [r_gomp, K_gomp],
                'ssr': ssr_gomp,
                'aic': aic_gomp,
                'predictions': N_pred_gomp,
                'param_names': ['r', 'K']
            }
        
        return models
    
    # Compare models using the synthetic data
    model_comparison = compare_population_models(t_obs, N_obs)
    
    print("Model Comparison Results:")
    print("=" * 30)
    print("Model      | Parameters | SSR    | AIC    | Δ AIC")
    print("-" * 50)
    
    # Calculate AIC differences
    min_aic = min(model['aic'] for model in model_comparison.values())
    
    for model_name, results in model_comparison.items():
        params_str = ', '.join([f"{name}={val:.3f}" for name, val in 
                               zip(results['param_names'], results['params'])])
        ssr = results['ssr']
        aic = results['aic']
        delta_aic = aic - min_aic
        
        print(f"{model_name:10} | {params_str:10} | {ssr:6.1f} | {aic:6.1f} | {delta_aic:5.1f}")
    
    # Find best model
    best_model = min(model_comparison.keys(), key=lambda x: model_comparison[x]['aic'])
    print(f"\\nBest model: {best_model}")
    
    # Model selection interpretation
    print("\\nModel Selection Interpretation:")
    print("Δ AIC < 2: Substantial support")
    print("Δ AIC 4-7: Considerably less support")
    print("Δ AIC > 10: Essentially no support")
    
    # Visualize model comparison
    comparison_data = pd.DataFrame({
        'time': t_obs,
        'observed': N_obs
    })
    
    comparison_plots = [hv.Scatter(comparison_data, 'time', 'observed', label='Observed').opts(
        color='black', size=8, alpha=0.8
    )]
    
    colors = ['red', 'blue', 'green']
    for i, (model_name, results) in enumerate(model_comparison.items()):
        model_data = pd.DataFrame({
            'time': t_obs,
            'predictions': results['predictions']
        })
        
        model_curve = hv.Curve(model_data, 'time', 'predictions', label=model_name).opts(
            color=colors[i % len(colors)], line_width=2
        )
        comparison_plots.append(model_curve)
    
    # Combine all plots
    model_comparison_plot = comparison_plots[0]
    for plot in comparison_plots[1:]:
        model_comparison_plot *= plot
    
    model_comparison_plot = model_comparison_plot.opts(
        title="Population Model Comparison",
        xlabel="Time",
        ylabel="Population Size",
        legend_position='bottom_right',
        width=600,
        height=400
    )
    
    print("\\nModel Comparison Visualization:")
    model_comparison_plot
    
    return (
        aic_exp,
        aic_gomp,
        aic_log,
        best_model,
        colors,
        comparison_data,
        comparison_plots,
        delta_aic,
        exponential_objective,
        gompertz_objective,
        k_exp,
        k_gomp,
        k_log,
        logistic_fit,
        min_aic,
        model_comparison,
        model_comparison_plot,
        model_curve,
        model_data,
        n,
        params_str,
        r_exp,
        r_gomp,
        result_exp,
        result_gomp,
        ssr_exp,
        ssr_gomp,
        ssr_log,
    )


@app.cell
def __():
    """
    ## Summary and Best Practices

    In this chapter, we covered mechanistic modeling approaches for ecological systems:

    ✓ **Population models**: Exponential and logistic growth dynamics
    ✓ **Predator-prey models**: Lotka-Volterra and Rosenzweig-MacArthur systems
    ✓ **Competition models**: Two-species competition and coexistence
    ✓ **Metapopulation models**: Levins model and spatial dynamics
    ✓ **Parameter estimation**: Fitting mechanistic models to data
    ✓ **Stability analysis**: Equilibrium points and phase plane analysis
    ✓ **Model comparison**: AIC-based model selection

    ### Key Advantages of Mechanistic Models:
    - **Biological realism**: Parameters have clear biological interpretation
    - **Predictive power**: Can extrapolate beyond observed conditions
    - **Hypothesis testing**: Test specific mechanisms and processes
    - **Management insight**: Understand leverage points for intervention
    - **Theoretical understanding**: Connect to ecological theory

    ### Python Tools for Mechanistic Modeling:
    - **scipy.integrate**: Differential equation solvers (odeint, solve_ivp)
    - **scipy.optimize**: Parameter estimation and model fitting
    - **numpy**: Numerical computations and array operations
    - **pandas**: Data manipulation and organization
    - **holoviews**: Dynamic visualization of model behavior

    ### Best Practices for Mechanistic Modeling:
    1. **Start simple**: Begin with basic models, add complexity gradually
    2. **Biological justification**: Ensure mechanisms are biologically reasonable
    3. **Parameter constraints**: Use realistic bounds and prior knowledge
    4. **Model validation**: Test predictions against independent data
    5. **Sensitivity analysis**: Understand parameter importance and uncertainty
    6. **Stability analysis**: Check for realistic equilibrium behavior
    7. **Model comparison**: Compare alternative mechanisms formally
    8. **Uncertainty quantification**: Propagate parameter uncertainty to predictions

    ### Common Applications in Ecology:
    - **Population viability analysis**: Extinction risk assessment
    - **Harvest management**: Sustainable yield calculations
    - **Biological control**: Predator-prey dynamics in pest management
    - **Conservation planning**: Metapopulation persistence and connectivity
    - **Climate change impacts**: Species responses to environmental change
    - **Ecosystem restoration**: Recovery dynamics and intervention timing
    - **Disease ecology**: Pathogen transmission and control strategies
    """
    
    mechanistic_summary = {
        'Population Models': {
            'Exponential Growth Rate': f"{r:.2f} year⁻¹",
            'Logistic Carrying Capacity': f"{K:.0f} individuals",
            'Logistic Growth Rate': f"{logistic_fit['params'][0]:.3f} year⁻¹" if logistic_fit['success'] else "N/A"
        },
        'Predator-Prey Dynamics': {
            'Lotka-Volterra Cycles': "Neutrally stable oscillations",
            'Rosenzweig-MacArthur': "Stable limit cycles or equilibrium",
            'Phase Space Behavior': "Species abundance trajectories in phase plane"
        },
        'Competition Outcomes': {
            'Coexistence Scenario': f"α₁₂ = {scenarios['Coexistence'][4]}, α₂₁ = {scenarios['Coexistence'][5]}",
            'Exclusion Scenario': f"Strong interspecific competition",
            'Equilibrium Analysis': "Nullclines determine coexistence conditions"
        },
        'Metapopulation Dynamics': {
            'Levins Equilibrium': f"p* = (c-e)/c when c > e",
            'Spatial Structure': "Local colonization-extinction balance",
            'Persistence Threshold': "Colonization rate must exceed extinction rate"
        },
        'Model Fitting': {
            'Best Model': best_model,
            'Parameter Estimation': "Least squares optimization",
            'Model Selection': "AIC-based comparison",
            'Goodness of Fit': f"R² = {fit_result['r_squared']:.3f}" if fit_result['success'] else "N/A"
        }
    }
    
    print("Mechanistic Modeling Summary:")
    print("=" * 35)
    
    for category, details in mechanistic_summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Chapter 13 complete! Mathematical Ecology with Python book finished!")
    print("\n🎉 Congratulations! You now have a complete 14-chapter book covering:")
    print("   Mathematical ecology from basic Python to advanced mechanistic modeling")
    
    return mechanistic_summary,


if __name__ == "__main__":
    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
