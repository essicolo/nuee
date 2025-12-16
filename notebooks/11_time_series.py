import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Chapter 11: Time Series Analysis for Ecological Data

    This chapter covers temporal analysis methods for ecological data, including
    trend detection, seasonal decomposition, and forecasting of ecological time series.

    ## Learning Objectives
    - Understand temporal patterns in ecological data
    - Perform time series decomposition and trend analysis
    - Apply autocorrelation and spectral analysis
    - Build forecasting models for ecological variables
    - Handle irregular and missing temporal data
    """
    )
    return


@app.cell
def __():
    # Essential imports for time series analysis
    import pandas as pd
    import numpy as np
    import holoviews as hv
    from holoviews import opts
    import scipy.stats as stats
    from scipy import signal
    import warnings
    warnings.filterwarnings('ignore')
    
    hv.extension('bokeh')
    
    print("✓ Time series analysis packages loaded")
    return hv, np, opts, pd, signal, stats, warnings


@app.cell
def __():
    """
    ## Create Ecological Time Series Data
    """
    # Generate realistic ecological time series
    np.random.seed(42)
    
    # Create daily time series for 5 years
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2023-12-31')
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Time components
    day_of_year = dates.dayofyear
    year = dates.year
    
    # Seasonal patterns
    # Temperature with annual cycle
    annual_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
    temp_noise = np.random.normal(0, 2, n_days)
    # Add climate change trend
    climate_trend = 0.02 * (year - 2019) * 365 + 0.02 * np.arange(n_days)
    temperature = annual_temp + temp_noise + climate_trend
    
    # Precipitation with seasonal variation and random events
    seasonal_precip = 5 + 3 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/3)
    # Add random precipitation events
    rain_events = np.random.exponential(0.5, n_days)
    precipitation = seasonal_precip + rain_events
    
    # Bird observations with complex patterns
    # Base level varies with temperature and season
    temp_effect = (temperature - temperature.mean()) / temperature.std()
    seasonal_bird = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365.25) + 5 * temp_effect
    
    # Migration peaks (spring and fall)
    migration_spring = 15 * np.exp(-((day_of_year - 120) / 20)**2)
    migration_fall = 12 * np.exp(-((day_of_year - 280) / 25)**2)
    
    # Weekly pattern (weekend effect)
    weekday_effect = 3 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Population decline trend
    decline_trend = -0.01 * np.arange(n_days)
    
    # Random noise
    bird_noise = np.random.normal(0, 3, n_days)
    
    bird_count = (seasonal_bird + migration_spring + migration_fall + 
                 weekday_effect + decline_trend + bird_noise)
    bird_count = np.maximum(bird_count, 0)  # No negative counts
    
    # Plant phenology (flowering dates)
    flowering_peak = 130 + np.random.normal(0, 10, len(dates))  # Around May 10
    flowering_intensity = 50 * np.exp(-((day_of_year - flowering_peak) / 15)**2)
    flowering_intensity = np.maximum(flowering_intensity, 0)
    
    # Create DataFrame
    ts_data = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'precipitation': precipitation,
        'bird_count': bird_count,
        'flowering_intensity': flowering_intensity,
        'day_of_year': day_of_year,
        'year': year,
        'month': dates.month,
        'weekday': dates.weekday
    })
    
    # Set date as index
    ts_data.set_index('date', inplace=True)
    
    print(f"Time series dataset created: {ts_data.shape}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Number of days: {n_days}")
    
    print("\nTime series summary:")
    print(ts_data.describe())
    
    return (
        annual_temp,
        bird_count,
        bird_noise,
        climate_trend,
        dates,
        day_of_year,
        decline_trend,
        end_date,
        flowering_intensity,
        flowering_peak,
        migration_fall,
        migration_spring,
        n_days,
        precipitation,
        rain_events,
        seasonal_bird,
        seasonal_precip,
        start_date,
        temp_effect,
        temp_noise,
        temperature,
        ts_data,
        weekday_effect,
        year,
    )


@app.cell
def __():
    """
    ## Basic Time Series Exploration
    """
    # Basic visualization
    temp_plot = hv.Curve(ts_data.reset_index(), 'date', 'temperature').opts(
        title="Temperature Time Series",
        xlabel="Date",
        ylabel="Temperature (°C)",
        width=800,
        height=300
    )
    
    bird_plot = hv.Curve(ts_data.reset_index(), 'date', 'bird_count').opts(
        title="Bird Count Time Series", 
        xlabel="Date",
        ylabel="Number of Birds",
        width=800,
        height=300
    )
    
    print("Time Series Visualizations:")
    temp_plot + bird_plot
    
    # Basic statistics
    print("\nBasic Time Series Statistics:")
    
    for variable in ['temperature', 'bird_count', 'precipitation']:
        series = ts_data[variable]
        
        print(f"\n{variable.title()}:")
        print(f"  Mean: {series.mean():.2f}")
        print(f"  Std: {series.std():.2f}")
        print(f"  Min: {series.min():.2f}")
        print(f"  Max: {series.max():.2f}")
        print(f"  Trend (linear): {stats.linregress(np.arange(len(series)), series).slope:.4f} per day")
    
    return bird_plot, series, temp_plot, variable


@app.cell
def __():
    """
    ## Trend Analysis
    """
    def calculate_trend(series):
        """Calculate linear trend in time series"""
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        trend_line = slope * x + intercept
        return slope, intercept, r_value, p_value, trend_line
    
    # Analyze trends in key variables
    trend_results = {}
    
    for var in ['temperature', 'bird_count', 'precipitation']:
        slope, intercept, r_value, p_value, trend_line = calculate_trend(ts_data[var])
        
        # Convert daily trend to annual trend
        annual_trend = slope * 365.25
        
        trend_results[var] = {
            'daily_slope': slope,
            'annual_trend': annual_trend,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_line': trend_line
        }
        
        print(f"\nTrend Analysis - {var.title()}:")
        print(f"  Annual trend: {annual_trend:.3f} units/year")
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            direction = "increasing" if annual_trend > 0 else "decreasing"
            print(f"  Significant {direction} trend detected")
        else:
            print(f"  No significant trend detected")
    
    # Mann-Kendall trend test (non-parametric)
    def mann_kendall_test(data):
        """
        Perform Mann-Kendall trend test
        """
        n = len(data)
        s = 0
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return s, z, p_value
    
    print("\nMann-Kendall Trend Tests:")
    for var in ['temperature', 'bird_count']:
        s, z, p_mk = mann_kendall_test(ts_data[var].values)
        print(f"{var.title()}: S = {s}, Z = {z:.3f}, p = {p_mk:.4f}")
    
    return calculate_trend, mann_kendall_test, p_mk, s, trend_results, var, z


@app.cell
def __():
    """
    ## Seasonal Decomposition
    """
    def seasonal_decompose(series, period=365):
        """
        Simple seasonal decomposition using moving averages
        """
        # Calculate trend using centered moving average
        trend = series.rolling(window=period, center=True).mean()
        
        # Calculate seasonal component
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index.dayofyear).mean()
        
        # Repeat seasonal pattern for all years
        seasonal_series = pd.Series(index=series.index, dtype=float)
        for i, date in enumerate(series.index):
            day_of_year = date.dayofyear
            # Handle leap years
            if day_of_year == 366:
                day_of_year = 365
            seasonal_series.iloc[i] = seasonal[day_of_year]
        
        # Calculate residuals
        residual = series - trend - seasonal_series
        
        return trend, seasonal_series, residual
    
    # Decompose temperature time series
    temp_trend, temp_seasonal, temp_residual = seasonal_decompose(ts_data['temperature'])
    
    print("Seasonal Decomposition - Temperature:")
    print(f"Original variance: {ts_data['temperature'].var():.2f}")
    print(f"Trend variance: {temp_trend.var():.2f}")
    print(f"Seasonal variance: {temp_seasonal.var():.2f}")
    print(f"Residual variance: {temp_residual.var():.2f}")
    
    # Decompose bird count time series
    bird_trend, bird_seasonal, bird_residual = seasonal_decompose(ts_data['bird_count'])
    
    print("\nSeasonal Decomposition - Bird Count:")
    print(f"Original variance: {ts_data['bird_count'].var():.2f}")
    print(f"Trend variance: {bird_trend.var():.2f}")
    print(f"Seasonal variance: {bird_seasonal.var():.2f}")
    print(f"Residual variance: {bird_residual.var():.2f}")
    
    # Create decomposition plots
    decomp_data = pd.DataFrame({
        'date': ts_data.index,
        'original': ts_data['bird_count'],
        'trend': bird_trend,
        'seasonal': bird_seasonal,
        'residual': bird_residual
    }).reset_index(drop=True)
    
    # Plot decomposition components
    original_plot = hv.Curve(decomp_data, 'date', 'original').opts(
        title="Original", ylabel="Bird Count", width=600, height=150
    )
    trend_plot = hv.Curve(decomp_data, 'date', 'trend').opts(
        title="Trend", ylabel="Trend", width=600, height=150
    )
    seasonal_plot = hv.Curve(decomp_data, 'date', 'seasonal').opts(
        title="Seasonal", ylabel="Seasonal", width=600, height=150
    )
    residual_plot = hv.Curve(decomp_data, 'date', 'residual').opts(
        title="Residual", ylabel="Residual", width=600, height=150
    )
    
    print("\nSeasonal Decomposition Plots:")
    (original_plot + trend_plot + seasonal_plot + residual_plot).cols(1)
    
    return (
        bird_residual,
        bird_seasonal,
        bird_trend,
        decomp_data,
        original_plot,
        residual_plot,
        seasonal_decompose,
        seasonal_plot,
        temp_residual,
        temp_seasonal,
        temp_trend,
        trend_plot,
    )


@app.cell
def __():
    """
    ## Autocorrelation Analysis
    """
    def calculate_autocorrelation(series, max_lag=50):
        """Calculate autocorrelation function"""
        n = len(series)
        series_centered = series - series.mean()
        
        autocorr = np.correlate(series_centered, series_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        return autocorr[:max_lag + 1]
    
    def calculate_partial_autocorr(series, max_lag=50):
        """Calculate partial autocorrelation function using Yule-Walker equations"""
        autocorr = calculate_autocorrelation(series, max_lag)
        partial_autocorr = np.zeros(max_lag + 1)
        partial_autocorr[0] = 1.0
        
        for k in range(1, max_lag + 1):
            if k == 1:
                partial_autocorr[k] = autocorr[1]
            else:
                # Solve Yule-Walker equations
                r = autocorr[1:k+1]
                R = np.array([[autocorr[abs(i-j)] for j in range(k)] for i in range(k)])
                try:
                    phi = np.linalg.solve(R, r)
                    partial_autocorr[k] = phi[-1]
                except np.linalg.LinAlgError:
                    partial_autocorr[k] = 0
        
        return partial_autocorr
    
    # Calculate autocorrelations for bird counts
    bird_acf = calculate_autocorrelation(ts_data['bird_count'], max_lag=100)
    bird_pacf = calculate_partial_autocorr(ts_data['bird_count'], max_lag=30)
    
    # Calculate autocorrelations for temperature
    temp_acf = calculate_autocorrelation(ts_data['temperature'], max_lag=100)
    
    print("Autocorrelation Analysis:")
    print(f"Bird count ACF at lag 1: {bird_acf[1]:.3f}")
    print(f"Bird count ACF at lag 7: {bird_acf[7]:.3f} (weekly pattern)")
    print(f"Bird count ACF at lag 365: {bird_acf[365] if len(bird_acf) > 365 else 'N/A'}")
    
    print(f"\nTemperature ACF at lag 1: {temp_acf[1]:.3f}")
    print(f"Temperature ACF at lag 365: {temp_acf[365] if len(temp_acf) > 365 else 'N/A'}")
    
    # Find significant lags (simplified - would use confidence intervals in practice)
    significant_lags = np.where(np.abs(bird_acf) > 0.1)[0]
    print(f"\nSignificant ACF lags for bird count: {significant_lags[:10]}")  # First 10
    
    # Test for white noise (Ljung-Box test simulation)
    def ljung_box_test(residuals, lags=10):
        """Simplified Ljung-Box test for autocorrelation"""
        n = len(residuals)
        acf = calculate_autocorrelation(residuals, lags)
        
        lb_stat = n * (n + 2) * np.sum([(acf[i]**2) / (n - i) for i in range(1, lags + 1)])
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return lb_stat, p_value
    
    # Test residuals from seasonal decomposition
    lb_stat, lb_p = ljung_box_test(bird_residual.dropna(), lags=10)
    print(f"\nLjung-Box test on residuals: LB = {lb_stat:.2f}, p = {lb_p:.4f}")
    
    if lb_p < 0.05:
        print("Residuals show significant autocorrelation")
    else:
        print("Residuals appear to be white noise")
    
    return (
        bird_acf,
        bird_pacf,
        calculate_autocorrelation,
        calculate_partial_autocorr,
        lb_p,
        lb_stat,
        ljung_box_test,
        significant_lags,
        temp_acf,
    )


@app.cell
def __():
    """
    ## Spectral Analysis
    """
    # Perform Fourier analysis to identify dominant frequencies
    def spectral_analysis(series, sampling_rate=1):
        """Perform spectral analysis using FFT"""
        # Remove trend
        detrended = signal.detrend(series.dropna())
        
        # Apply window to reduce spectral leakage
        windowed = detrended * signal.windows.hann(len(detrended))
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(windowed, fs=sampling_rate)
        
        # Convert frequency to period (in days)
        periods = 1 / freqs[1:]  # Exclude zero frequency
        psd = psd[1:]
        
        return periods, psd
    
    # Spectral analysis of bird counts
    bird_periods, bird_psd = spectral_analysis(ts_data['bird_count'])
    
    # Find dominant periods
    peak_indices = signal.find_peaks(bird_psd, height=np.percentile(bird_psd, 95))[0]
    dominant_periods = bird_periods[peak_indices]
    
    print("Spectral Analysis - Bird Count:")
    print("Dominant periods (days):")
    for period in sorted(dominant_periods):
        if period < 1000:  # Focus on periods less than 3 years
            print(f"  {period:.1f} days ({period/365.25:.2f} years)")
    
    # Expected periods
    expected_periods = [7, 365.25]  # Weekly and annual
    print(f"\nExpected periods: {expected_periods}")
    
    # Check if expected periods are found
    for expected in expected_periods:
        closest_found = dominant_periods[np.argmin(np.abs(dominant_periods - expected))]
        if abs(closest_found - expected) / expected < 0.1:  # Within 10%
            print(f"  {expected:.1f} day period detected: {closest_found:.1f} days")
        else:
            print(f"  {expected:.1f} day period not clearly detected")
    
    # Cross-spectral analysis between temperature and bird count
    def cross_spectral_analysis(series1, series2):
        """Calculate coherence between two time series"""
        # Remove NaN values
        valid_idx = ~(np.isnan(series1) | np.isnan(series2))
        s1 = series1[valid_idx]
        s2 = series2[valid_idx]
        
        # Calculate coherence
        freqs, coherence = signal.coherence(s1, s2, fs=1, nperseg=min(256, len(s1)//4))
        periods = 1 / freqs[1:]
        coherence = coherence[1:]
        
        return periods, coherence
    
    temp_bird_periods, temp_bird_coherence = cross_spectral_analysis(
        ts_data['temperature'].values, ts_data['bird_count'].values
    )
    
    # Find high coherence periods
    high_coherence = temp_bird_coherence > 0.7
    coherent_periods = temp_bird_periods[high_coherence]
    
    print(f"\nCoherence Analysis (Temperature vs Bird Count):")
    print(f"Periods with high coherence (>0.7):")
    for period in coherent_periods:
        if period < 1000:
            print(f"  {period:.1f} days")
    
    return (
        bird_periods,
        bird_psd,
        coherent_periods,
        cross_spectral_analysis,
        dominant_periods,
        expected_periods,
        high_coherence,
        peak_indices,
        spectral_analysis,
        temp_bird_coherence,
        temp_bird_periods,
    )


@app.cell
def __():
    """
    ## Forecasting Models
    """
    # Simple forecasting methods
    def naive_forecast(series, steps=30):
        """Naive forecast: last observation repeated"""
        last_value = series.iloc[-1]
        return np.full(steps, last_value)
    
    def seasonal_naive_forecast(series, steps=30, season_length=365):
        """Seasonal naive: repeat same season from previous year"""
        forecast = np.zeros(steps)
        for i in range(steps):
            season_idx = (len(series) + i) % season_length
            if season_idx < len(series):
                forecast[i] = series.iloc[-(season_length - season_idx)]
            else:
                forecast[i] = series.iloc[-1]  # fallback
        return forecast
    
    def linear_trend_forecast(series, steps=30):
        """Linear trend extrapolation"""
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series) + steps)
        return slope * future_x + intercept
    
    def exponential_smoothing(series, alpha=0.3, steps=30):
        """Simple exponential smoothing"""
        smoothed = np.zeros(len(series))
        smoothed[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            smoothed[i] = alpha * series.iloc[i] + (1 - alpha) * smoothed[i-1]
        
        # Forecast
        forecast = np.full(steps, smoothed[-1])
        return forecast
    
    # Split data for forecasting validation
    train_size = int(0.8 * len(ts_data))
    train_data = ts_data.iloc[:train_size]
    test_data = ts_data.iloc[train_size:]
    
    forecast_steps = len(test_data)
    
    print(f"Forecasting Setup:")
    print(f"Training period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"Forecast steps: {forecast_steps}")
    
    # Generate forecasts for bird count
    forecasts = {
        'Naive': naive_forecast(train_data['bird_count'], forecast_steps),
        'Seasonal Naive': seasonal_naive_forecast(train_data['bird_count'], forecast_steps),
        'Linear Trend': linear_trend_forecast(train_data['bird_count'], forecast_steps),
        'Exponential Smoothing': exponential_smoothing(train_data['bird_count'], steps=forecast_steps)
    }
    
    # Calculate forecast errors
    actual = test_data['bird_count'].values
    forecast_errors = {}
    
    for method, forecast in forecasts.items():
        mae = np.mean(np.abs(forecast - actual))
        mse = np.mean((forecast - actual)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        forecast_errors[method] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # Display forecast performance
    print(f"\nForecast Performance (Bird Count):")
    print("Method              | MAE   | RMSE  | MAPE (%)")
    print("-" * 45)
    
    for method, errors in forecast_errors.items():
        print(f"{method:18} | {errors['MAE']:5.2f} | {errors['RMSE']:5.2f} | {errors['MAPE']:8.1f}")
    
    # Find best method
    best_method = min(forecast_errors.keys(), key=lambda x: forecast_errors[x]['RMSE'])
    print(f"\nBest method by RMSE: {best_method}")
    
    return (
        actual,
        best_method,
        exponential_smoothing,
        forecast_errors,
        forecast_steps,
        forecasts,
        linear_trend_forecast,
        naive_forecast,
        seasonal_naive_forecast,
        test_data,
        train_data,
        train_size,
    )


@app.cell
def __():
    """
    ## Advanced Time Series Methods
    """
    # Moving averages for smoothing
    def calculate_moving_averages(series, windows=[7, 30, 365]):
        """Calculate multiple moving averages"""
        ma_dict = {}
        for window in windows:
            ma_dict[f'MA_{window}'] = series.rolling(window=window, center=True).mean()
        return ma_dict
    
    # Calculate moving averages for bird count
    bird_ma = calculate_moving_averages(ts_data['bird_count'])
    
    print("Moving Averages Analysis:")
    for ma_name, ma_series in bird_ma.items():
        variance_reduction = 1 - (ma_series.var() / ts_data['bird_count'].var())
        print(f"{ma_name}: Variance reduction = {variance_reduction:.1%}")
    
    # Change point detection (simplified)
    def detect_change_points(series, window=30):
        """Simple change point detection using rolling statistics"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Detect points where rolling statistics change significantly
        mean_changes = np.abs(rolling_mean.diff()) > 2 * rolling_mean.std()
        std_changes = np.abs(rolling_std.diff()) > 2 * rolling_std.std()
        
        change_points = mean_changes | std_changes
        return change_points
    
    # Detect change points in bird count
    change_points = detect_change_points(ts_data['bird_count'])
    change_dates = ts_data.index[change_points]
    
    print(f"\nChange Point Detection:")
    print(f"Number of potential change points: {change_points.sum()}")
    
    if len(change_dates) > 0:
        print("First few change points:")
        for date in change_dates[:5]:
            print(f"  {date.date()}")
    
    # Anomaly detection using z-scores
    def detect_anomalies(series, threshold=3):
        """Detect anomalies using z-score method"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        anomalies = z_scores > threshold
        return anomalies
    
    # Detect anomalies in residuals
    bird_anomalies = detect_anomalies(bird_residual.dropna())
    anomaly_dates = bird_residual.dropna().index[bird_anomalies]
    
    print(f"\nAnomaly Detection (Residuals):")
    print(f"Number of anomalies detected: {bird_anomalies.sum()}")
    print(f"Anomaly rate: {bird_anomalies.mean():.1%}")
    
    if len(anomaly_dates) > 0:
        print("Recent anomalies:")
        for date in anomaly_dates[-3:]:
            value = bird_residual.loc[date]
            print(f"  {date.date()}: residual = {value:.2f}")
    
    return (
        anomaly_dates,
        bird_anomalies,
        bird_ma,
        calculate_moving_averages,
        change_dates,
        change_points,
        detect_anomalies,
        detect_change_points,
        ma_dict,
        ma_name,
        ma_series,
        rolling_mean,
        rolling_std,
        variance_reduction,
    )


@app.cell
def __():
    """
    ## Ecological Interpretation
    """
    # Phenology analysis
    def analyze_phenology(series, threshold_percentile=50):
        """Analyze seasonal timing (phenology)"""
        annual_data = {}
        
        for year in series.index.year.unique():
            year_data = series[series.index.year == year]
            
            if len(year_data) > 300:  # Sufficient data for the year
                # Find peak timing
                peak_date = year_data.idxmax()
                peak_value = year_data.max()
                
                # Find start and end of season (above threshold)
                threshold = np.percentile(year_data, threshold_percentile)
                above_threshold = year_data > threshold
                
                if above_threshold.any():
                    season_start = year_data[above_threshold].index.min()
                    season_end = year_data[above_threshold].index.max()
                    season_length = (season_end - season_start).days
                else:
                    season_start = season_end = peak_date
                    season_length = 0
                
                annual_data[year] = {
                    'peak_date': peak_date,
                    'peak_doy': peak_date.dayofyear,
                    'peak_value': peak_value,
                    'season_start': season_start,
                    'season_end': season_end,
                    'season_length': season_length
                }
        
        return annual_data
    
    # Analyze flowering phenology
    flowering_phenology = analyze_phenology(ts_data['flowering_intensity'])
    
    print("Flowering Phenology Analysis:")
    print("Year | Peak DOY | Peak Value | Season Length")
    print("-" * 45)
    
    for year, data in flowering_phenology.items():
        print(f"{year} |   {data['peak_doy']:3d}    |   {data['peak_value']:6.1f}   |     {data['season_length']:3d}")
    
    # Calculate phenology trends
    if len(flowering_phenology) > 2:
        years = list(flowering_phenology.keys())
        peak_doys = [flowering_phenology[year]['peak_doy'] for year in years]
        season_lengths = [flowering_phenology[year]['season_length'] for year in years]
        
        # Trend in peak timing
        peak_slope, peak_intercept, peak_r, peak_p, _ = stats.linregress(years, peak_doys)
        
        # Trend in season length
        length_slope, length_intercept, length_r, length_p, _ = stats.linregress(years, season_lengths)
        
        print(f"\nPhenology Trends:")
        print(f"Peak timing trend: {peak_slope:.2f} days/year (p = {peak_p:.3f})")
        print(f"Season length trend: {length_slope:.2f} days/year (p = {length_p:.3f})")
        
        if peak_p < 0.05:
            direction = "earlier" if peak_slope < 0 else "later"
            print(f"Flowering is becoming significantly {direction}")
    
    # Population dynamics analysis
    def analyze_population_dynamics(series):
        """Analyze population trends and volatility"""
        # Annual means
        annual_means = series.groupby(series.index.year).mean()
        
        # Population growth rates
        growth_rates = annual_means.pct_change() * 100
        
        # Volatility (coefficient of variation)
        cv = series.std() / series.mean() * 100
        
        # Long-term trend
        years = annual_means.index.values
        trend_slope, _, trend_r, trend_p, _ = stats.linregress(years, annual_means.values)
        
        return {
            'annual_means': annual_means,
            'growth_rates': growth_rates,
            'volatility': cv,
            'trend_slope': trend_slope,
            'trend_r_squared': trend_r**2,
            'trend_p_value': trend_p
        }
    
    # Analyze bird population dynamics
    bird_dynamics = analyze_population_dynamics(ts_data['bird_count'])
    
    print(f"\nBird Population Dynamics:")
    print(f"Volatility (CV): {bird_dynamics['volatility']:.1f}%")
    print(f"Long-term trend: {bird_dynamics['trend_slope']:.2f} birds/year")
    print(f"Trend R-squared: {bird_dynamics['trend_r_squared']:.3f}")
    print(f"Trend p-value: {bird_dynamics['trend_p_value']:.3f}")
    
    print(f"\nAnnual growth rates:")
    for year, rate in bird_dynamics['growth_rates'].items():
        if not np.isnan(rate):
            print(f"{year}: {rate:+.1f}%")
    
    return (
        analyze_phenology,
        analyze_population_dynamics,
        annual_means,
        bird_dynamics,
        cv,
        flowering_phenology,
        growth_rates,
        length_intercept,
        length_p,
        length_r,
        length_slope,
        peak_doys,
        peak_intercept,
        peak_p,
        peak_r,
        peak_slope,
        season_lengths,
        trend_p,
        trend_r,
        trend_slope,
        years,
    )


@app.cell
def __():
    """
    ## Climate Change Impact Analysis
    """
    # Temperature trend analysis
    def analyze_climate_trends(temp_series):
        """Analyze climate trends and extremes"""
        # Annual statistics
        annual_stats = temp_series.groupby(temp_series.index.year).agg([
            'mean', 'min', 'max', 'std'
        ])
        annual_stats.columns = ['mean_temp', 'min_temp', 'max_temp', 'temp_variability']
        
        # Calculate trends
        years = annual_stats.index.values
        
        trends = {}
        for col in annual_stats.columns:
            slope, _, r_val, p_val, _ = stats.linregress(years, annual_stats[col])
            trends[col] = {
                'slope': slope,
                'r_squared': r_val**2,
                'p_value': p_val
            }
        
        # Extreme events analysis
        # Heat days (temperature > 95th percentile)
        heat_threshold = np.percentile(temp_series, 95)
        heat_days = temp_series > heat_threshold
        annual_heat_days = heat_days.groupby(heat_days.index.year).sum()
        
        # Cold days (temperature < 5th percentile)
        cold_threshold = np.percentile(temp_series, 5)
        cold_days = temp_series < cold_threshold
        annual_cold_days = cold_days.groupby(cold_days.index.year).sum()
        
        return {
            'annual_stats': annual_stats,
            'trends': trends,
            'heat_threshold': heat_threshold,
            'annual_heat_days': annual_heat_days,
            'cold_threshold': cold_threshold,
            'annual_cold_days': annual_cold_days
        }
    
    climate_analysis = analyze_climate_trends(ts_data['temperature'])
    
    print("Climate Change Analysis:")
    print("=" * 30)
    
    for metric, trend_data in climate_analysis['trends'].items():
        annual_change = trend_data['slope']
        p_value = trend_data['p_value']
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Annual change: {annual_change:+.3f} °C/year")
        print(f"  P-value: {p_value:.3f}")
        
        if p_value < 0.05:
            direction = "increasing" if annual_change > 0 else "decreasing"
            print(f"  Significant {direction} trend detected")
    
    # Extreme events trends
    heat_years = climate_analysis['annual_heat_days'].index.values
    heat_trend_slope, _, heat_trend_r, heat_trend_p, _ = stats.linregress(
        heat_years, climate_analysis['annual_heat_days'].values
    )
    
    print(f"\nExtreme Events:")
    print(f"Heat threshold: {climate_analysis['heat_threshold']:.1f}°C")
    print(f"Heat days trend: {heat_trend_slope:+.2f} days/year (p = {heat_trend_p:.3f})")
    
    # Species-climate relationships
    def analyze_species_climate_relationships(species_series, climate_series):
        """Analyze how species respond to climate variables"""
        # Monthly correlations
        monthly_corr = []
        
        for month in range(1, 13):
            species_month = species_series[species_series.index.month == month]
            climate_month = climate_series[climate_series.index.month == month]
            
            if len(species_month) > 10 and len(climate_month) > 10:
                # Ensure same dates
                common_dates = species_month.index.intersection(climate_month.index)
                if len(common_dates) > 10:
                    corr, p_val = stats.pearsonr(
                        species_month.loc[common_dates],
                        climate_month.loc[common_dates]
                    )
                    monthly_corr.append({
                        'month': month,
                        'correlation': corr,
                        'p_value': p_val
                    })
        
        return monthly_corr
    
    # Analyze bird-temperature relationships
    bird_temp_corr = analyze_species_climate_relationships(
        ts_data['bird_count'], ts_data['temperature']
    )
    
    print(f"\nBird-Temperature Relationships:")
    print("Month | Correlation | P-value")
    print("-" * 30)
    
    for corr_data in bird_temp_corr:
        month_name = pd.to_datetime(f"2020-{corr_data['month']:02d}-01").strftime('%b')
        corr = corr_data['correlation']
        p_val = corr_data['p_value']
        significance = "*" if p_val < 0.05 else " "
        
        print(f"{month_name:5} | {corr:10.3f} | {p_val:7.3f} {significance}")
    
    return (
        analyze_climate_trends,
        analyze_species_climate_relationships,
        annual_change,
        bird_temp_corr,
        climate_analysis,
        cold_days,
        cold_threshold,
        corr,
        heat_days,
        heat_threshold,
        heat_trend_p,
        heat_trend_r,
        heat_trend_slope,
        heat_years,
        month_name,
        monthly_corr,
        p_val,
        significance,
    )


@app.cell
def __():
    """
    ## Summary and Best Practices\n
    In this chapter, we covered comprehensive time series analysis for ecological data:\n
    ✓ **Time series exploration**: Basic patterns and visualization
    ✓ **Trend analysis**: Linear trends and Mann-Kendall tests
    ✓ **Seasonal decomposition**: Separating trend, seasonal, and residual components
    ✓ **Autocorrelation analysis**: ACF, PACF, and temporal dependencies
    ✓ **Spectral analysis**: Frequency domain analysis and coherence
    ✓ **Forecasting**: Multiple methods and performance evaluation
    ✓ **Advanced methods**: Moving averages, change points, anomaly detection
    ✓ **Ecological interpretation**: Phenology and population dynamics
    ✓ **Climate impact analysis**: Long-term trends and extreme events\n
    ### Key Python Packages for Time Series:
    - **pandas**: Time series data manipulation and resampling
    - **scipy.stats**: Statistical tests and trend analysis
    - **scipy.signal**: Spectral analysis and filtering
    - **numpy**: Numerical computations and array operations
    - **holoviews**: Interactive time series visualization\n
    ### Best Practices for Ecological Time Series:
    1. **Data quality**: Check for missing values, outliers, and measurement errors
    2. **Temporal resolution**: Match analysis to biological processes
    3. **Stationarity**: Test and account for non-stationary behavior
    4. **Seasonality**: Always consider seasonal patterns in ecology
    5. **Autocorrelation**: Account for temporal dependencies in statistical tests
    6. **Multiple scales**: Analyze daily, seasonal, and long-term patterns
    7. **External drivers**: Include climate and environmental variables
    8. **Validation**: Use out-of-sample testing for forecasting models\n
    ### Common Ecological Time Series Applications:
    - **Population monitoring**: Abundance trends and dynamics
    - **Phenology studies**: Timing of biological events
    - **Climate impact assessment**: Species responses to environmental change
    - **Conservation planning**: Trend-based risk assessment
    - **Ecosystem monitoring**: Multi-species community dynamics
    - **Early warning systems**: Anomaly detection for management
    """
    
    ts_summary = {
        'Dataset Characteristics': {
            'Duration': f"{(ts_data.index[-1] - ts_data.index[0]).days} days",
            'Variables': list(ts_data.columns[:4]),
            'Temporal Resolution': 'Daily'
        },
        'Key Findings': {
            'Temperature Trend': f"{climate_analysis['trends']['mean_temp']['slope']:.3f} °C/year",
            'Bird Population Trend': f"{bird_dynamics['trend_slope']:.2f} birds/year",
            'Dominant Periods': f"{sorted(dominant_periods)[:3]} days"
        },
        'Forecasting Performance': {
            'Best Method': best_method,
            'RMSE': f"{forecast_errors[best_method]['RMSE']:.2f}",
            'MAPE': f"{forecast_errors[best_method]['MAPE']:.1f}%"
        },
        'Anomalies Detected': {
            'Change Points': f"{change_points.sum()}",
            'Residual Anomalies': f"{bird_anomalies.sum()}",
            'Anomaly Rate': f"{bird_anomalies.mean():.1%}"
        }
    }
    
    print("Time Series Analysis Summary:")
    print("=" * 40)
    
    for category, details in ts_summary.items():
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Chapter 11 complete! Ready for spatial data analysis.")
    
    return ts_summary,


if __name__ == "__main__":
    app.run()

@app.cell
def _():
    import marimo as mo
    return (mo,)
