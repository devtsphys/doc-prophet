# Python Prophet Reference Card

## Table of Contents

1. [Installation & Setup](#installation--setup)
1. [Basic Usage](#basic-usage)
1. [Core Components](#core-components)
1. [Advanced Features](#advanced-features)
1. [Feature Engineering](#feature-engineering)
1. [Model Evaluation & Diagnostics](#model-evaluation--diagnostics)
1. [Hyperparameter Tuning](#hyperparameter-tuning)
1. [Best Practices](#best-practices)
1. [Common Pitfalls](#common-pitfalls)
1. [Example Workflows](#example-workflows)

## Installation & Setup

```python
# Installing Prophet
pip install prophet

# Basic imports
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Basic Usage

### Minimal Example

```python
# Load data (must have 'ds' and 'y' columns)
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365),
    'y': np.random.normal(0, 1, 365).cumsum()
})

# Initialize and fit model
m = Prophet()
m.fit(df)

# Create future dataframe for forecasting
future = m.make_future_dataframe(periods=30)

# Generate forecast
forecast = m.predict(future)

# Plot forecast
fig = m.plot(forecast)
```

### Required Data Format

- **ds**: Datetime column (date or datetime)
- **y**: Target metric to forecast (numeric)

### Forecast Output Columns

|Column                                              |Description                       |
|----------------------------------------------------|----------------------------------|
|`ds`                                                |Datetime                          |
|`yhat`                                              |Point forecast                    |
|`yhat_lower`                                        |Lower bound of prediction interval|
|`yhat_upper`                                        |Upper bound of prediction interval|
|`trend`                                             |Trend component                   |
|Components for seasonality, holidays, and regressors|                                  |

## Core Components

### Trend Models

```python
# Linear trend
m = Prophet()

# Logistic trend with cap
df['cap'] = 100
m = Prophet(growth='logistic')

# Logistic trend with cap and floor
df['cap'] = 100
df['floor'] = 10
m = Prophet(growth='logistic')

# Flat trend
m = Prophet(growth='flat')
```

### Seasonality

```python
# Auto-detect seasonality
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Custom seasonality
m.add_seasonality(
    name='monthly', 
    period=30.5, 
    fourier_order=5
)

# Customize Fourier order
m = Prophet(
    yearly_seasonality=10,  # Increase flexibility
    weekly_seasonality=3    # Simpler pattern
)

# Multiplicative seasonality
m = Prophet(seasonality_mode='multiplicative')
```

### Holidays & Events

```python
# Built-in holidays
from prophet.holidays import get_holiday_names
print(get_holiday_names())

# Add country holidays
m = Prophet(holidays=pd.DataFrame({
    'holiday': 'thanksgiving',
    'ds': pd.to_datetime(['2021-11-25', '2022-11-24']),
    'lower_window': -1,
    'upper_window': 1
}))

# Custom events
playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2021-01-10', '2021-01-17']),
    'lower_window': 0,
    'upper_window': 1
})
superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2021-02-07']),
    'lower_window': 0,
    'upper_window': 1
})
holidays = pd.concat([playoffs, superbowls])
m = Prophet(holidays=holidays)
```

## Advanced Features

### Changepoints

```python
# Automatic changepoint detection (default)
m = Prophet()

# Specify potential changepoints
m = Prophet(changepoints=['2020-03-15', '2020-06-01'])

# Adjust changepoint flexibility
m = Prophet(changepoint_prior_scale=0.05)  # Default: 0.05
                                          # Larger = more flexible

# Adjust number of potential changepoints
m = Prophet(n_changepoints=20)  # Default: 25

# Add changepoints visualization
from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
add_changepoints_to_plot(fig.gca(), m, forecast)
```

### Uncertainty Intervals

```python
# Adjust uncertainty interval width
m = Prophet(interval_width=0.95)  # 95% prediction intervals

# Customize uncertainty
m = Prophet(
    mcmc_samples=300,   # Use full Bayesian sampling
    uncertainty_samples=1000  # Number of simulations
)
```

### Regressors

```python
# Add external regressors
df['temperature'] = temp_data
m = Prophet()
m.add_regressor('temperature')

# Add regressor with prior scale
m.add_regressor('price', prior_scale=0.5)

# Add regressor with mode
m.add_regressor('marketing', mode='multiplicative')
```

### Working with Subdaily Data

```python
# Specify subdaily seasonality
m = Prophet(
    daily_seasonality=True,
    changepoint_prior_scale=0.01
)

# Custom subdaily seasonality
m.add_seasonality(
    name='hourly', 
    period=24/24,  # 1 hour
    fourier_order=3
)
```

## Feature Engineering

### Handling Holidays

```python
# Create holiday features
from prophet.make_holidays import make_holidays_df
playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2021-01-10', '2021-01-17']),
    'lower_window': 0,
    'upper_window': 1
})
m = Prophet(holidays=playoffs)

# View holiday components
holiday_effect = forecast[['ds', 'playoff']]
```

### Additional Regressors

```python
# Lag features
df['lag_1'] = df['y'].shift(1)
df['lag_7'] = df['y'].shift(7)
m = Prophet()
m.add_regressor('lag_1')
m.add_regressor('lag_7')

# Interaction terms
df['promo_weekend'] = df['promotion'] * df['is_weekend']
m.add_regressor('promo_weekend')

# Fourier features for custom cycles
for i in range(1, 5):
    df[f'sin_monthly_{i}'] = np.sin(2 * np.pi * i * df['day_of_month'] / 30)
    df[f'cos_monthly_{i}'] = np.cos(2 * np.pi * i * df['day_of_month'] / 30)
    m.add_regressor(f'sin_monthly_{i}')
    m.add_regressor(f'cos_monthly_{i}')
```

### Handling Missing Data

```python
# Impute missing values
df['y'] = df['y'].fillna(method='ffill')

# Remove outliers
from prophet.outlier import remove_outliers
cleaned_df = remove_outliers(df, outlier_method='zscore', outlier_threshold=3)
```

## Model Evaluation & Diagnostics

### Cross-Validation

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Generate cutoffs for cross-validation
df_cv = cross_validation(
    model=m,
    initial='365 days',  # Training period
    period='30 days',    # Spacing between cutoffs
    horizon='90 days'    # Prediction horizon
)

# Calculate performance metrics
df_metrics = performance_metrics(df_cv)
print(df_metrics.head())

# Plot metrics by horizon
from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
```

### Performance Metrics

```python
# Calculate and display various metrics
metrics = performance_metrics(
    df_cv, 
    metrics=['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']
)

# Group metrics by horizon
horizon_metrics = metrics.groupby('horizon').mean().reset_index()
```

### Component Analysis

```python
# Plot individual components
fig = m.plot_components(forecast)

# Extract components
trend = forecast['trend']
yearly = forecast['yearly'] if 'yearly' in forecast.columns else None
holidays = forecast['holidays'] if 'holidays' in forecast.columns else None

# Custom component plot
components_to_plot = ['trend', 'yearly', 'weekly']
fig, axes = plt.subplots(len(components_to_plot), 1, figsize=(10, 12))
for i, component in enumerate(components_to_plot):
    if component in forecast.columns:
        axes[i].plot(forecast['ds'], forecast[component])
        axes[i].set_title(f"{component} component")
```

### Feature Importance

```python
# Analyze regressor coefficients
regressor_coefs = pd.DataFrame({
    'regressor': m.extra_regressors.keys(),
    'coefficient': [m.extra_regressors[k]['beta'][0] for k in m.extra_regressors]
})
regressor_coefs = regressor_coefs.sort_values('coefficient', ascending=False)

# Calculate correlations between features and target
correlations = df.corr()['y'].sort_values(ascending=False)
```

## Hyperparameter Tuning

### Grid Search

```python
import itertools
from prophet.diagnostics import cross_validation, performance_metrics

# Define parameter grid
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each parameter combination

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(df)  # Fit model with given params
    df_cv = cross_validation(
        m, initial='730 days', period='180 days', horizon='365 days'
    )
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])
    
# Find the best parameters
best_params = all_params[np.argmin(rmses)]
print(best_params)
```

### Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Define the objective function
def objective(params):
    changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale = params
    m = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale
    )
    m.fit(df)
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='180 days')
    df_p = performance_metrics(df_cv)
    return df_p['rmse'].mean()

# Define the search space
space = [
    Real(0.001, 0.5, name='changepoint_prior_scale', prior='log-uniform'),
    Real(0.01, 10, name='seasonality_prior_scale', prior='log-uniform'),
    Real(0.01, 10, name='holidays_prior_scale', prior='log-uniform')
]

# Perform optimization
res_gp = gp_minimize(
    objective, space, n_calls=20, random_state=0, verbose=True
)

# Best parameters
best_params = {
    'changepoint_prior_scale': res_gp.x[0],
    'seasonality_prior_scale': res_gp.x[1],
    'holidays_prior_scale': res_gp.x[2]
}
```

## Best Practices

### Data Preparation

- **Minimum data requirements**: At least 2 full cycles (e.g., 2 years for yearly seasonality)
- **Handle outliers**: Remove extreme values or use robust methods
- **Check for stationarity**: Transform non-stationary data (differencing, log transforms)
- **Ensure consistent frequency**: Resample if needed to regular intervals
- **Handle missing data**: Impute or explicitly model missing periods

### Model Selection

- Start simple, gradually add complexity
- Test different trend models based on domain knowledge
- Only add regressors if they provide meaningful improvement
- Compare additive vs. multiplicative models when scale changes over time

### Common Configurations by Domain

**Retail Sales:**

```python
m = Prophet(
    seasonality_mode='multiplicative',  # Sales often scale with trend
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
```

**Website Traffic:**

```python
m = Prophet(
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=20,  # Higher flexibility for weekly patterns
    daily_seasonality=True,  # Include hourly patterns
    changepoint_prior_scale=0.01  # Less flexible trend
)
```

**Energy Consumption:**

```python
m = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=10,
    weekly_seasonality=True,
    daily_seasonality=True
)
m.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)
```

## Common Pitfalls

- **Overfitting**: Too flexible model (high prior scales) will fit to noise
- **Underfitting**: Too rigid model misses important patterns
- **Ignoring changepoints**: Missing significant shifts in the data
- **Extrapolating too far**: Forecasts become less reliable with longer horizons
- **Not accounting for special events**: Missing holidays or special events
- **Inappropriate seasonality mode**: Using additive when multiplicative is more appropriate
- **Not checking residuals**: Failing to detect pattern in errors

## Example Workflows

### E-Commerce Sales Forecasting

```python
# 1. Data preparation
sales_df = pd.DataFrame({
    'ds': order_dates,
    'y': daily_sales
})

# Add holidays and events
holidays = pd.concat([
    make_holidays_df('US', lower_window=-1, upper_window=1),
    pd.DataFrame({
        'holiday': 'BlackFriday',
        'ds': pd.to_datetime(['2020-11-27', '2021-11-26']),
        'lower_window': -1,
        'upper_window': 3
    })
])

# Add marketing spend as regressor
sales_df['marketing_spend'] = marketing_data

# 2. Model configuration
m = Prophet(
    holidays=holidays,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    yearly_seasonality=10
)
m.add_regressor('marketing_spend', mode='additive')

# 3. Fitting and forecasting
m.fit(sales_df)
future = m.make_future_dataframe(periods=90)
future['marketing_spend'] = forecast_marketing_spend
forecast = m.predict(future)

# 4. Evaluation and visualization
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
```

### Website Traffic Analysis with Anomaly Detection

```python
# 1. Data preparation
traffic_df = pd.DataFrame({
    'ds': timestamps,
    'y': pageviews
})

# 2. Initial model for anomaly detection
m_init = Prophet(interval_width=0.99)
m_init.fit(traffic_df)
forecast_init = m_init.predict(traffic_df)

# 3. Identify and remove anomalies
is_anomaly = (traffic_df['y'] < forecast_init['yhat_lower']) | (traffic_df['y'] > forecast_init['yhat_upper'])
cleaned_df = traffic_df[~is_anomaly].copy()

# 4. Build refined model on clean data
m = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.01
)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(cleaned_df)

# 5. Generate forecast
future = m.make_future_dataframe(periods=30, freq='H')
forecast = m.predict(future)

# 6. Visualize results
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
```