import pandas as pd

# Load the dataset 
df = pd.read_csv('Superstore Sales.csv', parse_dates=['Order Date'], index_col='Order Date')

# Resample the data by day and sum the sales for each day
df_daily = df.resample('D').sum()['Sales']

# Plotting the original data to check trends
df_daily.plot(figsize=(10, 6))
plt.title('Daily Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(df_daily)
print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")

# If the data is not stationary, we will difference the series
if result[1] > 0.05:
    df_daily = df_daily.diff().dropna()
  from statsmodels.tsa.statespace.sarimax import SARIMAX

# Train SARIMA model (example order: (1, 1, 1) for AR, differencing, and MA, and (1, 1, 1, 12) for seasonal order)
sarima_model = SARIMAX(df_daily, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_results = sarima_model.fit()

# Forecast the next 30 days
forecast = sarima_results.get_forecast(steps=30)
forecast_index = pd.date_range(df_daily.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Forecasted Sales'])

# Plot historical data along with the forecasted sales
plt.figure(figsize=(10, 6))
plt.plot(df_daily, label='Historical Sales')
plt.plot(forecast_df, label='Forecasted Sales', linestyle='--', color='red')
plt.title('Sales Forecast for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

