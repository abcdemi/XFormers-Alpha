# === Prophet Experiment for GOOGL Stock Forecasting ===
# Requirements: pip install prophet yfinance pandas matplotlib

# --- 1. Setup and Imports ---
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 2. Data Preparation ---
print("Step 2: Downloading and preparing data for Prophet...")
ticker = "GOOGL"
# Download a longer history for Prophet to better detect yearly seasonality
df = yf.download(ticker, period="10y", interval="1d", auto_adjust=True)

# Prophet requires a specific DataFrame format with two columns: 'ds' and 'y'.
# 'ds' must be a datetime object, and 'y' must be the numeric value to forecast.
prophet_df = df.reset_index()
prophet_df = prophet_df[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']

# Display the first few rows to confirm the format
print("Data prepared for Prophet:")
print(prophet_df.head())

# --- 3. The Prophet Model Definition and Training ---
print("\nStep 3: Defining and training the Prophet model...")

# Instantiate the Prophet model.
# By default, Prophet includes weekly and yearly seasonality.
# We will also add a built-in country-specific holiday model for the US.
model = Prophet(
    daily_seasonality=False, # Stock markets don't have strong daily patterns
    weekly_seasonality=True, # Is there a "day-of-the-week" effect?
    yearly_seasonality=True, # Is there a "time-of-year" effect?
    seasonality_mode='multiplicative' # Seasonality can be additive or multiplicative
)
model.add_country_holidays(country_name='US')

# Fit the model to our historical data.
model.fit(prophet_df)
print("Model training complete.")

# --- 4. Forecasting the Future ---
print("\nStep 4: Creating a future dataframe and generating forecast...")

# Create a dataframe that extends into the future for the period we want to forecast.
# Prophet's helper function makes this easy.
future = model.make_future_dataframe(periods=365)

# Use the trained model to make predictions on this future dataframe.
# The output 'forecast' dataframe contains many columns, including the prediction 'yhat'
# and the uncertainty intervals 'yhat_lower' and 'yhat_upper'.
forecast = model.predict(future)

print("Forecast generated. Displaying the last few rows:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# --- 5. Visualization and Explanation ---
print("\nStep 5: Plotting the forecast and its components...")

# Plot 1: The Main Forecast Plot
# Prophet's built-in plotting function creates a clear and informative visualization.
fig1 = model.plot(forecast)
plt.title(f"{ticker} Stock Price Forecast with Prophet", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.gca().axvline(prophet_df['ds'].max(), color='r', linestyle='--', lw=2, label='Forecast Start')
plt.legend()
plt.show()

print("\nExplanation of the Forecast Plot:")
print("- The black dots are the actual historical data points.")
print("- The dark blue line is the model's forecast ('yhat').")
print("- The light blue shaded area represents the uncertainty interval. Real-world values are expected to fall within this range.")
print("- The model projects a continuation of the long-term trend it has learned.")

# Plot 2: The Components Plot
# This is Prophet's key strength: explainability. We can see the individual
# components that are added together to produce the final forecast.
fig2 = model.plot_components(forecast)
plt.show()

print("\nExplanation of the Components Plot:")
print("- Trend: Shows the long-term, non-periodic growth of the stock price that Prophet has captured. You can see how Prophet models changes in the growth rate over time.")
print("- Weekly Seasonality: Shows the effect of each day of the week. For stocks, we often see a dip on weekends (when the market is closed and data is static) and slight variations on trading days.")
print("- Yearly Seasonality: Shows the pattern of price changes over the course of a year. It captures if the stock tends to be higher or lower at certain times of the year (e.g., end-of-year rallies).")