# train_tft.py
"""
Training script for the Temporal Fusion Transformer model.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Make sure your project is installed or the path is correct
from xformers_alpha.data.loaders import load_yfinance
from models.tft import TFTModel

# 1. --- LOAD THE FULL DATASET ---
try:
    full_df = load_yfinance('GOOGL', start='2020-01-01', end='2023-12-31')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# 2. --- RUN THE TFT MODEL ---
# The model handles its own data prep and train/validation split
tft_model = TFTModel(encoder_length=60, forecast_horizon=1)
predictions, y_test = tft_model.train_and_evaluate(full_df, test_size=0.2)

# 3. --- VISUALIZE PREDICTIONS ---
# The validation set starts after the training period
test_size = 0.2
split_idx = int(len(full_df) * (1 - test_size))

# We need to account for the encoder length to get the correct start time
# The first prediction corresponds to the first day of the validation set + encoder length
first_prediction_day = split_idx + tft_model.encoder_length
timestamps_test = full_df['timestamp'].iloc[first_prediction_day : first_prediction_day + len(y_test)]

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(timestamps_test, y_test, label='Actual Prices', color='blue', linewidth=2)
ax.plot(timestamps_test, predictions, label='Predicted (TFT)', color='red', linestyle='--')

ax.set_title('GOOGL Price Prediction: Temporal Fusion Transformer', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()