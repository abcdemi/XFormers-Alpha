# train_tft.py
"""
A clean, from-scratch training script dedicated to the Temporal Fusion Transformer.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import the necessary modules from your project
# NOTE: Ensure you have a working loaders.py file as created previously.
from xformers_alpha.data.loaders import load_yfinance 
from models.tft import TFTModel

# --- Configuration Block ---
# All major parameters are defined here for easy modification.
CONFIG = {
    "ticker": "GOOGL",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "test_size": 0.2,
    "encoder_length": 60, # How many past days the model sees (lookback window)
    "forecast_horizon": 1,   # How many days to predict into the future
    "hidden_size": 16,     # Model complexity
}

def get_aligned_timestamps(full_df, y_test, test_size, lookback_period):
    """A helper function to get the correct timestamps for plotting."""
    split_idx = int(len(full_df) * (1 - test_size))
    # The first prediction corresponds to the day after the first validation encoder sequence
    first_pred_idx = split_idx + lookback_period
    # Ensure we don't select more timestamps than we have predictions
    end_idx = first_pred_idx + len(y_test)
    return full_df['timestamp'].iloc[first_pred_idx:end_idx]

def main():
    """Main function to run the training and visualization pipeline."""
    
    # --- 1. Load Data ---
    print(f"Loading data for {CONFIG['ticker']}...")
    try:
        full_df = load_yfinance(
            ticker=CONFIG["ticker"], 
            start=CONFIG["start_date"], 
            end=CONFIG["end_date"]
        )
        print(f"Data loaded successfully. Shape: {full_df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        return

    # --- 2. Initialize and Run the TFT Model ---
    tft_model = TFTModel(
        encoder_length=CONFIG["encoder_length"], 
        forecast_horizon=CONFIG["forecast_horizon"], 
        hidden_size=CONFIG["hidden_size"]
    )
    predictions, y_test = tft_model.train_and_evaluate(full_df, test_size=CONFIG["test_size"])
    print("\nPrediction and evaluation complete.")

    # --- 3. Visualize Predictions ---
    print("Visualizing results...")
    timestamps = get_aligned_timestamps(
        full_df, y_test, CONFIG["test_size"], CONFIG["encoder_length"]
    )

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(timestamps, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(timestamps, predictions, label='Predicted (TFT)', color='red', linestyle='--')
    ax.set_title(f"{CONFIG['ticker']} Price Prediction: Temporal Fusion Transformer", fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()