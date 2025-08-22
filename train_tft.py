# train_tft.py
"""
A clean, from-scratch training script for the Temporal Fusion Transformer model.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import the necessary functions from your project
from xformers_alpha.data.loaders import load_yfinance
from models.tft import TFTModel

def main():
    """Main function to run the training and visualization pipeline."""
    
    # --- 1. Load Data ---
    print("Loading data...")
    try:
        full_df = load_yfinance('GOOGL', start='2020-01-01', end='2023-12-31')
        print(f"Data loaded successfully. Shape: {full_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Initialize and Run the TFT Model ---
    # The TFTModel class handles all the complex parts internally.
    tft_model = TFTModel(
        encoder_length=60, 
        forecast_horizon=1, 
        hidden_size=16
    )
    predictions, y_test = tft_model.train_and_evaluate(full_df, test_size=0.2)
    print("\nPrediction and evaluation complete.")

    # --- 3. Visualize Predictions ---
    print("Visualizing results...")
    
    # To plot correctly, we need the timestamps that correspond to our test set.
    # The first prediction is for the day after the first validation encoder sequence ends.
    test_size = 0.2
    split_idx = int(len(full_df) * (1 - test_size))
    first_prediction_day_index = split_idx + tft_model.encoder_length
    
    # Ensure we don't select more timestamps than we have predictions
    end_index = first_prediction_day_index + len(y_test)
    timestamps_test = full_df['timestamp'].iloc[first_prediction_day_index:end_index]

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(timestamps_test, y_test, label='Actual Prices', color='blue', linewidth=2)
    ax.plot(timestamps_test, predictions, label='Predicted (TFT)', color='red', linestyle='--')

    ax.set_title('GOOGL Price Prediction: Temporal Fusion Transformer', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()