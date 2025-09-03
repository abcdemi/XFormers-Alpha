# === Kaggle-Style Trading Strategy Simulation ===
# Requirements: pip install yfinance lightgbm pandas numpy matplotlib scikit-learn

# --- 1. Setup and Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re # Import the regular expression library

# --- 2. Data Acquisition and Feature Engineering ---
print("Step 2: Downloading data and engineering features...")
ticker = "GOOGL"
df = yf.download(ticker, period="10y", interval="1d", auto_adjust=True)

def create_features(data):
    """Create time-series features from the price data."""
    df_feat = data.copy()
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f'lag_return_{lag}'] = df_feat['Close'].pct_change(lag)
    for window in [5, 10, 20, 60]:
        df_feat[f'rolling_mean_{window}'] = df_feat['Close'].rolling(window=window).mean()
        df_feat[f'rolling_std_{window}'] = df_feat['Close'].rolling(window=window).std()
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['rsi'] = 100 - (100 / (1 + rs))
    df_feat = df_feat.dropna()
    return df_feat

df_features = create_features(df)

# --- 3. Target De