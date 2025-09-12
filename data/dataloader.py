# data/dataloader.py
"""
Handles data loading (with caching), feature generation, scaling, and splitting.
"""
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.features import generate_features
import sys
import os

def load_and_prepare_data(config: dict) -> dict:
    """
    Orchestrates the entire data loading and preparation pipeline using a local cache.
    """
    print("--- Starting Data Preparation ---")

    # --- 1. Load Raw Data (with Caching) ---
    universe = config['universe']
    cache_path = os.path.join("local_data", f"{universe}.parquet")
    
    if os.path.exists(cache_path):
        print(f"Loading cached data for {universe} from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"No cached data found. Please run 'python download_data.py' first.")
        print(f"Attempting to download data for {universe} now...")
        df = yf.download(universe, start='2010-01-01', end='2024-12-31')
        if df.empty:
            print(f"\nERROR: Failed to download data for {universe}.\n")
            sys.exit(1)
        # Create cache directory if it doesn't exist
        if not os.path.exists("local_data"):
            os.makedirs("local_data")
        df.to_parquet(cache_path)
        print(f"Data downloaded and cached to {cache_path}.")

    df.columns = [col.lower() for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={'date': 'timestamp', 'adj close': 'adj_close'}, inplace=True)
    print("Raw data loaded.")
    
    # ... (The rest of the file is exactly the same as before) ...
    print("Generating features...")
    featured_df = generate_features(df, config['features'])
    featured_df['target'] = featured_df['close'].shift(-config['horizon'])
    featured_df.dropna(inplace=True)
    print(f"Label '{config['label']['target']}' created for horizon t+{config['horizon']}.")
    y = featured_df['target']
    X = featured_df.drop(columns=['target'])
    test_size = config.get('test_size', 0.2)
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx].copy(); X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy(); y_test = y.iloc[split_idx:].copy()
    df_test_original = X_test[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    X_train.drop(columns='timestamp', inplace=True); X_test.drop(columns='timestamp', inplace=True)
    print(f"Data split into {len(X_train)} train and {len(X_test)} test samples.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print("Features scaled successfully.")
    
    return {
        "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "df_test_original": df_test_original
    }