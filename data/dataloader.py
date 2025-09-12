# data/dataloader.py
"""
Handles data loading, feature generation, scaling, and splitting.
"""
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.features import generate_features
import sys

def load_and_prepare_data(config: dict) -> dict:
    """
    Orchestrates the entire data loading and preparation pipeline.
    """
    print("--- Starting Data Preparation ---")

    # 1. Load Raw Data
    print(f"Loading data for universe: {config['universe']}...")
    df = yf.download(config['universe'], start='2010-01-01', end='2024-12-31')
    
    if df.empty:
        print("\nERROR: No data was downloaded. Check ticker symbol or network connection.\n")
        sys.exit(1)
    
    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        # This handles cases where yfinance returns ('Close', 'GOOGL')
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
        
    df.reset_index(inplace=True)
    
    # --- START OF FIX ---
    # The column from reset_index() is named 'Date' (capital D). We must target this name.
    df.rename(columns={'date': 'timestamp', 'adj close': 'adj_close'}, inplace=True)
    # It's safer to check for both cases, as yfinance behavior can vary.
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'timestamp'}, inplace=True)
    # --- END OF FIX ---
    
    print("Raw data loaded and timestamp column created.")

    # 2. Generate Features
    print("Generating features...")
    featured_df = generate_features(df, config['features'])
    
    # 3. Define the Label (Target Variable)
    featured_df['target'] = featured_df['close'].shift(-config['horizon'])
    featured_df.dropna(inplace=True)
    print(f"Label '{config['label']['target']}' created for horizon t+{config['horizon']}.")
    
    # 4. Split Data
    y = featured_df['target']
    X = featured_df.drop(columns=['target'])
    
    test_size = config.get('test_size', 0.2)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    df_test_original = X_test[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    X_train.drop(columns='timestamp', inplace=True)
    X_test.drop(columns='timestamp', inplace=True)
    print(f"Data split into {len(X_train)} train and {len(X_test)} test samples.")
    
    # 5. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print("Features scaled successfully.")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "df_test_original": df_test_original
    }