# data/dataloader.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.features import generate_features
import sys
import os
import numpy as np

def load_and_prepare_data(config: dict) -> dict:
    print("--- Starting Data Preparation ---")
    train_path = os.path.join("local_data", "Google_Stock_Train (2010-2022).csv")
    test_path = os.path.join("local_data", "Google_Stock_Test (2023).csv")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("\nERROR: Pre-split data files not found.\n")
        sys.exit(1)
        
    print(f"Loading train data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    def standardize_df(df):
        df.columns = [col.lower() for col in df.columns]
        df.rename(columns={'date': 'timestamp', 'adj close': 'adj_close'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    train_df = standardize_df(train_df)
    test_df = standardize_df(test_df)
    
    train_end_date = train_df['timestamp'].max()
    full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(by='timestamp')
    print("Combined data for robust feature generation.")

    print("Generating features...")
    featured_df = generate_features(full_df, config['features'])
    
    # --- START OF FIX: Change the target to forward returns ---
    horizon = config['horizon']
    # Calculate the future log return, which is a better target for financial models
    featured_df['target'] = np.log(featured_df['close'].shift(-horizon) / featured_df['close'])
    featured_df.dropna(inplace=True)
    print(f"Label changed to 'logret_fwd_{horizon}'.")
    # --- END OF FIX ---
    
    split_index = featured_df[featured_df['timestamp'] <= train_end_date].index.max()
    
    train_featured_df = featured_df.loc[:split_index].copy()
    test_featured_df = featured_df.loc[split_index + 1:].copy()

    y_train = train_featured_df['target']
    X_train = train_featured_df.drop(columns=['target'])
    y_test = test_featured_df['target']
    X_test = test_featured_df.drop(columns=['target'])
    
    df_test_original = X_test[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    X_train.drop(columns='timestamp', inplace=True)
    X_test.drop(columns='timestamp', inplace=True)
    print(f"Data re-split into {len(X_train)} train and {len(X_test)} test samples.")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print("Features scaled successfully.")
    
    return {
        "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "df_test_original": df_test_original
    }