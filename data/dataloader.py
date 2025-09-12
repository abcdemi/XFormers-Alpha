# data/dataloader.py
"""
Handles data loading from pre-split Kaggle CSVs, feature generation, scaling, and splitting.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.features import generate_features
import sys
import os

def load_and_prepare_data(config: dict) -> dict:
    """
    Orchestrates the data pipeline using pre-split train and test CSV files.
    """
    print("--- Starting Data Preparation ---")

    # --- 1. Load Raw Data from Local Pre-Split CSVs ---
    train_path = os.path.join("local_data", "Google_Stock_Train (2010-2022).csv")
    test_path = os.path.join("local_data", "Google_Stock_Test (2023).csv")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("\n" + "="*50)
        print("ERROR: Pre-split data files not found.")
        print("Please download the data from Kaggle and place the following files in the 'local_data' directory:")
        print("- Google_Stock_Train (2010-2022).csv")
        print("- Google_Stock_Test (2023).csv")
        print("="*50 + "\n")
        sys.exit(1)
        
    print(f"Loading train data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    # --- 2. Standardize and Combine for Feature Engineering ---
    def standardize_df(df):
        df.columns = [col.lower() for col in df.columns]
        df.rename(columns={'date': 'timestamp', 'adj close': 'adj_close'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    train_df = standardize_df(train_df)
    test_df = standardize_df(test_df)
    
    # Store the timestamp for the split point
    train_end_date = train_df['timestamp'].max()
    
    # Combine data to ensure feature continuity (e.g., for rolling averages)
    full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(by='timestamp')
    print("Combined data for robust feature generation.")

    # 3. Generate Features on the full dataset
    print("Generating features...")
    featured_df = generate_features(full_df, config['features'])
    
    # 4. Define the Label (Target Variable)
    featured_df['target'] = featured_df['close'].shift(-config['horizon'])
    featured_df.dropna(inplace=True)
    print(f"Label '{config['label']['target']}' created for horizon t+{config['horizon']}.")
    
    # 5. Re-split the data back into Train and Test sets
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
    
    # 6. Scale Data
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