# data/features.py
"""
Functions for generating predictive features from raw financial data.
Each function takes a DataFrame and returns it with new columns added.
"""
import pandas as pd
import numpy as np

def add_price_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Adds price-based technical indicators to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        feature_list (list): A list of feature names to generate.

    Returns:
        pd.DataFrame: DataFrame with new feature columns.
    """
    if 'logret_1' in feature_list:
        df['logret_1'] = np.log(df['close'] / df['close'].shift(1))
    if 'logret_5' in feature_list:
        df['logret_5'] = np.log(df['close'] / df['close'].shift(5))
    if 'rsi_14' in feature_list:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
    if 'sma_20' in feature_list:
        df['sma_20'] = df['close'].rolling(20).mean()
    if 'bb_pct' in feature_list:
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        df['bb_pct'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    return df

def add_volume_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Adds volume-based technical indicators to the DataFrame.
    """
    if 'vol_scaled_20' in feature_list:
        sma_vol_20 = df['volume'].rolling(20).mean()
        df['vol_scaled_20'] = df['volume'] / sma_vol_20
        
    return df

def add_calendar_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Adds calendar-based features to the DataFrame.
    """
    if 'dow' in feature_list:
        df['dow'] = df['timestamp'].dt.dayofweek
    if 'month' in feature_list:
        df['month'] = df['timestamp'].dt.month
        
    return df


def generate_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Main function to orchestrate feature generation based on a config file.

    Args:
        df (pd.DataFrame): Raw data with timestamp and OHLCV columns.
        config (dict): The 'features' section of the YAML config.

    Returns:
        pd.DataFrame: DataFrame with all generated features.
    """
    df = df.copy()
    
    if 'price' in config:
        df = add_price_features(df, config['price'])
    if 'volume' in config:
        df = add_volume_features(df, config['volume'])
    if 'calendar' in config:
        df = add_calendar_features(df, config['calendar'])
        
    # Drop rows with NaNs created by rolling windows/lags
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df