# evaluation/backtester.py
"""
A simple backtesting engine to simulate a trading strategy.
"""
import pandas as pd
import numpy as np

def run_backtest(predictions: np.ndarray, df_test_original: pd.DataFrame, config: dict) -> dict:
    """
    Runs a simple single-asset backtest based on model predictions.

    Args:
        predictions (np.ndarray): Raw numerical predictions from a model.
        df_test_original (pd.DataFrame): The original test data with OHLCV and timestamps.
        config (dict): The 'backtest' section of the YAML config.

    Returns:
        dict: A dictionary containing the equity curve, returns series, and weights DataFrame.
    """
    print("--- Running Backtest ---")
    
    # Align predictions with the test dataframe
    # Predictions are for t+1, so they align with the price change from t to t+1
    if len(predictions) != len(df_test_original):
        # This can happen with sequence models, we need to align
        offset = len(df_test_original) - len(predictions)
        df_test = df_test_original.iloc[offset:].copy()
        df_test['prediction'] = predictions
    else:
        df_test = df_test_original.copy()
        df_test['prediction'] = predictions

    # --- 1. Signal to Weight Allocation ---
    # Convert raw predictions into portfolio weights based on the chosen allocator
    allocator = config.get('allocator', 'regression_long_only')
    
    if allocator == 'regression_long_only':
        # A simple allocator: if prediction > current price, go long.
        # Weight is proportional to the predicted % increase.
        # This is a naive example; a real allocator would be more complex.
        predicted_change = (df_test['prediction'] - df_test['close']) / df_test['close']
        # Be long if predicted change is positive, otherwise be in cash
        weights = np.clip(predicted_change, 0, 1) # Clip at 100% weight
    else:
        raise NotImplementedError(f"Allocator '{allocator}' not implemented.")

    df_test['weight'] = weights
    
    # --- 2. Calculate Strategy Returns ---
    # Assume trades are entered at 'close' and held for one day
    # The return for day 't+1' is the price change from 't' to 't+1', multiplied by the weight decided at 't'
    
    # Calculate the next day's return for the asset
    df_test['asset_return'] = df_test['close'].pct_change().shift(-1)
    
    # The strategy return is the asset return multiplied by our weight from the *previous* day
    df_test['strategy_return'] = df_test['asset_return'] * df_test['weight']
    
    # --- 3. Account for Costs ---
    costs_bps = config.get('costs_bps', 0)
    turnover = df_test['weight'].diff().abs()
    transaction_costs = turnover * (costs_bps / 10000.0)
    df_test['strategy_return'] -= transaction_costs
    
    df_test.dropna(inplace=True)

    # --- 4. Generate Equity Curve ---
    initial_capital = 100_000
    df_test['equity_curve'] = initial_capital * (1 + df_test['strategy_return']).cumprod()
    
    # Prepare weights DataFrame for metrics calculation
    weights_df = pd.DataFrame(index=df_test['timestamp'])
    weights_df['asset'] = df_test['weight'].values
    weights_df['cash'] = 1 - df_test['weight'].values

    return {
        "returns": df_test['strategy_return'],
        "equity_curve": df_test['equity_curve'],
        "weights": weights_df,
        "timestamps": df_test['timestamp']
    }