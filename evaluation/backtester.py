# evaluation/backtester.py
import pandas as pd
import numpy as np

def run_backtest(predictions: np.ndarray, df_test_original: pd.DataFrame, config: dict) -> dict:
    print("--- Running Backtest with Adaptive Directional Allocator---")
    
    if len(predictions) != len(df_test_original):
        offset = len(df_test_original) - len(predictions)
        df_test = df_test_original.iloc[offset:].copy()
    else:
        df_test = df_test_original.copy()
        
    df_test['prediction'] = predictions

    # --- START OF DEFINITIVE FIX: Adaptive, Bias-Correcting Allocator ---
    # The model has a strong ranking ability (IC) but a systematic negative bias.
    # We will trade on the signal relative to its own recent history.
    
    # Calculate the 21-day rolling average of the predictions themselves.
    prediction_moving_avg = df_test['prediction'].rolling(window=21, min_periods=1).mean()
    
    # The signal is to BUY only if the current prediction is stronger than its recent average.
    # This removes the systematic bias and trades on the relative signal.
    signal = (df_test['prediction'] > prediction_moving_avg).astype(int)
    weights = signal
    # --- END OF DEFINITIVE FIX ---

    df_test['weight'] = weights
    
    df_test['asset_return'] = df_test['close'].pct_change().shift(-1)
    df_test['strategy_return'] = df_test['asset_return'] * df_test['weight']
    
    costs_bps = config.get('costs_bps', 0)
    turnover = df_test['weight'].diff().abs()
    transaction_costs = turnover * (costs_bps / 10000.0)
    df_test['strategy_return'] -= transaction_costs
    
    df_test.dropna(inplace=True)

    initial_capital = 100_000
    if not df_test.empty and 'strategy_return' in df_test:
        df_test['equity_curve'] = initial_capital * (1 + df_test['strategy_return']).cumprod()
    else:
        # Handle case with no trades
        df_test['equity_curve'] = pd.Series([initial_capital], index=df_test.index[:1])

    weights_df = pd.DataFrame(index=df_test['timestamp'])
    weights_df['asset'] = df_test.get('weight', pd.Series(0, index=df_test.index))
    weights_df['cash'] = 1 - weights_df['asset']

    return {
        "returns": df_test.get('strategy_return', pd.Series()),
        "equity_curve": df_test['equity_curve'],
        "weights": weights_df,
        "timestamps": df_test['timestamp']
    }