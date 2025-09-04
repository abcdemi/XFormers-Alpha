# === Kaggle-Style Trading Strategy Simulation (v10: The Definitive Test) ===
# This version combines our most robust model (V4) with a strategy logic
# that is explicitly optimized for the best financial metric (Sharpe Ratio).
# Requirements: pip install yfinance lightgbm pandas numpy matplotlib scikit-learn

# --- 1. Setup and Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import re

# --- 2. Data Acquisition and Feature Engineering ---
print("Step 2: Downloading data and engineering features...")
df_dict = yf.download("GOOGL SPY", period="10y", interval="1d", auto_adjust=True)
df = df_dict.loc[:, (slice(None), 'GOOGL')]; df.columns = df.columns.droplevel(1)
spy_df = df_dict.loc[:, (slice(None), 'SPY')]; spy_df.columns = spy_df.columns.droplevel(1)

def create_final_features(data, spy_data):
    # ... [Same final feature engineering function as before] ...
    df_feat = data.copy()
    for lag in [1, 2, 3, 5, 10]: df_feat[f'lag_return_{lag}'] = df_feat['Close'].pct_change(lag)
    for window in [5, 10, 20, 60]: df_feat[f'rolling_std_{window}'] = df_feat['Close'].rolling(window=window).std()
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean(); ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = ema_12 - ema_26; df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    bb_window = 20; rolling_mean_bb = df_feat['Close'].rolling(window=bb_window).mean(); rolling_std_bb = df_feat['Close'].rolling(window=bb_window).std()
    df_feat['bb_width'] = ((rolling_mean_bb + (rolling_std_bb*2)) - (rolling_mean_bb - (rolling_std_bb*2))) / rolling_mean_bb
    df_feat['bb_percent'] = (df_feat['Close'] - (rolling_mean_bb - (rolling_std_bb*2))) / ((rolling_mean_bb + (rolling_std_bb*2)) - (rolling_mean_bb - (rolling_std_bb*2)))
    high_low = df_feat['High'] - df_feat['Low']; high_close = np.abs(df_feat['High'] - df_feat['Close'].shift()); low_close = np.abs(df_feat['Low'] - df_feat['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1); df_feat['atr'] = tr.rolling(window=14).mean()
    df_feat['day_of_week'] = df_feat.index.dayofweek; df_feat['month'] = df_feat.index.month
    spy_returns = spy_data['Close'].pct_change(); df_feat['spy_return'] = spy_returns
    df_feat['spy_rolling_std_10'] = spy_returns.rolling(window=10).std()
    df_feat['interaction_lag1_spy'] = df_feat['lag_return_1'] * df_feat['spy_return']
    return df_feat.dropna()

df_features = create_final_features(df, spy_df)

# --- 3. Target Definition and Rigorous Data Splitting ---
df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
df_features = df_features.dropna()
X = df_features.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'target'], axis=1)
y = df_features['target']
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]

train_size = int(0.70 * len(X)); val_size = int(0.85 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

# --- 4. Training the Champion V4 Model ---
print("\nStep 4: Training our robust V4 model...")
params_v4 = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 250, 'learning_rate': 0.02, 'num_leaves': 60,
    'max_depth': 8, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7
}

# Train on the training set to get probabilities for the validation set
model_for_val = lgb.LGBMClassifier(**params_v4)
model_for_val.fit(X_train, y_train)
probas_val = model_for_val.predict_proba(X_val)[:, 1]

# Train the final model on all data before the test set
X_train_full = pd.concat([X_train, X_val]); y_train_full = pd.concat([y_train, y_val])
final_model = lgb.LGBMClassifier(**params_v4)
final_model.fit(X_train_full, y_train_full)
probas_test = final_model.predict_proba(X_test)[:, 1]

# --- 5. Optimize Strategy Thresholds for Sharpe Ratio ---
print("\nStep 5: Optimizing strategy thresholds for Sharpe Ratio on the validation set...")
def run_threshold_backtest(probabilities, test_data, threshold_low, threshold_high):
    initial_capital = 100000.0; portfolio_value = initial_capital; portfolio_history = []
    daily_returns = test_data['Close'].pct_change().dropna()
    aligned_probas = probabilities[:len(daily_returns)]
    for i in range(len(daily_returns)):
        proba = aligned_probas[i]
        market_return = daily_returns.iloc[i]
        position = 1.0 if proba > threshold_high else 0.5 if proba > threshold_low else 0.0
        strategy_return = market_return * position
        portfolio_value *= (1 + strategy_return)
        portfolio_history.append(portfolio_value)
    return pd.Series(portfolio_history, index=daily_returns.index)

def calculate_sharpe(portfolio_history):
    returns = portfolio_history.pct_change().dropna()
    return (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0

best_sharpe = -np.inf
best_thresholds = (0, 0)
# Brute-force search for the best thresholds on the validation set
for low in np.arange(0.50, 0.55, 0.01):
    for high in np.arange(0.55, 0.65, 0.01):
        if high <= low: continue
        portfolio_val = run_threshold_backtest(probas_val, df.loc[X_val.index], low, high)
        sharpe = calculate_sharpe(portfolio_val)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thresholds = (low, high)

print(f"Threshold optimization finished.")
print(f"Best Sharpe Ratio on Validation Set: {best_sharpe:.2f}")
print(f"Optimal Thresholds Found: Low={best_thresholds[0]:.2f}, High={best_thresholds[1]:.2f}")

# --- 6. Final Backtest and Evaluation on Test Set ---
print("\nStep 6: Running final backtest on the unseen test set...")
portfolio_history = run_threshold_backtest(probas_test, df.loc[X_test.index], best_thresholds[0], best_thresholds[1])

def evaluate_strategy(portfolio_history, test_data):
    total_return = (portfolio_history.iloc[-1] / portfolio_history.iloc[0]) - 1
    daily_strategy_returns = portfolio_history.pct_change().dropna()
    sharpe_ratio = (daily_strategy_returns.mean() / daily_strategy_returns.std()) * np.sqrt(252) if daily_strategy_returns.std() != 0 else 0.0
    rolling_max = portfolio_history.cummax(); daily_drawdown = portfolio_history / rolling_max - 1.0; max_drawdown = daily_drawdown.min()
    buy_hold_data = test_data['Close'].loc[portfolio_history.index]
    buy_hold_return = (buy_hold_data.iloc[-1] / buy_hold_data.iloc[0]) - 1
    print(f"--- Final Strategy Performance (Test Set) ---")
    print(f"Total Return: {total_return:.2%}"); print(f"Sharpe Ratio: {sharpe_ratio:.2f}"); print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"\n--- Benchmark ---"); print(f"Buy & Hold Return: {buy_hold_return:.2%}")

evaluate_strategy(portfolio_history, df)

# --- 7. Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))
plt.plot(portfolio_history, label='LGBM Strategy (V4 Model + Optimized Thresholds)', color='royalblue')
buy_hold_equity = (df['Close'].loc[portfolio_history.index].pct_change().add(1).cumprod() * 100000)
plt.plot(buy_hold_equity, label='Buy & Hold', color='gray', linestyle='--')
plt.title(f"{ticker} Strategy vs. Buy & Hold", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.show()