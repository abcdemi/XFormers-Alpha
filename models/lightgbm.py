# === Kaggle-Style Trading Strategy Simulation (v11: The Robust System) ===
# This version combines our best model with a professional-grade validation and execution framework:
# 1. Walk-Forward Cross-Validation for robust performance evaluation.
# 2. An Ensemble of diverse models for a more stable signal.
# 3. A Market Regime Filter for capital preservation during downturns.
# Requirements: pip install yfinance lightgbm pandas numpy matplotlib scikit-learn

# --- 1. Setup and Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import re

# --- 2. Data Acquisition and Feature Engineering ---
print("Step 2: Downloading data and engineering features...")
df_dict = yf.download("GOOGL SPY", period="10y", interval="1d", auto_adjust=True)
df = df_dict.loc[:, (slice(None), 'GOOGL')]; df.columns = df.columns.droplevel(1)
spy_df = df_dict.loc[:, (slice(None), 'SPY')]; spy_df.columns = spy_df.columns.droplevel(1)

def create_final_features(data, spy_data):
    # ... [Same final feature engineering function as V4] ...
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

# --- 3. Target Definition ---
df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
df_features = df_features.dropna()
X = df_features.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'target'], axis=1)
y = df_features['target']
X.columns = [re.sub(r'[^a_zA-Z0-9_]', '_', col) for col in X.columns]

# --- 4. The Robust Backtesting Loop ---
print("\nStep 4: Running Walk-Forward Cross-Validation...")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

fold_results = []
all_equity_curves = []

# Pre-calculate the regime filter data
spy_200_sma = spy_df['Close'].rolling(window=200).mean()

def run_backtest_with_filter(predictions, test_data_indices, spy_full_data, spy_sma):
    initial_capital = 100000.0
    portfolio_value = initial_capital
    portfolio_history = []
    
    daily_returns = df['Close'].loc[test_data_indices].pct_change().dropna()
    aligned_preds = predictions[:len(daily_returns)]
    
    for i in range(len(daily_returns)):
        current_date = daily_returns.index[i]
        market_return = daily_returns.iloc[i]
        
        # --- REGIME FILTER LOGIC ---
        if spy_full_data['Close'].loc[current_date] < spy_sma.loc[current_date]:
            strategy_return = 0.0  # Market is in a downturn, stay in cash
        else:
            # Market is healthy, follow the binary model signal
            signal = 1 if aligned_preds[i] > 0.5 else 0 # Convert avg probability to binary
            strategy_return = market_return if signal == 1 else 0.0
            
        portfolio_value *= (1 + strategy_return)
        portfolio_history.append(portfolio_value)
        
    return pd.Series(portfolio_history, index=daily_returns.index)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"--- Processing Fold {fold + 1}/{n_splits} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- Ensemble Model Training ---
    # 1. The Expert (V4 Model)
    params_v4 = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'n_estimators': 250, 'learning_rate': 0.02, 'num_leaves': 60, 'max_depth': 8, 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'colsample_bytree': 0.7, 'subsample': 0.7}
    model_expert = lgb.LGBMClassifier(**params_v4)
    model_expert.fit(X_train, y_train)
    
    # 2. The Contrarian (Logistic Regression) - requires scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model_contrarian = LogisticRegression(solver='liblinear', random_state=42)
    model_contrarian.fit(X_train_scaled, y_train)

    # 3. The Simpleton (Simple LGBM)
    params_simple = {'objective': 'binary', 'n_estimators': 50, 'max_depth': 3, 'seed': 42, 'n_jobs': -1, 'verbose': -1}
    model_simple = lgb.LGBMClassifier(**params_simple)
    model_simple.fit(X_train, y_train)

    # --- Generate Ensemble Prediction ---
    proba_expert = model_expert.predict_proba(X_test)[:, 1]
    proba_contrarian = model_contrarian.predict_proba(X_test_scaled)[:, 1]
    proba_simple = model_simple.predict_proba(X_test)[:, 1]
    
    ensemble_probas = (proba_expert + proba_contrarian + proba_simple) / 3.0
    
    # --- Backtest and Store Results for this Fold ---
    equity_curve = run_backtest_with_filter(ensemble_probas, X_test.index, spy_df, spy_200_sma)
    all_equity_curves.append(equity_curve)
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
    
    fold_results.append({'total_return': total_return, 'sharpe_ratio': sharpe_ratio})

# --- 5. Final, Robust Evaluation ---
print("\nStep 5: Aggregating Walk-Forward Results...")
results_df = pd.DataFrame(fold_results)
print("--- Robust Strategy Performance (Averaged over all Folds) ---")
print(f"Average Total Return: {results_df['total_return'].mean():.2%}")
print(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}")
print("\n--- Performance Consistency (Standard Deviation over all Folds) ---")
print(f"Std Dev of Total Return: {results_df['total_return'].std():.2%}")
print(f"Std Dev of Sharpe Ratio: {results_df['sharpe_ratio'].std():.2f}")

# --- 6. Visualization ---
print("\nStep 6: Visualizing the consistency of the strategy...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))

for i, equity_curve in enumerate(all_equity_curves):
    plt.plot(equity_curve, label=f'Fold {i+1}', alpha=0.7)

# Add a benchmark for context (Buy & Hold over the final fold's test period)
last_fold_test_indices = all_equity_curves[-1].index
buy_hold_equity = (df['Close'].loc[last_fold_test_indices].pct_change().add(1).cumprod() * 100000)
plt.plot(buy_hold_equity, label='Buy & Hold (Final Fold)', color='black', linestyle='--')

plt.title(f"Walk-Forward Equity Curves for the Robust GOOGL Strategy", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend()
plt.show()