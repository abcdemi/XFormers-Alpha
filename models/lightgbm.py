# === Kaggle-Style Trading Strategy Simulation (v4: Final Feature Engineering) ===
# Requirements: pip install yfinance lightgbm pandas numpy matplotlib scikit-learn

# --- 1. Setup and Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

# --- 2. Data Acquisition ---
print("Step 2: Downloading data for GOOGL and SPY benchmark...")
df_dict = yf.download("GOOGL SPY", period="10y", interval="1d", auto_adjust=True)
df = df_dict.loc[:, (slice(None), 'GOOGL')]
df.columns = df.columns.droplevel(1)
spy_df = df_dict.loc[:, (slice(None), 'SPY')]
spy_df.columns = spy_df.columns.droplevel(1)

# --- 3. Final Feature Engineering ---
print("Step 3: Engineering the final, refined feature set...")

def create_final_features(data, spy_data):
    """Create a refined set of features based on importance analysis."""
    df_feat = data.copy()
    
    # --- Keep Proven Features ---
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f'lag_return_{lag}'] = df_feat['Close'].pct_change(lag)
    # Volatility was important, so we keep rolling_std
    for window in [5, 10, 20, 60]:
        df_feat[f'rolling_std_{window}'] = df_feat['Close'].rolling(window=window).std()
    
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = ema_12 - ema_26
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    
    bb_window = 20
    rolling_mean_bb = df_feat['Close'].rolling(window=bb_window).mean()
    rolling_std_bb = df_feat['Close'].rolling(window=bb_window).std()
    # Pruning: We only keep the important BB features: width and percent
    df_feat['bb_width'] = ((rolling_mean_bb + (rolling_std_bb*2)) - (rolling_mean_bb - (rolling_std_bb*2))) / rolling_mean_bb
    df_feat['bb_percent'] = (df_feat['Close'] - (rolling_mean_bb - (rolling_std_bb*2))) / ((rolling_mean_bb + (rolling_std_bb*2)) - (rolling_mean_bb - (rolling_std_bb*2)))
    
    high_low = df_feat['High'] - df_feat['Low']
    high_close = np.abs(df_feat['High'] - df_feat['Close'].shift())
    low_close = np.abs(df_feat['Low'] - df_feat['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_feat['atr'] = tr.rolling(window=14).mean()
    
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month

    # --- Double Down on What Works: More SPY and Interaction Features ---
    spy_returns = spy_data['Close'].pct_change()
    df_feat['spy_return'] = spy_returns
    # Add market volatility as a feature
    df_feat['spy_rolling_std_10'] = spy_returns.rolling(window=10).std()
    
    # Add the powerful interaction feature
    df_feat['interaction_lag1_spy'] = df_feat['lag_return_1'] * df_feat['spy_return']
    
    df_feat = df_feat.dropna()
    return df_feat

df_features = create_final_features(df, spy_df)

# --- 4. Target Definition and Data Splitting ---
df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
df_features = df_features.dropna()
X = df_features.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'target'], axis=1)
y = df_features['target']
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 5. Model Training ---
print("\nStep 5: Training the final LightGBM model...")
# We can slightly increase model complexity to handle the new features
params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 250, 'learning_rate': 0.02, 'num_leaves': 60,
    'max_depth': 8, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7
}
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# --- 6. Backtesting and Evaluation ---
print("\nStep 6: Running backtest and evaluating final strategy...")
def run_backtest(predictions, test_data):
    initial_capital = 100000.0
    portfolio_value = initial_capital
    portfolio_history = []
    daily_returns = test_data['Close'].pct_change().dropna()
    aligned_preds = predictions[:len(daily_returns)]
    for i in range(len(daily_returns)):
        signal = aligned_preds[i]
        market_return = daily_returns.iloc[i]
        strategy_return = market_return if signal == 1 else 0.0
        portfolio_value *= (1 + strategy_return)
        portfolio_history.append(portfolio_value)
    return pd.Series(portfolio_history, index=daily_returns.index)

def evaluate_strategy(portfolio_history, test_data):
    total_return = (portfolio_history.iloc[-1] / portfolio_history.iloc[0]) - 1
    daily_strategy_returns = portfolio_history.pct_change().dropna()
    sharpe_ratio = (daily_strategy_returns.mean() / daily_strategy_returns.std()) * np.sqrt(252) if daily_strategy_returns.std() != 0 else 0.0
    rolling_max = portfolio_history.cummax()
    daily_drawdown = portfolio_history / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    buy_hold_data = test_data['Close'].loc[portfolio_history.index]
    buy_hold_return = (buy_hold_data.iloc[-1] / buy_hold_data.iloc[0]) - 1
    print(f"--- Strategy Performance ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"\n--- Benchmark ---")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")

portfolio_history = run_backtest(preds, df.loc[X_test.index])
evaluate_strategy(portfolio_history, df)

# --- 7. Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))
portfolio_history.plot(label='LGBM Strategy (Final Features)', color='royalblue')
buy_hold_equity = (df['Close'].loc[portfolio_history.index].pct_change().add(1).cumprod() * 100000)
buy_hold_equity.plot(label='Buy & Hold', color='gray', linestyle='--')
plt.title(f"{ticker} Strategy vs. Buy & Hold", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.show()