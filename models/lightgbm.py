# === Kaggle-Style Trading Strategy Simulation (Corrected for TypeError in Evaluation) ===
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

# --- 2. Data Acquisition and Feature Engineering ---
print("Step 2: Downloading data and engineering features...")
ticker = "GOOGL"
df = yf.download(ticker, period="10y", interval="1d", auto_adjust=True)

# --- START OF FIX ---
# yfinance can return columns as a MultiIndex. This flattens it to a simple index.
# This is the most robust way to prevent downstream errors.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1) # Drop the 'Ticker' level, e.g. ('Close', 'GOOGL') -> 'Close'
    print("Flattened MultiIndex columns to a simple index.")
# --- END OF FIX ---


def create_features(data):
    """Create time-series features from the price data."""
    df_feat = data.copy()
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f'lag_return_{lag}'] = df_feat['Close'].pct_change(lag)
    for window in [5, 10, 20, 60]:
        df_feat[f'rolling_mean_{window}'] = df_feat['Close'].rolling(window=window).mean()
        df_feat[f'rolling_std_{window}'] = df_feat['Close'].rolling(window=window).std()
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['rsi'] = 100 - (100 / (1 + rs))
    df_feat = df_feat.dropna()
    return df_feat

df_features = create_features(df)

# --- 3. Target Definition and Data Splitting ---
df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
df_features = df_features.dropna()

X = df_features.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'target'], axis=1)
y = df_features['target']

# Sanitize feature names for LightGBM compatibility.
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]

# Split into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 4. Model Training ---
print("\nStep 4: Training the LightGBM model...")
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# --- 5. Backtesting Simulation ---
print("\nStep 5: Running the backtesting simulation...")

def run_backtest(predictions, test_data):
    initial_capital = 100000.0
    portfolio_history = []
    daily_returns = test_data['Close'].pct_change().dropna()
    aligned_preds = predictions[:len(daily_returns)]
    
    # We start with capital and no position
    portfolio_value = initial_capital
    
    for i in range(len(daily_returns)):
        signal = aligned_preds[i]
        market_return = daily_returns.iloc[i]
        
        if signal == 1:
            strategy_return = market_return
        else:
            strategy_return = 0.0
            
        portfolio_value *= (1 + strategy_return)
        portfolio_history.append(portfolio_value)
        
    return pd.Series(portfolio_history, index=daily_returns.index)

portfolio_history = run_backtest(preds, df.loc[X_test.index])

# --- 6. Evaluation and Visualization ---
print("\nStep 6: Evaluating the strategy and plotting results...")

def evaluate_strategy(portfolio_history, test_data):
    total_return = (portfolio_history.iloc[-1] / portfolio_history.iloc[0]) - 1
    daily_strategy_returns = portfolio_history.pct_change().dropna()
    
    # Handle the case where std is zero (no trades or flat returns)
    if daily_strategy_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_strategy_returns.mean() / daily_strategy_returns.std()) * np.sqrt(252)

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

evaluate_strategy(portfolio_history, df)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))
portfolio_history.plot(label='LGBM Strategy', color='royalblue')
buy_hold_equity = (df['Close'].loc[portfolio_history.index].pct_change().add(1).cumprod() * 100000)
buy_hold_equity.plot(label='Buy & Hold', color='gray', linestyle='--')
plt.title(f"{ticker} Strategy vs. Buy & Hold", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.show()