# === Kaggle-Style Trading Strategy Simulation (v3: Feature Importance Analysis) ===
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

# --- 3. Advanced Feature Engineering ---
print("Step 3: Engineering advanced features...")
def create_more_features(data, spy_data):
    df_feat = data.copy()
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f'lag_return_{lag}'] = df_feat['Close'].pct_change(lag)
    for window in [5, 10, 20, 60]:
        df_feat[f'rolling_mean_{window}'] = df_feat['Close'].rolling(window=window).mean()
        df_feat[f'rolling_std_{window}'] = df_feat['Close'].rolling(window=window).std()
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = ema_12 - ema_26
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    bb_window = 20
    rolling_mean = df_feat['Close'].rolling(window=bb_window).mean()
    rolling_std = df_feat['Close'].rolling(window=bb_window).std()
    df_feat['bb_upper'] = rolling_mean + (rolling_std * 2)
    df_feat['bb_lower'] = rolling_mean - (rolling_std * 2)
    df_feat['bb_width'] = (df_feat['bb_upper'] - df_feat['bb_lower']) / rolling_mean
    df_feat['bb_percent'] = (df_feat['Close'] - df_feat['bb_lower']) / (df_feat['bb_upper'] - df_feat['bb_lower'])
    high_low = df_feat['High'] - df_feat['Low']
    high_close = np.abs(df_feat['High'] - df_feat['Close'].shift())
    low_close = np.abs(df_feat['Low'] - df_feat['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_feat['atr'] = tr.rolling(window=14).mean()
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['spy_return'] = spy_data['Close'].pct_change()
    df_feat = df_feat.dropna()
    return df_feat

df_features = create_more_features(df, spy_df)

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
print("\nStep 5: Training the LightGBM model with new features...")
params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 50,
    'max_depth': 7, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.8, 'subsample': 0.8
}
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# --- 6. Feature Importance Analysis ---
print("\nStep 6: Analyzing feature importance...")
plt.style.use('seaborn-v0_8-whitegrid')
# We will use 'gain' as the importance type, which is more informative than 'split'.
lgb.plot_importance(model, importance_type='gain', max_num_features=20, figsize=(10, 8), 
                    title='LightGBM Feature Importance (Gain)')
plt.show()

print("\nExplanation of the Feature Importance Plot:")
print("- This plot shows which features the model found most valuable for making accurate predictions.")
print("- 'Gain' means the total improvement to accuracy brought by a feature.")
print("- A higher gain indicates a more important feature.")
print("- This helps us understand the model's 'thinking' and confirms if our new features are useful.")

# --- 7. Backtesting Simulation ---
print("\nStep 7: Running the backtesting simulation...")
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

portfolio_history = run_backtest(preds, df.loc[X_test.index])

# --- 8. Evaluation and Visualization ---
print("\nStep 8: Evaluating the strategy and plotting results...")
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

evaluate_strategy(portfolio_history, df)

plt.figure(figsize=(14, 7))
portfolio_history.plot(label='LGBM Strategy (Advanced Features)', color='royalblue')
buy_hold_equity = (df['Close'].loc[portfolio_history.index].pct_change().add(1).cumprod() * 100000)
buy_hold_equity.plot(label='Buy & Hold', color='gray', linestyle='--')
plt.title(f"{ticker} Strategy vs. Buy & Hold", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.show()