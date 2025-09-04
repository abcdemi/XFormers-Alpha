# === Kaggle-Style Trading Strategy Simulation (v6: GridSearchCV Tuning) ===
# Requirements: pip install yfinance lightgbm pandas numpy matplotlib scikit-learn

# --- 1. Setup and Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import re

# --- 2. Data Acquisition and Feature Engineering ---
print("Step 2: Downloading data and engineering features...")
df_dict = yf.download("GOOGL SPY", period="10y", interval="1d", auto_adjust=True)
df = df_dict.loc[:, (slice(None), 'GOOGL')]
df.columns = df.columns.droplevel(1)
spy_df = df_dict.loc[:, (slice(None), 'SPY')]
spy_df.columns = spy_df.columns.droplevel(1)

def create_final_features(data, spy_data):
    df_feat = data.copy()
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f'lag_return_{lag}'] = df_feat['Close'].pct_change(lag)
    for window in [5, 10, 20, 60]:
        df_feat[f'rolling_std_{window}'] = df_feat['Close'].rolling(window=window).std()
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = ema_12 - ema_26
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    bb_window = 20
    rolling_mean_bb = df_feat['Close'].rolling(window=bb_window).mean()
    rolling_std_bb = df_feat['Close'].rolling(window=bb_window).std()
    df_feat['bb_width'] = ((rolling_mean_bb + (rolling_std_bb*2)) - (rolling_mean_bb - (rolling_std_bb*2))) / rolling_mean_bb
    df_feat['bb_percent'] = (df_feat['Close'] - (rolling_mean_bb - (rolling_std_bb*2))) / ((rolling_mean_bb + (rolling_std_bb*2)) - (rolling_mean_bb - (rolling_std_bb*2)))
    high_low = df_feat['High'] - df_feat['Low']
    high_close = np.abs(df_feat['High'] - df_feat['Close'].shift())
    low_close = np.abs(df_feat['Low'] - df_feat['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_feat['atr'] = tr.rolling(window=14).mean()
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    spy_returns = spy_data['Close'].pct_change()
    df_feat['spy_return'] = spy_returns
    df_feat['spy_rolling_std_10'] = spy_returns.rolling(window=10).std()
    df_feat['interaction_lag1_spy'] = df_feat['lag_return_1'] * df_feat['spy_return']
    df_feat = df_feat.dropna()
    return df_feat

df_features = create_final_features(df, spy_df)

# --- 3. Target Definition and Data Splitting ---
df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
df_features = df_features.dropna()
X = df_features.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'target'], axis=1)
y = df_features['target']
X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]

# Split into Train (85%) and Test (15%)
# GridSearchCV will handle the internal validation
train_size = int(0.85 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# --- 4. GridSearchCV Hyperparameter Tuning ---
print("\nStep 4: Running GridSearchCV hyperparameter tuning...")
print("This may take several minutes...")

# Define the parameter grid to search. This is a smaller grid to keep runtime reasonable.
param_grid = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'max_depth': [7, 10]
}

# Use TimeSeriesSplit for cross-validation to respect the temporal order of data
tscv = TimeSeriesSplit(n_splits=5)

# Instantiate the model and GridSearchCV
model = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt', seed=42, n_jobs=-1, verbose=-1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=tscv, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

print("GridSearchCV finished.")
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score (Accuracy):", grid_search.best_score_)

# --- 5. Training Final Model with Best Parameters ---
print("\nStep 5: Evaluating the final model...")
final_model = grid_search.best_estimator_

# Evaluate on the unseen test set
final_preds = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Final Model Accuracy on Test Set: {final_accuracy:.4f}")

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

portfolio_history = run_backtest(final_preds, df.loc[X_test.index])
evaluate_strategy(portfolio_history, df)

# --- 7. Visualization ---
plt.style.use('seaborn-v_eight')
plt.figure(figsize=(14, 7))
portfolio_history.plot(label='LGBM Strategy (GridSearched)', color='royalblue')
buy_hold_equity = (df['Close'].loc[portfolio_history.index].pct_change().add(1).cumprod() * 100000)
buy_hold_equity.plot(label='Buy & Hold', color='gray', linestyle='--')
plt.title(f"{ticker} Strategy vs. Buy & Hold", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.show()