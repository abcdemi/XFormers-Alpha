# In your notebook: 01_data_explore.ipynb

# 1. --- IMPORTS AND DATA LOADING ---
import pandas as pd
from xformers_alpha.data.loaders import load_yfinance
from models.baselines import LinearBaseline, TreeBaseline, LSTMBaseline, create_lagged_features

# Load data for a single stock
try:
    data_df = load_yfinance('GOOGL', start='2020-01-01', end='2023-12-31')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# 2. --- SPLIT DATA INTO TRAINING AND TESTING SETS ---
# We will use the first 80% of the data for training and the last 20% for testing.
split_index = int(len(data_df) * 0.8)
train_df = data_df.iloc[:split_index]
test_df = data_df.iloc[split_index:]

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape:  {test_df.shape}")


# 3. --- RUN THE BASELINE MODELS ---

# --- Linear Baseline ---
linear_model = LinearBaseline(lags=10)
linear_model.fit(train_df)
# Note: The predict function will internally create lags, so the output
# will be shorter than the input test_df. We need to align them.
linear_predictions = linear_model.predict(test_df)
# The actual values corresponding to the predictions
y_test_linear = create_lagged_features(test_df, lags=10)['close'].values
print(f"Linear predictions generated. Shape: {linear_predictions.shape}")


# --- Tree Baseline (LightGBM) ---
#tree_model = TreeBaseline(lags=10, n_estimators=100)
#tree_model.fit(train_df)
#tree_predictions = tree_model.predict(test_df)
#y_test_tree = create_lagged_features(test_df, lags=10)['close'].values
#print(f"Tree predictions generated. Shape: {tree_predictions.shape}")


# --- LSTM Baseline ---
# Note: LSTMs can take a moment to train.
lstm_model = LSTMBaseline(sequence_len=15, epochs=25)
lstm_model.fit(train_df)
lstm_predictions = lstm_model.predict(test_df)
# The actual values for LSTM start after the first sequence_len period
y_test_lstm = test_df['close'].values[lstm_model.sequence_len:]
print(f"LSTM predictions generated. Shape: {lstm_predictions.shape}")


# 4. --- VISUALIZE PREDICTIONS (Example for one model) ---
import matplotlib.pyplot as plt

# Let's visualize the LSTM model's predictions
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(test_df['timestamp'].iloc[-len(y_test_lstm):], y_test_lstm, label='Actual Prices', color='blue')
ax.plot(test_df['timestamp'].iloc[-len(y_test_lstm):], lstm_predictions, label='Predicted Prices (LSTM)', color='orange', linestyle='--')
ax.plot(test_df['timestamp'].iloc[-len(y_test_linear):], linear_predictions, label='Predicted Prices (Linear)', color='green', linestyle='-.')

ax.set_title('Google (GOOGL) Price Prediction - Baselines', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend()
plt.show()