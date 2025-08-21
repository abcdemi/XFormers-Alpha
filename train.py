# In your notebook (e.g., 01_data_explore.ipynb)

# 1. --- IMPORTS AND DATA LOADING ---
import pandas as pd
import matplotlib.pyplot as plt
from xformers_alpha.data.loaders import load_yfinance
from models.baselines import LinearBaseline, TreeBaseline, LSTMBaseline

# Load the full dataset
try:
    full_df = load_yfinance('GOOGL', start='2020-01-01', end='2023-12-31')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# 2. --- RUN THE NEW BASELINE MODELS ---

# -- Linear Model (now uses technical features) --
linear_model = LinearBaseline()
linear_predictions, linear_y_test = linear_model.train_and_evaluate(full_df, test_size=0.2)

# -- Tree Model (now uses technical features) --
tree_model = TreeBaseline(n_estimators=100)
tree_predictions, tree_y_test = tree_model.train_and_evaluate(full_df, test_size=0.2)

# -- LSTM Model (now uses multivariate input) --
lstm_model = LSTMBaseline(sequence_len=15, epochs=25)
lstm_predictions, lstm_y_test = lstm_model.train_and_evaluate(full_df, test_size=0.2)

# 3. --- VISUALIZE PREDICTIONS ---
# This part requires careful alignment of timestamps, as the feature engineering
# changes the length of the dataframes.

# The ground truth for Linear/Tree is the same length
test_start_index_tree = len(full_df) - len(tree_y_test)
timestamps_tree = full_df['timestamp'].iloc[test_start_index_tree:]

# LSTM alignment is different
test_start_index_lstm = len(full_df) - len(lstm_y_test)
timestamps_lstm = full_df['timestamp'].iloc[test_start_index_lstm:]

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(timestamps_tree, tree_y_test, label='Actual Prices', color='blue', linewidth=2)
ax.plot(timestamps_tree, linear_predictions, label='Predicted (Linear)', color='green', linestyle=':')
ax.plot(timestamps_tree, tree_predictions, label='Predicted (Tree)', color='orange', linestyle='--')
ax.plot(timestamps_lstm, lstm_predictions, label='Predicted (LSTM)', color='red', linestyle='-.')
ax.set_title('GOOGL Price Prediction (Advanced Features)', fontsize=16)
ax.legend()
plt.show()