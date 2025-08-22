# models/baselines.py
"""
Baseline forecasting models using technical indicators and multivariate inputs.
- Linear/Tree models use technical analysis features instead of simple lags.
- LSTM model uses a multivariate input (open, high, low, close, volume).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import re

# PyTorch specific imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Helper Function (No changes needed here)
def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new['sma_10'] = df_new['close'].rolling(10).mean()
    df_new['sma_30'] = df_new['close'].rolling(30).mean()
    delta = df_new['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_new['rsi'] = 100 - (100 / (1 + rs))
    df_new['day_of_week'] = df_new['timestamp'].dt.dayofweek
    df_new['month'] = df_new['timestamp'].dt.month
    df_new['target'] = df_new['close'].shift(-1)
    df_new.dropna(inplace=True)
    return df_new

# ------------------
# Model 1: Linear Baseline (No changes, as it doesn't need sanitization)
# ------------------
class LinearBaseline:
    def __init__(self):
        self.model = LinearRegression()

    def train_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2):
        print("--- Running Linear Baseline (with Technical Features) ---")
        featured_df = create_technical_features(df)
        y = featured_df['target']
        X = featured_df.drop(columns=['target', 'timestamp'])
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        return predictions, y_test.values

# ------------------
# Model 2: Tree-based Baseline (WITH FIX)
# ------------------
class TreeBaseline:
    def __init__(self, **lgb_params):
        self.model = lgb.LGBMRegressor(random_state=42, **lgb_params)

    def train_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2):
        print("--- Running Tree Baseline (with Technical Features) ---")
        featured_df = create_technical_features(df)
        y = featured_df['target']
        X = featured_df.drop(columns=['target', 'timestamp'])

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
        
        # --- FIX: Robustly handle potential tuples in column names ---
        sanitized_names = []
        for col in X_train.columns:
            # Check if the column name is a tuple
            if isinstance(col, tuple):
                # If so, join it into a single string
                col_str = '_'.join(map(str, col))
            else:
                col_str = str(col)
            # Sanitize the resulting string
            sanitized_names.append(re.sub(r'[^A-Za-z0-9_]+', '', col_str))

        X_train.columns = sanitized_names
        X_test.columns = sanitized_names
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        return predictions, y_test.values

# ------------------
# Model 3: Multivariate LSTM Baseline (No changes needed here)
# ------------------
class PyTorchLSTM(nn.Module):
    # ... (code is identical to previous version)
    def __init__(self, input_size, hidden_layer_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class LSTMBaseline:
    # ... (code is identical to previous version)
    def __init__(self, sequence_len: int = 15, epochs: int = 25, batch_size: int = 32, hidden_size: int = 50):
        self.sequence_len = sequence_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.hidden_size = hidden_size
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume']
    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_len):
            X.append(data[i:(i + self.sequence_len), :])
            y.append(data[i + self.sequence_len, 3])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)
    def train_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2):
        print(f"--- Running Multivariate LSTM Baseline using device: {self.device} ---")
        self.model = PyTorchLSTM(input_size=len(self.feature_cols), hidden_layer_size=self.hidden_size).to(self.device)
        split_idx = int(len(df) * (1 - test_size))
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        print(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples.")
        scaled_train_data = self.scaler.fit_transform(train_df[self.feature_cols])
        scaled_test_data = self.scaler.transform(test_df[self.feature_cols])
        X_train, y_train = self._create_sequences(scaled_train_data)
        X_test, _ = self._create_sequences(scaled_test_data)
        y_test_actual = test_df['close'].values[self.sequence_len:]
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(self.epochs):
            for seqs, labels in train_loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(seqs)
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_test.to(self.device)).cpu().numpy()
        dummy_array = np.zeros((len(predictions_scaled), len(self.feature_cols)))
        dummy_array[:, 3] = predictions_scaled.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, 3]
        return predictions, y_test_actual