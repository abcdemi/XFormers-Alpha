# xformers_alpha/baselines.py
"""
Baseline forecasting models for financial time series.

Includes implementations for:
1. Linear Regression Model (using lagged features)
2. Gradient Boosting Tree Model (LightGBM, using lagged features)
3. Long Short-Term Memory (LSTM) Neural Network (using PyTorch)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from typing import List

# --- PyTorch specific imports ---
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ------------------
# Helper Function for Feature Engineering
# ------------------

def create_lagged_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Creates a new DataFrame with lagged features from the 'close' column.
    """
    df_new = df.copy()
    for i in range(1, lags + 1):
        df_new[f'lag_{i}'] = df_new['close'].shift(i)
    df_new.dropna(inplace=True)
    return df_new

# ------------------
# Model 1: Linear Baseline (No changes)
# ------------------

class LinearBaseline:
    """A linear regression model using past values to predict the next."""
    def __init__(self, lags: int = 5):
        self.lags = lags
        self.model = LinearRegression()

    def fit(self, df_train: pd.DataFrame):
        """Trains the linear regression model."""
        print("Fitting Linear Baseline...")
        featured_df = create_lagged_features(df_train, self.lags)
        X_train = featured_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        y_train = featured_df['close']
        self.model.fit(X_train, y_train)

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Makes predictions on new data."""
        print("Predicting with Linear Baseline...")
        featured_df = create_lagged_features(df_test, self.lags)
        X_test = featured_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        return self.model.predict(X_test)

# ------------------
# Model 2: Tree-based Baseline (No changes)
# ------------------

class TreeBaseline:
    """A Gradient Boosting model (LightGBM) using past values."""
    def __init__(self, lags: int = 5, **lgb_params):
        self.lags = lags
        self.model = lgb.LGBMRegressor(random_state=42, **lgb_params)

    def fit(self, df_train: pd.DataFrame):
        """Trains the LightGBM model."""
        print("Fitting Tree Baseline (LightGBM)...")
        featured_df = create_lagged_features(df_train, self.lags)
        X_train = featured_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        y_train = featured_df['close']
        self.model.fit(X_train, y_train)

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Makes predictions on new data."""
        print("Predicting with Tree Baseline (LightGBM)...")
        featured_df = create_lagged_features(df_test, self.lags)
        X_test = featured_df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        return self.model.predict(X_test)

# ------------------
# Model 3: LSTM Baseline (PyTorch Implementation)
# ------------------

class PyTorchLSTM(nn.Module):
    """The PyTorch model architecture for the LSTM."""
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(input_seq)
        # Pass the output of the last time step to the linear layer
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class LSTMBaseline:
    """An LSTM model for time series forecasting using PyTorch."""
    def __init__(self, sequence_len: int = 10, epochs: int = 20, batch_size: int = 32, hidden_size: int = 50):
        self.sequence_len = sequence_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Set device (use GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for PyTorch LSTM: {self.device}")
        
        self.model = PyTorchLSTM(hidden_layer_size=hidden_size).to(self.device)

    def _create_sequences(self, data: np.ndarray) -> (torch.Tensor, torch.Tensor):
        """Prepares data into sequences for LSTM and returns PyTorch Tensors."""
        X, y = [], []
        for i in range(len(data) - self.sequence_len):
            X.append(data[i:(i + self.sequence_len)])
            y.append(data[i + self.sequence_len])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        return X, y

    def fit(self, df_train: pd.DataFrame):
        """Scales data, prepares sequences, and trains the LSTM model."""
        print("Fitting LSTM Baseline (PyTorch)...")
        close_prices = df_train['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        X, y = self._create_sequences(scaled_data)

        # Create DataLoader for batching
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train() # Set model to training mode
        for epoch in range(self.epochs):
            for seqs, labels in train_loader:
                # Move data to the appropriate device
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(seqs)
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}')

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Makes predictions using the trained LSTM model."""
        print("Predicting with LSTM Baseline (PyTorch)...")
        close_prices = df_test['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(close_prices)
        X_test, _ = self._create_sequences(scaled_data)

        self.model.eval() # Set model to evaluation mode
        predictions_scaled = []
        with torch.no_grad():
            for i in range(len(X_test)):
                seq = X_test[i:i+1].to(self.device) # Get one sequence and send to device
                pred = self.model(seq)
                predictions_scaled.append(pred.cpu().numpy())
        
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()