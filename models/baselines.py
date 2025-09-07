# models/baselines.py
"""
Baseline models: Linear Regression, LightGBM, and a standard LSTM.
Each class is designed to be called by the main training script.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class LinearBaseline:
    """A simple linear regression model."""
    def __init__(self, config: dict):
        self.model = LinearRegression()
        self.config = config

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("--- Training Linear Baseline ---")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with Linear Baseline ---")
        return self.model.predict(X_test)

class TreeBaseline:
    """A tree-based model using LightGBM."""
    def __init__(self, config: dict):
        # Extract model-specific params from the main config
        model_params = config['model']
        self.model = lgb.LGBMRegressor(random_state=config['seed'], **model_params)
        self.config = config

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("--- Training Tree Baseline (LightGBM) ---")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with Tree Baseline ---")
        return self.model.predict(X_test)

class LSTMBaseline:
    """A standard LSTM model using PyTorch."""
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']
        self.window = config['window']
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.train_config['device'] == 'cuda' else "cpu")
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i:(i + self.window)])
            y.append(data[i + self.window, -1]) # Target is the last column
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"--- Training LSTM Baseline on {self.device} ---")
        
        # LSTM needs the target for sequence creation
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        scaled_data = self.scaler.fit_transform(train_data)
        X_seq, y_seq = self._create_sequences(scaled_data)
        
        self.model = PyTorchLSTM(
            input_size=X_seq.shape[2],
            hidden_layer_size=self.model_config['hidden_size'],
            num_layers=self.model_config['depth']
        ).to(self.device)

        train_dataset = TensorDataset(X_seq, y_seq)
        train_loader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'], shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config['lr'])

        self.model.train()
        for epoch in range(self.train_config['epochs']):
            for seqs, labels in train_loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(seqs)
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.train_config["epochs"]}, Loss: {loss.item():.5f}')

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with LSTM Baseline ---")
        self.model.eval()
        
        # We need to scale the test data using the scaler fitted on training data
        # Note: We don't have the target for the test set, so we add a dummy column
        test_data = X_test.copy()
        test_data['target'] = 0 # Dummy column
        
        scaled_data = self.scaler.transform(test_data)
        X_seq, _ = self._create_sequences(scaled_data)

        with torch.no_grad():
            predictions_scaled = self.model(X_seq.to(self.device)).cpu().numpy()

        # Inverse transform requires the same number of features as the scaler was fit on
        dummy_array = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
        dummy_array[:, -1] = predictions_scaled.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, -1]
        
        return predictions

class PyTorchLSTM(nn.Module):
    """The PyTorch LSTM model architecture."""
    def __init__(self, input_size, hidden_layer_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :]) # Predict from last time step
        return predictions