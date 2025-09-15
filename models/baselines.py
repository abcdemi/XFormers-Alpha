# ... (Keep LinearBaseline and TreeBaseline classes as they are) ...
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import re

class LinearBaseline: # ... (no changes)
    def __init__(self, config: dict):
        self.model = LinearRegression(); self.config = config
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("--- Training Linear Baseline ---"); self.model.fit(X_train, y_train)
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with Linear Baseline ---"); return self.model.predict(X_test)

class TreeBaseline: # ... (no changes)
    def __init__(self, config: dict):
        model_params = config['model']; model_params.pop('name', None)
        self.model = lgb.LGBMRegressor(random_state=config['seed'], **model_params); self.config = config
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("--- Training Tree Baseline (LightGBM) ---"); self.model.fit(X_train, y_train)
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with Tree Baseline ---"); return self.model.predict(X_test)


# --- START OF REPLACEMENT ---

class LSTMBaseline:
    """A standard LSTM model using PyTorch with a robust prediction method."""
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']
        self.window = config['window']
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.train_config.get('device') == 'cuda' else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.target_col_idx = -1

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i:(i + self.window)])
            y.append(data[i + self.window, self.target_col_idx])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"--- Training Multivariate LSTM Baseline on {self.device} ---")
        
        train_df = X_train.copy()
        train_df['target'] = y_train
        
        # --- START OF FIX ---
        # Save the target column index to the class instance so it's available during prediction
        self.target_col_idx = train_df.columns.get_loc('target')
        # --- END OF FIX ---
        
        scaled_data = self.scaler.fit_transform(train_df)
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

    def predict(self, X_test: pd.DataFrame, X_train: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using a multivariate rolling-window approach.
        """
        print("--- Predicting with Multivariate LSTM (Rolling Window) ---")
        self.model.eval()

        full_X_df = pd.concat([X_train, X_test], ignore_index=True)
        full_X_df['target'] = 0 # Add dummy target column
        
        scaled_full_X = self.scaler.transform(full_X_df)
        
        predictions_scaled = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                sequence_end_index = len(X_train) + i
                sequence_start_index = sequence_end_index - self.window
                
                input_sequence_scaled = scaled_full_X[sequence_start_index:sequence_end_index]
                
                input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                prediction_scaled = self.model(input_tensor)
                
                predictions_scaled.append(prediction_scaled.item())

        predictions_scaled_np = np.array(predictions_scaled).reshape(-1, 1)
        
        dummy_array = np.zeros((len(predictions_scaled_np), self.scaler.n_features_in_))
        # This line will now work correctly
        dummy_array[:, self.target_col_idx] = predictions_scaled_np.flatten()
        
        final_predictions = self.scaler.inverse_transform(dummy_array)[:, self.target_col_idx]
        
        return final_predictions

class PyTorchLSTM(nn.Module):
    # ... (unchanged) ...
    def __init__(self, input_size, hidden_layer_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
