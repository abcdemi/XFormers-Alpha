# models/patchtst.py
"""
A PatchTST model, adapted for the project structure.
This version includes a robust forward pass to handle dynamic input shapes.
Reference: https://arxiv.org/abs/2211.14730
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class PatchTSTModel:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']
        self.window = config['window']
        self.horizon = config['horizon']
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.train_config.get('device') == 'cuda' else "cpu")
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window - self.horizon):
            X.append(data[i:(i + self.window)])
            y.append(data[i + self.window : i + self.window + self.horizon])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"--- Training PatchTSTModel on {self.device} ---")
        target_series_train = y_train.values.reshape(-1, 1)
        scaled_target = self.scaler.fit_transform(target_series_train)
        X_seq, y_seq = self._create_sequences(scaled_target)

        self.model = PyTorchPatchTST(
            prediction_length=self.horizon,
            patch_len=self.model_config['patch_len'], stride=self.model_config['stride'],
            model_dim=self.model_config['d_model'], num_heads=self.model_config['n_heads'],
            num_layers=self.model_config['depth']
        ).to(self.device)

        train_dataset = TensorDataset(X_seq, y_seq)
        train_loader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'], shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config['lr'])

        self.model.train()
        for epoch in range(self.train_config['epochs']):
            for seqs, labels in train_loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(seqs)
                loss = criterion(y_pred, labels.squeeze(-1))
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.train_config["epochs"]}, Loss: {loss.item():.5f}')

    # --- START OF DEFINITIVE FIX ---
    def predict(self, X_test: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """
        Generates predictions using a robust rolling-window approach.
        """
        print("--- Predicting with PatchTSTModel (Rolling Window) ---")
        self.model.eval()

        # Get the historical 'close' prices from the training set
        historical_data = y_train.values.reshape(-1, 1)
        
        # Scale the historical data
        scaled_history = self.scaler.transform(historical_data)
        
        # This will hold our final predictions
        predictions_scaled = []
        
        # The first input for our rolling window is the last 'window' of training data
        current_sequence = scaled_history[-self.window:].tolist()

        with torch.no_grad():
            for i in range(len(X_test)):
                # Convert the current sequence (a list of lists) to a NumPy array
                input_array = np.array(current_sequence)
                
                # Convert to a 3D tensor: (batch_size, sequence_length, feature_dimension)
                input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # The shape should now be (1, 96, 1), which is a valid 3D tensor.
                
                # Get the multi-step prediction
                prediction_horizon_scaled = self.model(input_tensor)
                
                # We only care about the first prediction for our t+1 strategy
                first_step_prediction = prediction_horizon_scaled[0, 0].item()
                predictions_scaled.append(first_step_prediction)
                
                # --- Roll the window forward ---
                # To get the next prediction, we need the true value for the current day.
                # A true live system wouldn't have this, but a backtest MUST use it
                # to avoid propagating its own errors.
                true_next_value_scaled = self.scaler.transform(X_test.iloc[i:i+1][['close']].values)[0]
                
                # Remove the oldest value and add the new true value
                current_sequence.pop(0)
                current_sequence.append(true_next_value_scaled)

        # Inverse transform the collected predictions
        predictions_scaled_np = np.array(predictions_scaled).reshape(-1, 1)
        final_predictions = self.scaler.inverse_transform(predictions_scaled_np)
        
        return final_predictions.flatten()
    # --- END OF DEFINITIVE FIX ---

# ... [PyTorchPatchTST class remains unchanged] ...
class PyTorchPatchTST(nn.Module):
    def __init__(self, prediction_length, patch_len, stride, model_dim, num_heads, num_layers):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.model_dim = model_dim
        self.prediction_length = prediction_length
        self.embedding = nn.Linear(patch_len, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.head = None
        self.pos_encoder = None

    def forward(self, x):
        x_sq = x.squeeze(-1)
        patches = x_sq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        num_patches = patches.shape[1]
        if self.pos_encoder is None or self.pos_encoder.shape[1] != num_patches:
            self.pos_encoder = nn.Parameter(torch.randn(1, num_patches, self.model_dim), requires_grad=True).to(x.device)
        embedding = self.embedding(patches)
        embedding = embedding + self.pos_encoder
        transformer_output = self.transformer_encoder(embedding)
        transformer_output_flat = transformer_output.reshape(transformer_output.size(0), -1)
        if self.head is None:
            in_features = transformer_output_flat.shape[1]
            self.head = nn.Linear(in_features, self.prediction_length).to(x.device)
        prediction = self.head(transformer_output_flat)
        return prediction