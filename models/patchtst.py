# models/patchtst.py
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
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.train_config['device'] == 'cuda' else "cpu")
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
        scaled_target = self.scaler.fit_transform(y_train.values.reshape(-1, 1))
        X_seq, y_seq = self._create_sequences(scaled_target)

        self.model = PyTorchPatchTST(
            context_length=self.window, prediction_length=self.horizon,
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

    # --- START OF NEW IMPLEMENTATION ---
    def predict(self, X_test: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """
        Generates predictions using a rolling-window approach.
        """
        print("--- Predicting with PatchTSTModel (Rolling Window) ---")
        self.model.eval()

        # Combine the history (y_train) with the test period for continuous sequencing
        # We use the 'close' price from the original test data for the true values
        historical_data = y_train.values
        
        # Scale the entire history using the scaler fitted on training data
        scaled_history = self.scaler.transform(historical_data.reshape(-1, 1))
        
        predictions_scaled = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                # The end of our input sequence is the i-th step into the test set
                sequence_end_index = len(y_train) + i
                # The start is 'window' days before that
                sequence_start_index = sequence_end_index - self.window
                
                # We need to get the input sequence from the full original data, not X_test
                input_sequence_scaled = scaled_history[sequence_start_index:sequence_end_index]
                
                # Convert to tensor and add batch/feature dimensions
                input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get the prediction (horizon)
                prediction_horizon_scaled = self.model(input_tensor)
                
                # We only care about the t+1 prediction
                first_step_prediction = prediction_horizon_scaled[0, 0].item()
                predictions_scaled.append(first_step_prediction)

        # Inverse transform the collected predictions
        predictions_scaled_np = np.array(predictions_scaled).reshape(-1, 1)
        final_predictions = self.scaler.inverse_transform(predictions_scaled_np)
        
        return final_predictions.flatten()
    # --- END OF NEW IMPLEMENTATION ---

# ... [PyTorchPatchTST class remains unchanged] ...
class PyTorchPatchTST(nn.Module):
    def __init__(self, context_length, prediction_length, patch_len, stride, model_dim, num_heads, num_layers):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (context_length - patch_len) // stride + 1
        self.embedding = nn.Linear(patch_len, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.head = nn.Linear(model_dim * self.num_patches, prediction_length)

    def forward(self, x):
        x_sq = x.squeeze(-1)
        patches = x_sq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        embedding = self.embedding(patches)
        embedding = embedding + self.pos_encoder
        transformer_output = self.transformer_encoder(embedding)
        transformer_output = transformer_output.reshape(transformer_output.size(0), -1)
        prediction = self.head(transformer_output)
        return prediction