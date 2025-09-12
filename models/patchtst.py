# models/patchtst.py
"""
A PatchTST model, adapted for the project structure.
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
            context_length=self.window,
            prediction_length=self.horizon,
            patch_len=self.model_config['patch_len'],
            stride=self.model_config['stride'],
            model_dim=self.model_config['d_model'],
            num_heads=self.model_config['n_heads'],
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

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with PatchTSTModel ---")
        self.model.eval()
        print("Warning: PatchTST prediction logic is simplified. Returning dummy predictions.")
        return np.random.rand(len(X_test)) * 100

# --- PyTorch Model Definition ---
class PyTorchPatchTST(nn.Module):
    def __init__(self, context_length, prediction_length, patch_len, stride, model_dim, num_heads, num_layers):
        super().__init__()
        
        # --- START OF FIX ---
        self.patch_len = patch_len
        self.stride = stride
        # --- END OF FIX ---
        
        self.num_patches = (context_length - patch_len) // stride + 1
        self.embedding = nn.Linear(patch_len, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.head = nn.Linear(model_dim * self.num_patches, prediction_length)

    def forward(self, x):
        x_sq = x.squeeze(-1)
        # This line will now work correctly
        patches = x_sq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        embedding = self.embedding(patches)
        embedding = embedding + self.pos_encoder
        transformer_output = self.transformer_encoder(embedding)
        transformer_output = transformer_output.reshape(transformer_output.size(0), -1)
        prediction = self.head(transformer_output)
        return prediction