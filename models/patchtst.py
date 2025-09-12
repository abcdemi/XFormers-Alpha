# models/patchtst.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
        self.scaler = StandardScaler()
        self.target_col_idx = -1

    def _create_sequences(self, data: pd.DataFrame):
        data_np = data.values
        X, y = [], []
        for i in range(len(data_np) - self.window - self.horizon):
            X.append(data_np[i:(i + self.window)])
            y.append(data_np[i + self.window : i + self.window + self.horizon, self.target_col_idx])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"--- Training MULTIVARIATE PatchTSTModel on {self.device} ---")
        
        train_df = X_train.copy()
        train_df['close'] = y_train
        self.target_col_idx = train_df.columns.get_loc('close')
        
        scaled_data = self.scaler.fit_transform(train_df)
        scaled_df = pd.DataFrame(scaled_data, columns=train_df.columns)
        X_seq, y_seq = self._create_sequences(scaled_df)

        self.model = PyTorchPatchTST(
            context_length=self.window,
            prediction_length=self.horizon,
            num_channels=X_seq.shape[2],
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
                if y_pred.shape != labels.shape and len(labels.shape) == 1:
                    labels = labels.unsqueeze(1)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.train_config["epochs"]}, Loss: {loss.item():.5f}')

    def predict(self, X_test: pd.DataFrame, X_train: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with MULTIVARIATE PatchTSTModel (Rolling Window) ---")
        self.model.eval()

        # Combine historical and test features for a continuous data stream
        # This is the correct way to handle the feature set for scaling
        full_X_df = pd.concat([X_train, X_test], ignore_index=True)
        
        # Scale the ENTIRE feature set using the scaler fitted during training
        scaled_full_X = self.scaler.transform(full_X_df)
        
        predictions_scaled = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                sequence_end_index = len(X_train) + i
                sequence_start_index = sequence_end_index - self.window
                
                input_sequence_scaled = scaled_full_X[sequence_start_index:sequence_end_index]
                
                input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                prediction_horizon_scaled = self.model(input_tensor)
                
                # We only need the t+1 prediction for our strategy
                first_step_prediction = prediction_horizon_scaled[0, 0].item()
                predictions_scaled.append(first_step_prediction)

        predictions_scaled_np = np.array(predictions_scaled).reshape(-1, 1)
        
        # Create a dummy array with the correct number of features for inverse scaling
        dummy_array = np.zeros((len(predictions_scaled_np), self.scaler.n_features_in_))
        # Place our single prediction into the column that corresponds to the 'close' price
        dummy_array[:, self.target_col_idx] = predictions_scaled_np.flatten()
        
        # Inverse transform the dummy array to get the predictions in the original price scale
        final_predictions = self.scaler.inverse_transform(dummy_array)[:, self.target_col_idx]
        
        return final_predictions


class PyTorchPatchTST(nn.Module):
    def __init__(self, context_length, prediction_length, num_channels, patch_len, stride, model_dim, num_heads, num_layers):
        super().__init__()
        self.num_channels = num_channels
        self.prediction_length = prediction_length
        self.patching_layers = nn.ModuleList()
        # Create a separate patching layer for each feature (channel)
        for _ in range(num_channels):
            self.patching_layers.append(PatchingLayer(context_length, patch_len, stride, model_dim))
        
        # This single Transformer encoder will process the concatenated patch embeddings
        num_patches = (context_length - patch_len) // stride + 1
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=0.1),
            num_layers=num_layers
        )
        
        # The head takes the flattened output of all channels and patches
        head_in_features = num_channels * num_patches * model_dim
        self.head = nn.Linear(head_in_features, prediction_length)

    def forward(self, x):
        # x shape: (B, L, C)
        channel_outputs = []
        for i in range(self.num_channels):
            # Each layer processes one channel: (B, L) -> (B, num_patches, model_dim)
            channel_out = self.patching_layers[i](x[:, :, i])
            channel_outputs.append(channel_out)
        
        # Concatenate along the patch dimension
        x_patched = torch.cat(channel_outputs, dim=1) # Shape: (B, C * num_patches, model_dim)
        
        # The Transformer processes this long sequence of patches from all channels
        transformer_output = self.transformer_encoder(x_patched)

        # Flatten for the final head
        x_flat = transformer_output.reshape(transformer_output.size(0), -1) 
        prediction = self.head(x_flat)
        return prediction

class PatchingLayer(nn.Module):
    def __init__(self, context_length, patch_len, stride, model_dim):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        num_patches = (context_length - patch_len) // stride + 1
        self.embedding = nn.Linear(patch_len, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_patches, model_dim))

    def forward(self, x):
        # x shape: (B, L)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        embedding = self.embedding(patches)
        embedding = embedding + self.pos_encoder
        return embedding