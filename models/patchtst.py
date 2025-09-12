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

    def _create_sequences(self, data: pd.DataFrame): # Expects a DataFrame now
        X, y = [], []
        
        # --- START OF FIX ---
        # Convert the DataFrame to a NumPy array BEFORE the loop for efficient slicing
        data_np = data.values
        # --- END OF FIX ---

        for i in range(len(data_np) - self.window - self.horizon):
            X.append(data_np[i:(i + self.window)])
            # The target is the 'close' price 'horizon' steps ahead
            # Now this NumPy-style slicing will work correctly
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
                # For horizon=1, label shape might be (B,), pred is (B,1), so we unsqueeze
                if y_pred.shape != labels.shape and len(labels.shape) == 1:
                    labels = labels.unsqueeze(1)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.train_config["epochs"]}, Loss: {loss.item():.5f}')

    def predict(self, X_test: pd.DataFrame, X_train: pd.DataFrame) -> np.ndarray:
        print("--- Predicting with Multivariate PatchTSTModel (Rolling Window) ---")
        self.model.eval()

        # Combine train and test features for a continuous history
        # We need to add a dummy 'close' column to X_train for concat
        X_train_with_dummy = X_train.copy()
        X_train_with_dummy['close'] = 0 # This will be ignored by the scaler
        
        full_X = pd.concat([X_train_with_dummy, X_test], ignore_index=True)
        
        scaled_full_X = self.scaler.transform(full_X)
        
        predictions_scaled = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                sequence_end_index = len(X_train) + i
                sequence_start_index = sequence_end_index - self.window
                
                input_sequence_scaled = scaled_full_X[sequence_start_index:sequence_end_index]
                
                input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                prediction_horizon_scaled = self.model(input_tensor)
                
                first_step_prediction = prediction_horizon_scaled[0, 0].item()
                predictions_scaled.append(first_step_prediction)

        predictions_scaled_np = np.array(predictions_scaled).reshape(-1, 1)
        
        dummy_array = np.zeros((len(predictions_scaled_np), self.scaler.n_features_in_))
        dummy_array[:, self.target_col_idx] = predictions_scaled_np.flatten()
        
        final_predictions = self.scaler.inverse_transform(dummy_array)[:, self.target_col_idx]
        
        return final_predictions.flatten()


class PyTorchPatchTST(nn.Module):
    # This sub-class remains unchanged
    def __init__(self, context_length, prediction_length, num_channels, patch_len, stride, model_dim, num_heads, num_layers):
        super().__init__()
        self.num_channels = num_channels
        self.prediction_length = prediction_length
        self.model_dim = model_dim
        self.patching_layers = nn.ModuleList()
        for _ in range(num_channels):
            self.patching_layers.append(PatchingLayer(context_length, patch_len, stride, model_dim))
        
        self.head = nn.Linear(num_channels * ((context_length - patch_len) // stride + 1) * model_dim, prediction_length)

    def forward(self, x):
        channel_outputs = []
        for i in range(self.num_channels):
            channel_out = self.patching_layers[i](x[:, :, i])
            channel_outputs.append(channel_out)
        x_patched = torch.cat(channel_outputs, dim=1)
        x_flat = x_patched.reshape(x_patched.size(0), -1) 
        prediction = self.head(x_flat)
        return prediction

class PatchingLayer(nn.Module):
    # This sub-class remains unchanged
    def __init__(self, context_length, patch_len, stride, model_dim):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        num_patches = (context_length - patch_len) // stride + 1
        self.embedding = nn.Linear(patch_len, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_patches, model_dim))

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        embedding = self.embedding(patches)
        embedding = embedding + self.pos_encoder
        return embedding