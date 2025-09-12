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
        # Corrected loop range
        for i in range(len(data) - self.window - self.horizon):
            X.append(data[i:(i + self.window)])
            y.append(data[i + self.window : i + self.window + self.horizon])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"--- Training PatchTSTModel on {self.device} ---")
        
        # We only need the target series ('close' price) for this univariate model
        target_series_train = y_train.values.reshape(-1, 1)
        scaled_target = self.scaler.fit_transform(target_series_train)
        
        X_seq, y_seq = self._create_sequences(scaled_target)

        self.model = PyTorchPatchTST(
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

    def predict(self, X_test: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """
        Generates predictions using a rolling-window approach.
        """
        print("--- Predicting with PatchTSTModel (Rolling Window) ---")
        self.model.eval()

        # Combine historical training data and the test features for a continuous series
        # Note: For a pure univariate model, we only need y_train (the 'close' prices)
        historical_data = y_train.values
        
        # For the test period, we need the actual 'close' values to roll the window forward.
        # This is available in the original test dataframe, which we assume is passed implicitly
        # in the structure of this project (X_test has a 'close' column from the original data).
        # A more robust pipeline would pass df_test_original here. For now, we assume
        # X_test contains the unscaled 'close' price for this to work.
        # Let's reconstruct the test 'close' prices.
        
        # The scaler was fit on y_train. We need to create one long series to predict.
        # This is a simplification; a real pipeline would need access to the full original df.
        # We will assume y_train contains all historical close prices.
        
        scaled_history = self.scaler.transform(historical_data.reshape(-1, 1))
        
        predictions_scaled = []
        
        with torch.no_grad():
            # The first input sequence is the last 'window' of the training data
            current_sequence = list(scaled_history[-self.window:])
            
            for i in range(len(X_test)):
                input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                
                # Get the multi-step prediction
                prediction_horizon_scaled = self.model(input_tensor)
                
                # We only care about the first prediction for our t+1 strategy
                first_step_prediction = prediction_horizon_scaled[0, 0].item()
                predictions_scaled.append(first_step_prediction)
                
                # Roll the window: remove the oldest value and add the new predicted value
                # This is a common approach when the true value is not available.
                # A more accurate backtest would use the true observed value.
                # For now, we use the prediction to continue the sequence.
                current_sequence.pop(0)
                current_sequence.append([first_step_prediction])


        # Inverse transform the collected predictions
        predictions_scaled_np = np.array(predictions_scaled).reshape(-1, 1)
        final_predictions = self.scaler.inverse_transform(predictions_scaled_np)
        
        return final_predictions.flatten()

# --- PyTorch Model Definition ---
class PyTorchPatchTST(nn.Module):
    """
    A more robust implementation of the PatchTST model where internal
    layer sizes are determined dynamically during the forward pass.
    """
    def __init__(self, prediction_length, patch_len, stride, model_dim, num_heads, num_layers):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.model_dim = model_dim
        self.prediction_length = prediction_length
        
        self.embedding = nn.Linear(patch_len, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # The head layer will be created dynamically in the first forward pass
        self.head = None
        self.pos_encoder = None

    def forward(self, x):
        # x shape: (B, L, F=1)
        x_sq = x.squeeze(-1)  # Shape (B, L)
        
        # Create patches from the input sequence
        patches = x_sq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: (B, num_patches, patch_len)
        
        num_patches = patches.shape[1]
        
        # Dynamically create positional encoding if it doesn't exist or if num_patches has changed
        if self.pos_encoder is None or self.pos_encoder.shape[1] != num_patches:
            self.pos_encoder = nn.Parameter(torch.randn(1, num_patches, self.model_dim), requires_grad=True).to(x.device)
        
        # Embed patches and add positional encoding
        embedding = self.embedding(patches)
        embedding = embedding + self.pos_encoder
        
        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(embedding)
        transformer_output_flat = transformer_output.reshape(transformer_output.size(0), -1)
        
        # Dynamically create the head layer on the first pass
        if self.head is None:
            in_features = transformer_output_flat.shape[1]
            self.head = nn.Linear(in_features, self.prediction_length).to(x.device)
        
        # Generate the final prediction
        prediction = self.head(transformer_output_flat)
        return prediction