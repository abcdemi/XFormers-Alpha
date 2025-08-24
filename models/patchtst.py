import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings

# Suppress potential warnings from yfinance and set device
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Acquisition ---
print("Step 1: Downloading S&P 500 (^GSPC) data from Yahoo Finance...")
try:
    ticker = "^GSPC"
    # Download data (ensuring we handle potential empty results gracefully)
    data = yf.download(ticker, start="2010-01-01", end="2024-12-31", progress=False)
    if data.empty:
        raise ValueError("No data downloaded.")
    close_prices = data[['Close']]
    print("Data downloaded successfully.")
except Exception as e:
    print(f"Failed to download data: {e}")
    exit()

# --- 2. Data Preprocessing (CRITICAL FIX HERE) ---
print("\nStep 2: Preprocessing and reshaping the data...")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
# IMPORTANT: Use .values.reshape(-1, 1) to ensure the data is 2D (Samples, Features)
scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# Define a function to create input/output sequences
def create_sequences(data, context_length, prediction_length):
    """Creates sequences and corresponding labels for time series forecasting."""
    x, y = [], []
    # Data is already 2D (Time, 1), so we iterate over the time dimension (axis 0)
    for i in range(len(data) - context_length - prediction_length + 1):
        x.append(data[i:(i + context_length), :])
        y.append(data[(i + context_length):(i + context_length + prediction_length), :])
    return np.array(x), np.array(y)

# Model and data parameters
context_length = 96  # L: Number of past days to use for prediction
prediction_length = 32  # O: Number of future days to predict

# X shape: (N, L, 1), y shape: (N, O, 1)
X, y = create_sequences(scaled_data, context_length, prediction_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert data to PyTorch tensors and move to device
# Tensors retain the shape (N, L, 1) and (N, O, 1)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create PyTorch DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Input batch shape: {next(iter(train_loader))[0].shape} (Batch, Seq Length, Features)")

# --- 3. Model Definition ---
print("\nStep 3: Defining the PatchTST model...")
class PatchTST(nn.Module):
    def __init__(self, context_length, prediction_length, patch_len, stride, model_dim, num_heads, num_layers):
        super(PatchTST, self).__init__()
        
        # We assume channel independence, so we operate only on the sequence dimension (dim 1)
        # Input shape per batch: (B, L, F=1)
        self.context_length = context_length
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (context_length - patch_len) // stride + 1

        # We need an additional layer to handle the single feature dimension (F=1)
        self.feature_extractor = nn.Linear(1, model_dim) 
        
        # Patching and embedding (linear layer operates on patch_len)
        self.embedding = nn.Linear(patch_len, model_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, model_dim))

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Prediction Head: Maps flattened Transformer output to prediction_length
        self.head = nn.Linear(model_dim * self.num_patches, prediction_length)

    def forward(self, x):
        # x shape: (B, L, F=1). 
        
        # Since PatchTST is Channel-Independent, we can squeeze the feature dim (F=1) 
        # or handle the patching across the sequence length (L).
        
        # For simplicity in this univariate case, we squeeze to (B, L) before patching:
        x_sq = x.squeeze(-1) # Shape (B, L)
        
        # Patching: dimension=1 (sequence length), size=patch_len, step=stride
        # Output shape of unfold: (B, num_patches, patch_len)
        patches = x_sq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # Now, patches has shape (B, N_patches, patch_len).
        # We need to treat N_patches as the sequence length for the Transformer.
        
        # Embedding: Maps patch_len -> model_dim
        # embedding shape: (B, N_patches, model_dim)
        embedding = self.embedding(patches)
        
        # Add positional encoding
        embedding = embedding + self.pos_encoder
        
        # Transformer Encoder
        transformer_output = self.transformer_encoder(embedding) 
        
        # Flatten the output: (B, N_patches * model_dim)
        transformer_output = transformer_output.reshape(transformer_output.size(0), -1)

        # Head for prediction
        # Output shape: (B, prediction_length)
        prediction = self.head(transformer_output)
        return prediction

# Model hyperparameters
model_dim = 128
num_heads = 8
num_layers = 3
patch_len = 16
stride = 8

model = PatchTST(context_length, prediction_length, patch_len, stride, model_dim, num_heads, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
print("Model defined successfully.")

# --- 4. Model Training ---
print("\nStep 4: Starting model training...")
num_epochs = 25
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # y_pred shape: (B, O)
        y_pred = model(X_batch)
        # Target y_batch shape: (B, O, 1). We squeeze the feature dim for loss comparison.
        loss = criterion(y_pred, y_batch.squeeze(-1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.squeeze(-1))
            epoch_test_loss += loss.item()
    test_losses.append(epoch_test_loss / len(test_loader))

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.5f}, Test Loss: {test_losses[-1]:.5f}')
print("Training complete.")

# --- 5. Prediction and Plotting ---
print("\nStep 5: Making predictions and plotting results...")
model.eval()
with torch.no_grad():
    # Input X_test_tensor (N_test, L, 1) -> Output predictions_scaled (N_test, O)
    predictions_scaled = model(X_test_tensor).cpu().numpy()

# Inverse scale the predictions and actuals
# Predictions are (N, O), so inverse_transform is simple.
predictions = scaler.inverse_transform(predictions_scaled)
# Actuals are (N, O, 1). Must squeeze/reshape to (N*O, 1) for inverse_transform.
actuals = scaler.inverse_transform(y_test_tensor.squeeze(-1).cpu().numpy())


# Plotting only the 1-day ahead prediction (index 0 of the prediction window)
plot_start_index = 100
plot_length = 200
plot_end_index = plot_start_index + plot_length

# Get the corresponding dates from the original dataframe index
# The test set starts at index train_size * (Context + Prediction) in the original sequence list.
# The dates corresponding to the *start* of the forecast (t+1) are needed.
test_dates = close_prices.index[train_size + context_length : train_size + context_length + len(actuals)]
date_index = test_dates[plot_start_index:plot_end_index]


plt.figure(figsize=(15, 7))
plt.plot(date_index, actuals[plot_start_index:plot_end_index, 0], label='Actual Prices', color='blue', marker='.')
plt.plot(date_index, predictions[plot_start_index:plot_end_index, 0], label='Predicted Prices (1-day ahead)', color='red', linestyle='--', marker='.')

plt.title('S&P 500 Price Prediction using PatchTST (1-Day Ahead Forecast)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nExperiment finished. The plot shows the model's 1-day ahead predictions against the actual prices.")