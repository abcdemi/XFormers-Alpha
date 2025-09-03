# === N-BEATS Experiment for GOOGL Stock Forecasting ===
# Requirements: pip install yfinance torch matplotlib numpy pandas

# --- 1. Setup and Environment Fix ---
# This addresses a common "OMP: Error #15" on Windows systems.
# It MUST be placed before importing numpy or torch.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# --- 2. Data Preparation ---
print("Step 2: Downloading and preparing data...")
ticker = "GOOGL"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)

# Squeeze to ensure it's a 1D array, critical for preventing dimension errors
close_prices = df["Close"].dropna().astype("float32").values.squeeze()

# Scale the data for better neural network performance
mean, std = close_prices.mean(), close_prices.std()
scaled_series = (close_prices - mean) / (std + 1e-8)

# --- 3. Create Sliding Windows ---
LOOKBACK_LEN = 90
HORIZON = 30

def make_windows(data, lookback_len, horizon):
    X, Y = [], []
    for i in range(len(data) - lookback_len - horizon + 1):
        X.append(data[i : i + lookback_len])
        Y.append(data[i + lookback_len : i + lookback_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X, Y = make_windows(scaled_series, LOOKBACK_LEN, HORIZON)

# Split data
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_test  = torch.from_numpy(X_test)
Y_test  = torch.from_numpy(Y_test)

# --- 4. The N-BEATS Model Definition ---
print("Step 4: Defining the N-BEATS model...")

class NBEATSBlock(nn.Module):
    """ A single block of the N-BEATS model. It is a simple feed-forward network """
    def __init__(self, in_features, out_features, hidden_units=256):
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(in_features, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, out_features)
        )

    def forward(self, x):
        return self.fc_stack(x)

class NBEATS(nn.Module):
    """ The main N-BEATS model, which is a stack of blocks with residual connections """
    def __init__(self, lookback_len, horizon, num_blocks=3):
        super().__init__()
        self.lookback_len = lookback_len
        self.horizon = horizon
        
        # Create a stack of blocks. Each block will produce a forecast and a backcast.
        self.stack = nn.ModuleList([
            NBEATSBlock(lookback_len, lookback_len + horizon) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x is the input history, with shape [batch_size, lookback_len]
        batch_size = x.size(0)
        
        # Initialize the total forecast and the current input for the first block
        total_forecast = torch.zeros(batch_size, self.horizon, device=x.device)
        backcast_residual = x
        
        for block in self.stack:
            # Get the block's output (both backcast and forecast)
            block_output = block(backcast_residual)
            
            # Split the output into the backcast and forecast parts
            block_backcast = block_output[:, :self.lookback_len]
            block_forecast = block_output[:, self.lookback_len:]
            
            # Update the total forecast by adding this block's forecast
            total_forecast += block_forecast
            
            # Update the residual for the next block by subtracting the backcast
            backcast_residual = backcast_residual - block_backcast
            
        return total_forecast

# --- 5. Training the Model ---
print("Step 5: Training the model...")
BATCH_SIZE = 128
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = TensorDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model = NBEATS(LOOKBACK_LEN, HORIZON).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_dl.dataset)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Train MSE: {avg_loss:.6f}")

# --- 6. Forecasting and Visualization ---
print("Step 6: Generating forecast and plotting results...")
model.eval()
with torch.no_grad():
    # We will predict on the first sample of the test set
    test_sample_idx = 0
    history = X_test[test_sample_idx].unsqueeze(0).to(device) # Add batch dim
    
    # Get the model's prediction
    forecast = model(history).cpu().numpy().squeeze()
    
    # Get the actual future values
    actual = Y_test[test_sample_idx].numpy().squeeze()
    
    # Un-scale the data to see the real stock prices
    history_unscaled = X_test[test_sample_idx].numpy().squeeze() * std + mean
    forecast_unscaled = forecast * std + mean
    actual_unscaled = actual * std + mean

# Plotting the results
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))

# Plot history
plt.plot(np.arange(LOOKBACK_LEN), history_unscaled, label="Historical Data", color='royalblue')

# Plot actual future
plt.plot(np.arange(LOOKBACK_LEN, LOOKBACK_LEN + HORIZON), actual_unscaled, label="Actual Future", color='mediumseagreen')

# Plot forecast
plt.plot(np.arange(LOOKBACK_LEN, LOOKBACK_LEN + HORIZON), forecast_unscaled, label="N-BEATS Forecast", color='tomato', linestyle='--')

plt.title(f"{ticker} Stock Price Forecast with N-BEATS", fontsize=16)
plt.xlabel("Days", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend(fontsize=11)
plt.show()