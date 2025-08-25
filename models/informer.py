# === Working Informer-style Transformer for GOOGL (Corrected for OMP Error) ===
# Requirements: pip install yfinance torch matplotlib numpy pandas

# This is a common environment issue, not a code bug.
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

# -----------------------------
# 1) Download data
# -----------------------------
ticker = "GOOGL"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
close = df["Close"].dropna().astype("float32").values.squeeze()

mean, std = close.mean(), close.std()
series = (close - mean) / (std + 1e-8)

# -----------------------------
# 2) Sliding windows
# -----------------------------
INPUT_LEN = 128
HORIZON = 30

def make_windows(arr, input_len=INPUT_LEN, horizon=HORIZON):
    X, Y = [], []
    for i in range(len(arr) - input_len - horizon):
        X.append(arr[i:i+input_len])
        Y.append(arr[i+input_len:i+input_len+horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X, Y = make_windows(series)
split = int(0.8*len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# -----------------------------
# 3) Convert to tensors with correct shapes
# X: [N, seq_len, 1], Y: [N, horizon, 1]
# -----------------------------
X_train = torch.from_numpy(X_train).float().unsqueeze(-1)
Y_train = torch.from_numpy(Y_train).float().unsqueeze(-1)
X_test  = torch.from_numpy(X_test).float().unsqueeze(-1)
Y_test  = torch.from_numpy(Y_test).float().unsqueeze(-1)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

# -----------------------------
# 4) DataLoader
# -----------------------------
BATCH_SIZE = 64
train_ds = TensorDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# 5) Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

# -----------------------------
# 6) Causal mask
# -----------------------------
def generate_square_subsequent_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

# -----------------------------
# 7) Simplified Informer
# -----------------------------
class Informer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1, horizon=HORIZON):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
        x = self.input_proj(x) * (self.d_model ** 0.5)
        x = self.pos_enc(x)
        memory = self.encoder(x)

        B = x.size(0)
        dec_in = torch.zeros(B, self.horizon, self.d_model, device=x.device)
        dec_in = self.pos_enc(dec_in)
        tgt_mask = generate_square_subsequent_mask(self.horizon, x.device)

        out = self.decoder(tgt=dec_in, memory=memory, tgt_mask=tgt_mask)
        out = self.fc_out(out)
        return out

# -----------------------------
# 8) Training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Informer().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 15
for epoch in range(1, EPOCHS+1):
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
    print(f"Epoch {epoch:02d} | Train MSE: {total_loss/len(train_dl.dataset):.6f}")

# -----------------------------
# 9) Forecast & plot
# -----------------------------
model.eval()
with torch.no_grad():
    test_sample_idx = 0
    test_input = X_test[test_sample_idx].unsqueeze(0)
    
    preds = model(test_input.to(device)).cpu().numpy().squeeze()
    
    actual = Y_test[test_sample_idx].numpy().squeeze()
    history = X_test[test_sample_idx].numpy().squeeze()

pred_unscaled = preds * std + mean
actual_unscaled = actual * std + mean
history_unscaled = history * std + mean

plt.figure(figsize=(12,6))
plt.plot(range(len(history_unscaled)), history_unscaled, label="History")
plt.plot(range(len(history_unscaled), len(history_unscaled)+len(actual_unscaled)), actual_unscaled, label="Actual")
plt.plot(range(len(history_unscaled), len(history_unscaled)+len(pred_unscaled)), pred_unscaled, label="Informer Forecast")
plt.title(f"{ticker} Stock Forecast — Simplified Informer")
plt.legend()
plt.grid(True)
plt.show()