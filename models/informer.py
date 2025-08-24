# === Minimal Informer for Stock Forecasting ===
# Requirements: pip install yfinance torch matplotlib pandas numpy

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1) Load stock data
# -----------------------------
ticker = "GOOGL"
df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
series = df["Close"].dropna().values

# Normalize
mean, std = series.mean(), series.std()
series_norm = (series - mean) / std

# -----------------------------
# 2) Create sliding windows
# -----------------------------
def make_windows(data, input_len=128, horizon=30):
    X, Y = [], []
    for i in range(len(data) - input_len - horizon):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + horizon])
    return np.array(X), np.array(Y)

input_len, horizon = 128, 30
X, Y = make_windows(series_norm, input_len, horizon)

split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
Y_test  = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1)

# -----------------------------
# 3) Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -----------------------------
# 4) Simplified Informer (Transformer Encoder-Decoder)
# -----------------------------
class Informer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=1, horizon=30):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True),
            num_layers=num_encoder_layers,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True),
            num_layers=num_decoder_layers,
        )

        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, input_len, 1]
        x = self.input_proj(x) * np.sqrt(self.d_model)
        x = self.pos_enc(x)
        memory = self.encoder(x)

        # decoder input = zeros (teacher forcing can be added)
        dec_in = torch.zeros(x.size(0), self.horizon, self.d_model, device=x.device)
        dec_in = self.pos_enc(dec_in)

        out = self.decoder(dec_in, memory)
        out = self.fc_out(out)
        return out  # [B, horizon, 1]

# -----------------------------
# 5) Train
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Informer(horizon=horizon).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 10
batch_size = 32

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = X_train[idx].to(device), Y_train[idx].to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    print(f"Epoch {epoch}, Train MSE: {total_loss / len(X_train):.6f}")

# -----------------------------
# 6) Forecast on test set
# -----------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy().squeeze(-1)

# take first test window for plotting
pred = preds[0] * std + mean
actual = Y_test[0].numpy().squeeze() * std + mean
history = X_test[0].numpy().squeeze() * std + mean

# -----------------------------
# 7) Plot
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(range(len(history)), history, label="History")
plt.plot(range(len(history), len(history) + len(actual)), actual, label="Actual")
plt.plot(range(len(history), len(history) + len(pred)), pred, label="Informer Forecast")
plt.title("GOOGL Stock Price Forecast with Informer (Scratch PyTorch)")
plt.legend()
plt.show()
