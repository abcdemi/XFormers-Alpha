import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import matplotlib.pyplot as plt

# @title Stock Market Prediction with Informer Model

# @markdown ### Enter Stock Ticker and Date Range
stock_ticker = "GOOGL"  # @param {type:"string"}
start_date = "2020-01-01"  # @param {type:"date"}
end_date = "2025-01-01"  # @param {type:"date"}

# @markdown ### Model and Training Parameters
prediction_length = 20 # @param {type:"integer"}
context_length = 60  # @param {type:"integer"}
num_epochs = 10 # @param {type:"integer"}
batch_size = 32 # @param {type:"integer"}

# Download data
data = yf.download(stock_ticker, start=start_date, end=end_date)
prices = data['Close'].values

# Normalize data
min_price = np.min(prices)
max_price = np.max(prices)
scaled_prices = (prices - min_price) / (max_price - min_price)

# Create sequences
def create_sequences(data, context_length, prediction_length):
    sequences = []
    labels = []
    for i in range(len(data) - context_length - prediction_length + 1):
        sequences.append(data[i:i+context_length])
        labels.append(data[i+context_length:i+context_length+prediction_length])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(scaled_prices, context_length, prediction_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create PyTorch DataLoaders
train_dataset = TensorDataset(torch.from_numpy(X_train).float().unsqueeze(-1), torch.from_numpy(y_train).float().unsqueeze(-1))
test_dataset = TensorDataset(torch.from_numpy(X_test).float().unsqueeze(-1), torch.from_numpy(y_test).float().unsqueeze(-1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Configure and instantiate the model
config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=1,
    num_time_features=0,
    num_static_categorical_features=0,
    feature_size=10,
    encoder_layers=2,
    decoder_layers=2,
    d_model=32,
)
model = TimeSeriesTransformerForPrediction(config)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        past_values = batch[0]
        future_values = batch[1]
        outputs = model(past_values=past_values, future_values=future_values)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Generate predictions
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        past_values = batch[0]
        outputs = model.generate(past_values=past_values)
        predictions.extend(outputs.sequences.numpy())

# Inverse scale predictions and actuals
predicted_prices_scaled = np.array([p[0] for p in predictions])
predicted_prices = predicted_prices_scaled * (max_price - min_price) + min_price
actual_prices_scaled = y_test[:, 0]
actual_prices = actual_prices_scaled * (max_price - min_price) + min_price
test_dates = data.index[train_size + context_length: train_size + context_length + len(actual_prices)]


# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_dates, actual_prices, label='Actual Prices', color='blue')
plt.plot(test_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
plt.title(f'{stock_ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()