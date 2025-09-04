!pip install lightning pytorch-forecasting yfinance -q
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import matplotlib.pyplot as plt
import yfinance as yf

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
# --- START OF FIX ---
# Import the correct loss function for quantile forecasting
from pytorch_forecasting.metrics import QuantileLoss
# --- END OF FIX ---

# Set a seed for reproducibility
pl.seed_everything(42)

# 1. Data Acquisition and Preparation

# Download Apple Inc. (AAPL) stock data from Yahoo Finance
data = yf.download("AAPL", start="2018-01-01", end="2023-12-31")

# Flatten the MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Create a group identifier and a continuous time index
data["ticker"] = "AAPL"
data["time_idx"] = (data.index - data.index.min()).days

# Define training parameters
max_prediction_length = 30
max_encoder_length = 90
training_cutoff = data["time_idx"].max() - max_prediction_length

# Create the TimeSeriesDataSet for training
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    group_ids=["ticker"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["ticker"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["Volume", "Open", "High", "Low"],
    target_normalizer=GroupNormalizer(
        groups=["ticker"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)

# Create a validation set from the training data
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# Create dataloaders for handling batching of the data
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# 2. Model Training

# Configure the PyTorch Lightning trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="cpu",
    devices=1,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)

# Initialize the Temporal Fusion Transformer model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    # --- START OF FIX ---
    # Use the QuantileLoss, which is designed for probabilistic forecasting.
    loss=QuantileLoss(),
    # --- END OF FIX ---
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit the model to the training data
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 3. Prediction and Visualization

# Load the best performing model from training
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Generate predictions on the validation data
raw_predictions = best_tft.predict(val_dataloader, return_index=True)

# Plot the predictions against the actual values
best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
plt.title("Temporal Fusion Transformer Predictions vs. Actuals for AAPL")
plt.xlabel("Time Index")
plt.ylabel("Stock Price")
plt.show()

# Visualize the interpretation of the model's predictions
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
best_tft.plot_interpretation(interpretation)
plt.show()