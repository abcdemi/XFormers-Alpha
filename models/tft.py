# xformers_alpha/models/tft.py
"""
A robust, from-scratch implementation of the Temporal Fusion Transformer (TFT).

This file defines the self-contained TFTModel class, which encapsulates all
the logic for data preparation, training, and prediction for the TFT.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") # Suppress common pytorch-lightning warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

class TFTModel:
    """
    Encapsulates the entire TFT workflow: data preparation, training, and prediction.
    """
    def __init__(self, encoder_length=60, forecast_horizon=1, hidden_size=16, learning_rate=0.03):
        self.encoder_length = encoder_length
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.lr = learning_rate

    def _prepare_data_for_tft(self, df: pd.DataFrame):
        """
        Prepares a DataFrame for the TFT using a robust, step-by-step method.
        This is the most critical step for preventing library errors.
        """
        print("Preparing data for TFT...")
        # Start with a fresh copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # --- Add mandatory and time-based columns one by one ---
        # This explicit method is safer than using a dictionary constructor.
        data['time_idx'] = np.arange(len(data))
        data['group_id'] = "stock_0" # A single, consistent group ID is required.
        data['month'] = data['timestamp'].dt.month
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # --- Enforce strict data types for every column ---
        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = data[col].astype(np.float32)
            
        for col in ["group_id", "month", "day_of_week"]:
            data[col] = data[col].astype("category")
            
        return data

    def train_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Orchestrates the full training and evaluation pipeline.
        """
        print("--- Starting Temporal Fusion Transformer Workflow ---")

        # 1. Prepare the data into the special TFT format
        data = self._prepare_data_for_tft(df)

        # 2. Perform a chronological train/validation split
        split_idx = int(len(data) * (1 - test_size))
        train_data, val_data = data.iloc[:split_idx], data.iloc[split_idx:]
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples.")

        # 3. Create the TimeSeriesDataSet object
        # This is the core data object for the pytorch-forecasting library.
        dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="close", # The column we want to predict
            group_ids=["group_id"],
            max_encoder_length=self.encoder_length,
            max_prediction_length=self.forecast_horizon,
            
            # Define which columns are of which type
            static_categoricals=["group_id"],
            time_varying_known_categoricals=["month", "day_of_week"],
            # CRITICAL: 'close' is the target, so it is NOT listed here.
            # The library automatically uses the target as an input for the encoder.
            time_varying_unknown_reals=["open", "high", "low", "volume"],
        )
        
        # 4. Create validation set and dataloaders
        val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_data, predict=True, stop_randomization=True)
        train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

        # 5. Configure the PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=30, accelerator="auto", gradient_clip_val=0.1,
            limit_train_batches=30, # For quick runs; remove for full training
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
            logger=False, enable_checkpointing=True,
        )

        # 6. Configure and create the TFT model
        tft = TemporalFusionTransformer.from_dataset(
            dataset, learning_rate=self.lr, hidden_size=self.hidden_size,
            attention_head_size=1, dropout=0.1, hidden_continuous_size=8,
            output_size=7, # The model predicts 7 quantiles for the target
            loss=QuantileLoss(),
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        # 7. Train the model
        print("\nStarting model training...")
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print("Training finished.")

        # 8. Load the best model from the checkpoint and predict
        print("Loading best model and making predictions...")
        best_model = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        raw_preds, x = best_model.predict(val_loader, return_x=True)
        
        # Extract the median prediction (quantile 0.5, which is index 3 of 7)
        predictions = raw_preds[:, 0, 3].numpy()
        y_test = x["decoder_target"].squeeze().numpy()
        
        return predictions, y_test