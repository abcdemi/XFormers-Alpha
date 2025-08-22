# models/tft.py
"""
Implementation of the Temporal Fusion Transformer (TFT) model.

This file uses the pytorch-forecasting library to streamline the data
preparation, training, and prediction process for the TFT model.
"""

import pandas as pd
import numpy as np

# PyTorch Lightning and Forecasting imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# ------------------
# Helper Function for TFT Data Preparation
# ------------------
def prepare_tft_data(df: pd.DataFrame, group_id: str = "GOOGL"):
    """
    Transforms a raw stock data DataFrame into the format required by
    the TimeSeriesDataSet for pytorch-forecasting.
    """
    df_new = df.copy()
    
    # Create the mandatory 'time_idx' column
    df_new['time_idx'] = np.arange(len(df_new))
    
    # Create a 'group' column (even if we only have one stock)
    df_new['group'] = group_id
    
    # Add time-based features that are "known" in the future
    df_new['month'] = df_new['timestamp'].dt.month.astype(str)
    df_new['day_of_week'] = df_new['timestamp'].dt.dayofweek.astype(str)
    
    return df_new

# ------------------
# The Main TFT Model Class
# ------------------
class TFTModel:
    def __init__(self, encoder_length=30, forecast_horizon=1):
        """
        Args:
            encoder_length (int): How many past time steps the model sees.
            forecast_horizon (int): How many time steps to predict into the future.
                                   For price prediction, this is typically 1.
        """
        self.encoder_length = encoder_length
        self.forecast_horizon = forecast_horizon

    def train_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2):
        print("--- Running Temporal Fusion Transformer (TFT) ---")
        
        # 1. Prepare data and split chronologically
        data = prepare_tft_data(df)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples.")

        # 2. Create the TimeSeriesDataSet object
        # This is the core data object for pytorch-forecasting
        training_cutoff = train_data["time_idx"].max()
        
        dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="close",
            group_ids=["group"],
            max_encoder_length=self.encoder_length,
            max_prediction_length=self.forecast_horizon,
            static_categoricals=["group"],
            time_varying_known_categoricals=["month", "day_of_week"],
            time_varying_unknown_reals=["open", "high", "low", "close", "volume"],
            target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        )

        # 3. Create validation set and dataloaders
        validation_dataset = TimeSeriesDataSet.from_dataset(dataset, val_data, predict=True, stop_randomization=True)
        train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = validation_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        # 4. Configure the trainer and model
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="auto", # Uses GPU if available
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[early_stop_callback],
        )

        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7, # Number of quantiles to predict
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        # 5. Train the model
        trainer.fit(
            tft,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # 6. Load the best model and make predictions
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        raw_predictions, x = best_tft.predict(val_loader, return_x=True)
        
        # We take the median prediction (index 3 of 7 quantiles)
        predictions = raw_predictions[:, 0, 3] 
        
        # Get ground truth values
        y_test = x["decoder_target"].squeeze().numpy()

        return predictions.numpy(), y_test