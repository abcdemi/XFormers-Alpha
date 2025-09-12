# train.py
"""
Main training script.
This script orchestrates the data loading, model selection, and training process
based on a provided YAML configuration file.
"""
import yaml
import argparse
import pandas as pd
import pickle
import random
import numpy as np
import torch

from data.dataloader import load_and_prepare_data
from models.baselines import LinearBaseline, TreeBaseline, LSTMBaseline
from models.informer import InformerModel
from models.patchtst import PatchTSTModel
from models.tft_wrapper import TFTModel

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # --- 1. Argument Parsing and Config Loading ---
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("--- Configuration Loaded ---")
    print(config)

    # --- 2. Set Seed for Reproducibility ---
    set_seed(config['seed'])

    # --- 3. Data Loading and Preparation ---
    data_dict = load_and_prepare_data(config)
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    
    # --- 4. Model Selection ---
    model_name = config['model']['name'].lower()
    print(f"\n--- Selecting Model: {model_name} ---")

    models = {
        'linear': LinearBaseline,
        'tree': TreeBaseline,
        'lstm': LSTMBaseline,
        'informer': InformerModel,
        'patchtst': PatchTSTModel,
        'tft': TFTModel
    }

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(models.keys())}")
    
    model = models[model_name](config)
    
    # --- 5. Model Training ---
    model.train(X_train, y_train)

    # --- 6. Save Artifacts for Evaluation ---
    # We save the trained model and the test data needed for prediction/backtesting
    output_path = f"results/{model_name}_artifacts.pkl"
    print(f"\n--- Saving artifacts to {output_path} ---")
    
    artifacts = {
        'model': model,
        'data_dict': data_dict,
        'config': config
    }
    
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
        
    with open(output_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print("\n--- Training complete. Artifacts saved. ---")

if __name__ == '__main__':
    main()