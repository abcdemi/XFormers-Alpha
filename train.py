# train.py
import yaml
import argparse
import pandas as pd
import pickle
import random
import numpy as np
import torch
import optuna

from data.dataloader import load_and_prepare_data
from models.baselines import LinearBaseline, TreeBaseline, LSTMBaseline
from models.informer import InformerModel
from models.patchtst import PatchTSTModel
from models.tft_wrapper import TFTModel

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("--- Configuration Loaded ---"); print(config)
    set_seed(config['seed'])
    
    data_dict = load_and_prepare_data(config)
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    
    # --- Check for Hyperparameter Tuning ---
    if config.get('tune', {}).get('enabled', False):
        print("\n--- Starting Hyperparameter Tuning with Optuna ---")
        
        def objective(trial):
            # Suggest new hyperparameters for this trial
            tune_config = config['tune']['search_space']
            config['model']['hidden_size'] = trial.suggest_int('hidden_size', **tune_config['hidden_size'])
            config['model']['depth'] = trial.suggest_int('depth', **tune_config['depth'])
            config['train']['lr'] = trial.suggest_float('lr', **tune_config['lr'])
            
            # Use a validation set for tuning (e.g., last 20% of training data)
            val_split_idx = int(len(X_train) * 0.8)
            X_train_fold, X_val_fold = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
            y_train_fold, y_val_fold = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
            
            model_class = models[config['model']['name'].lower()]
            model = model_class(config)
            model.train(X_train_fold, y_train_fold)
            
            # Evaluate on the validation fold
            predictions = model.predict(X_val_fold, X_train_fold)
            # We will optimize for IC
            from evaluation.metrics import calculate_information_coefficient
            ic = calculate_information_coefficient(predictions, y_val_fold.values)
            return ic

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config['tune']['n_trials'])
        
        print("\n--- Tuning Complete ---")
        print("Best IC on validation set:", study.best_value)
        print("Best parameters found:", study.best_params)
        
        # Update the config with the best parameters for the final training run
        config['model']['hidden_size'] = study.best_params['hidden_size']
        config['model']['depth'] = study.best_params['depth']
        config['train']['lr'] = study.best_params['lr']

    # --- Final Model Training ---
    model_name = config['model']['name'].lower()
    print(f"\n--- Training Final Model: {model_name} ---")
    models = { 'linear': LinearBaseline, 'tree': TreeBaseline, 'lstm': LSTMBaseline, 'informer': InformerModel, 'patchtst': PatchTSTModel, 'tft': TFTModel }
    model = models[model_name](config)
    model.train(X_train, y_train)

    # --- Save Artifacts ---
    output_path = f"results/{model_name}_tuned_artifacts.pkl"
    print(f"\n--- Saving artifacts to {output_path} ---")
    artifacts = {'model': model, 'data_dict': data_dict, 'config': config}
    import os
    if not os.path.exists('results'): os.makedirs('results')
    with open(output_path, 'wb') as f: pickle.dump(artifacts, f)
    print("\n--- Training complete. Artifacts saved. ---")

if __name__ == '__main__':
    main()