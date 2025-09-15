# train.py
import yaml
import argparse
import pandas as pd
import pickle
import random
import numpy as np
import torch
import optuna
from copy import deepcopy # Import the deepcopy function

from data.dataloader import load_and_prepare_data
from models.baselines import LinearBaseline, TreeBaseline, LSTMBaseline
from models.informer import InformerModel
from models.patchtst import PatchTSTModel
from models.tft_wrapper import TFTModel

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
    
    models = { 'linear': LinearBaseline, 'tree': TreeBaseline, 'lstm': LSTMBaseline, 'informer': InformerModel, 'patchtst': PatchTSTModel, 'tft': TFTModel }

    if config.get('tune', {}).get('enabled', False):
        print("\n--- Starting Hyperparameter Tuning with Optuna ---")
        
        def objective(trial):
            # --- START OF FIX ---
            # Use deepcopy to create a completely independent copy of the config for each trial
            trial_config = deepcopy(config)
            # --- END OF FIX ---
            
            # Suggest new hyperparameters for this trial
            for param, settings in trial_config['tune']['search_space'].items():
                param_type = settings.pop('type')
                
                target_config = None
                if param in trial_config['model']: target_config = trial_config['model']
                elif param in trial_config['train']: target_config = trial_config['train']
                else: raise ValueError(f"Parameter '{param}' not found in model or train config.")

                if param_type == 'int':
                    target_config[param] = trial.suggest_int(param, **settings)
                elif param_type == 'float':
                    target_config[param] = trial.suggest_float(param, **settings)
            
            val_split_idx = int(len(X_train) * 0.8)
            X_train_fold, X_val_fold = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
            y_train_fold, y_val_fold = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
            
            model_class = models[trial_config['model']['name'].lower()]
            model = model_class(trial_config)
            model.train(X_train_fold, y_train_fold)
            
            # Pass correct historical context for prediction
            predictions = model.predict(X_val_fold, X_train_fold)
            
            from evaluation.metrics import calculate_information_coefficient
            # Ensure predictions and actuals are aligned for IC calculation
            offset = len(y_val_fold) - len(predictions)
            aligned_y_val = y_val_fold.iloc[offset:]
            ic = calculate_information_coefficient(predictions, aligned_y_val.values)
            return ic

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config['tune']['n_trials'])
        
        print("\n--- Tuning Complete ---")
        print("Best IC on validation set:", study.best_value)
        print("Best parameters found:", study.best_params)
        
        # Update the main config with the best parameters
        for param, value in study.best_params.items():
             if param in config['model']: config['model'][param] = value
             if param in config['train']: config['train'][param] = value

    # --- Final Model Training ---
    model_name = config['model']['name'].lower()
    print(f"\n--- Training Final Model: {model_name} ---")
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