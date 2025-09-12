# evaluate.py
"""
Main evaluation script.
This script loads trained model artifacts, generates predictions, runs a backtest,
calculates performance metrics, and plots the results.
"""
import pickle
import argparse
import pandas as pd

from evaluation.backtester import run_backtest
from evaluation.metrics import summary_stats, calculate_information_coefficient
from evaluation.plots import plot_equity_curve

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate a trained forecasting model.")
    parser.add_argument('--artifacts', type=str, required=True, help='Path to the saved artifacts .pkl file.')
    args = parser.parse_args()
    
    # --- 2. Load Artifacts ---
    print(f"--- Loading artifacts from {args.artifacts} ---")
    with open(args.artifacts, 'rb') as f:
        artifacts = pickle.load(f)
        
    model = artifacts['model']
    data_dict = artifacts['data_dict']
    config = artifacts['config']

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    df_test_original = data_dict['df_test_original']
    
    # --- 3. Generate Predictions ---
    predictions = model.predict(X_test)
    
    # --- 4. Run Backtest ---
    backtest_results = run_backtest(predictions, df_test_original, config['backtest'])
    
    # --- 5. Calculate and Print Metrics ---
    print("\n--- Performance Metrics ---")
    
    # Align actuals for IC calculation
    offset = len(y_test) - len(predictions)
    aligned_y_test = y_test.iloc[offset:]
    
    ic = calculate_information_coefficient(predictions, aligned_y_test.values)
    print(f"Information Coefficient (IC): {ic:.4f}")
    
    stats = summary_stats(
        backtest_results['returns'], 
        backtest_results['equity_curve'], 
        backtest_results['weights']
    )
    
    for metric, value in stats.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
            
    # --- 6. Plot Results ---
    print("\n--- Generating Equity Curve Plot ---")
    model_name = config['model']['name']
    plot_equity_curve(backtest_results, model_name, df_test_original)

    print("\n--- Evaluation complete. ---")

if __name__ == '__main__':
    main()