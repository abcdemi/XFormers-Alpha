# evaluation/plots.py
"""
Functions for plotting backtest results.
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(results: dict, model_name: str, benchmark_df: pd.DataFrame = None):
    """
    Plots the equity curve of a strategy against a benchmark.

    Args:
        results (dict): The output dictionary from the backtester.
        model_name (str): The name of the model being plotted.
        benchmark_df (pd.DataFrame): The original test DataFrame to plot a 'Buy & Hold' benchmark.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    equity_curve = results['equity_curve']
    timestamps = results['timestamps']

    # Plot the strategy's equity curve
    ax.plot(timestamps, equity_curve, label=f'{model_name} Strategy', color='royalblue', linewidth=2)

    # Plot a 'Buy & Hold' benchmark
    if benchmark_df is not None:
        # Align the benchmark data with the strategy's timestamps
        benchmark_slice = benchmark_df[benchmark_df['timestamp'].isin(timestamps)]
        initial_price = benchmark_slice['close'].iloc[0]
        
        # Calculate the value of the benchmark portfolio
        benchmark_equity = 100_000 * (benchmark_slice['close'] / initial_price)
        
        ax.plot(timestamps, benchmark_equity, label='Buy & Hold', color='gray', linestyle='--', linewidth=2)

    ax.set_title(f'Strategy Performance: {model_name} vs. Buy & Hold', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    
    plt.show()