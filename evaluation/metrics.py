# evaluation/metrics.py
"""
Functions for calculating financial performance metrics to evaluate a strategy.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def calculate_information_coefficient(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculates the Spearman Rank Information Coefficient (IC).
    This measures the rank correlation between predictions and actual outcomes.
    A value of 1.0 is a perfect ranking, -1.0 is a perfect inverse ranking, 0 is no correlation.

    Args:
        predictions (np.ndarray): The model's predictions.
        actuals (np.ndarray): The true target values.

    Returns:
        float: The Spearman IC.
    """
    ic, _ = spearmanr(predictions, actuals)
    return ic

def calculate_sharpe_ratio(returns: pd.Series, annualization_factor: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio of a returns series.
    Measures risk-adjusted return. Assumes a risk-free rate of 0.

    Args:
        returns (pd.Series): A series of periodic (e.g., daily) returns.
        annualization_factor (int): 252 for daily, 52 for weekly, etc.

    Returns:
        float: The annualized Sharpe Ratio.
    """
    if returns.std() == 0:
        return 0.0  # Avoid division by zero
    
    return (returns.mean() / returns.std()) * np.sqrt(annualization_factor)

def calculate_turnover(weights: pd.DataFrame) -> float:
    """
    Calculates the average daily turnover of the portfolio.
    Measures how much the portfolio's holdings change each day.

    Args:
        weights (pd.DataFrame): A DataFrame of portfolio weights over time.
                                Columns are assets, rows are dates.

    Returns:
        float: The average daily turnover.
    """
    # The change in weights from one day to the next, summed across all assets
    daily_turnover = weights.diff().abs().sum(axis=1)
    return daily_turnover.mean()

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the maximum drawdown of an equity curve.
    Measures the largest peak-to-trough decline (worst-case loss).

    Args:
        equity_curve (pd.Series): A series representing portfolio value over time.

    Returns:
        float: The maximum drawdown as a negative percentage.
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def summary_stats(returns: pd.Series, equity_curve: pd.Series, weights: pd.DataFrame) -> dict:
    """
    Generates a dictionary of summary performance statistics.
    """
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(equity_curve)
    turnover = calculate_turnover(weights)
    
    # Calculate a simple turnover-adjusted Sharpe
    # This is a basic form of penalization; more complex methods exist
    turnover_penalty = turnover * 0.5 # Example penalty factor
    adjusted_sharpe = sharpe - turnover_penalty
    
    return {
        "Total Return": (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        "Annualized Sharpe Ratio": sharpe,
        "Turnover-Adjusted Sharpe": adjusted_sharpe,

        "Max Drawdown": max_dd,
        "Average Daily Turnover": turnover,
    }