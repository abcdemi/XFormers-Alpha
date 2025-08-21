"""
loaders.py
Data loaders for financial time series.

Supports:
- Yahoo Finance (via yfinance)
- Local CSV files

Includes schema validation and reproducible output.
"""

import os
import pandas as pd
import yfinance as yf
from typing import List, Optional

# ------------------
# Schema enforcement
# ------------------
EXPECTED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _check_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has required columns and sorted timestamps."""
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # enforce types
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ------------------
# CSV Loader
# ------------------
def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file and validate schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return _check_schema(df)


# ------------------
# Yahoo Finance Loader
# ------------------
def load_yfinance(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Args:
        ticker (str): Ticker symbol, e.g. 'AAPL' or 'SPY'.
        start (str, optional): Start date ("YYYY-MM-DD").
        end (str, optional): End date ("YYYY-MM-DD").
        interval (str): Data frequency, e.g. '1d', '1h', '1m'.

    Returns:
        pd.DataFrame with schema [timestamp, open, high, low, close, volume].
    """
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}.")

    df = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    ).reset_index()

    # Normalize timestamp column
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})

    # Keep only required columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return _check_schema(df)


# ------------------
# Multi-asset loader
# ------------------
def load_universe(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    source: str = "yfinance",
    data_dir: Optional[str] = None,
) -> dict:
    """
    Load multiple assets as dict of DataFrames keyed by ticker.

    Args:
        tickers (List[str]): List of ticker symbols.
        start, end (str): Date range.
        interval (str): Frequency.
        source (str): 'yfinance' or 'csv'.
        data_dir (str, optional): Required if source='csv'.

    Returns:
        dict[str, pd.DataFrame]
    """
    results = {}
    for t in tickers:
        if source == "yfinance":
            results[t] = load_yfinance(t, start, end, interval)
        elif source == "csv":
            if data_dir is None:
                raise ValueError("data_dir must be provided for CSV source.")
            path = os.path.join(data_dir, f"{t}.csv")
            results[t] = load_csv(path)
        else:
            raise ValueError(f"Unknown source: {source}")
    return results
