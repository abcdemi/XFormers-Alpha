# download_data.py
"""
A standalone script to download and cache financial data.
Run this once to get the data, then all other scripts will load from the local file.
"""
import yfinance as yf
import os

TICKER = "GOOGL"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = "local_data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{TICKER}.parquet")

def main():
    print(f"--- Data Caching Script ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print(f"Downloading data for {TICKER} from {START_DATE} to {END_DATE}...")
    try:
        data = yf.download(TICKER, start=START_DATE, end=END_DATE)
        if data.empty:
            raise ValueError("No data downloaded. Check ticker and date range.")
        
        # Save as Parquet for efficient storage and fast reading
        data.to_parquet(OUTPUT_PATH)
        print(f"Data downloaded and saved successfully to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()