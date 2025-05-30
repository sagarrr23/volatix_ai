import os
import pandas as pd
import duckdb
from tqdm import tqdm

# Paths
CSV_FOLDER = "data_raw"  # e.g. "C:/NEWW/volatix_ai/raw_data/"
DUCKDB_FILE = "btc_features.duckdb"

# Column names (standard Binance 1m data)
columns = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]

# Connect to DuckDB
con = duckdb.connect(DUCKDB_FILE)

# Create table if not exists
con.execute(f"""
    CREATE TABLE IF NOT EXISTS btc_1m_raw (
        timestamp TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        close_time TIMESTAMP,
        quote_asset_volume DOUBLE,
        number_of_trades INTEGER,
        taker_buy_volume DOUBLE,
        taker_buy_quote_volume DOUBLE,
        ignore INTEGER
    )
""")

# Get list of CSVs
csv_files = sorted([f for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")])

# Load and insert each file
for file in tqdm(csv_files, desc="Processing CSVs"):
    try:
        df = pd.read_csv(os.path.join(CSV_FOLDER, file), header=None, names=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Optional: remove duplicates and NaNs
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.dropna(inplace=True)

        con.execute("INSERT INTO btc_1m_raw SELECT * FROM df")
    except Exception as e:
        print(f"Failed on {file}: {e}")

con.close()
print(f"\n✅ Feature store created at: {DUCKDB_FILE}")
