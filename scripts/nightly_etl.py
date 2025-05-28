"""
Nightly ETL pipeline for Volatix AI
----------------------------------
1. Download yesterday's Binance ZIP for each timeframe we train on.
2. Append (dedup/sort) to master Parquet per timeframe.
3. Re-build DuckDB feature table (5-minute grid).
"""

import os, io, zipfile, requests
import pandas as pd, numpy as np, duckdb
import pyarrow.parquet as pq
from datetime import datetime, timedelta, timezone

# ───────── CONFIG ────────────────────────────────────────────────
SYM       = "BTCUSDT"
TFS       = ["1m", "5m", "15m", "1h"]          # TFs we keep
RAW_DIR   = "data_raw"
PARQ_DIR  = "parquet"
DB_FILE   = "feature_store/btc_features.duckdb"
TABLE_5M  = "btc_features_5m"
START_COL = ["timestamp", "open", "high", "low", "close", "volume"]
BASE_URL  = ("https://data.binance.vision/data/spot/daily/klines/"
             "{sym}/{tf}/{sym}-{tf}-{date}.zip")
# ────────────────────────────────────────────────────────────────

os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PARQ_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)           # for BAT redirection

def fetch_zip(tf: str, day: str):
    url = BASE_URL.format(sym=SYM, tf=tf, date=day)
    print("Downloading:", url)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"No file for {tf} {day} (status {r.status_code})")
        return None
    z  = zipfile.ZipFile(io.BytesIO(r.content))
    csv = z.namelist()[0]
    df  = pd.read_csv(z.open(csv), header=None,
           names=["timestamp","open","high","low","close",
                  "volume","close_time","quote_vol",
                  "num_trades","taker_base","taker_quote","ignore"])[START_COL]
    # robust timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"),
                                     unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] < "2100-01-01"]
    return df

def append_parquet(tf: str, new_df: pd.DataFrame):
    parq = f"{PARQ_DIR}/btcusdt_{tf}_full.parquet"
    if os.path.exists(parq):
        master = pq.read_table(parq).to_pandas()
        combined = pd.concat([master, new_df])
    else:
        combined = new_df
    combined = (combined.drop_duplicates("timestamp")
                        .sort_values("timestamp")
                        .reset_index(drop=True))
    combined.to_parquet(parq)
    print(f"Updated {tf} → {len(combined):,} rows")

def resample_from_1m():
    src = f"{PARQ_DIR}/btcusdt_1m_full.parquet"
    df1 = pq.read_table(src).to_pandas().set_index("timestamp")
    df1.index = pd.to_datetime(df1.index, utc=True)
    rule_map = {"5m":"5min", "15m":"15min", "1h":"1h"}
    for label, rule in rule_map.items():
        df = df1.resample(rule).agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"}).dropna()
        df.to_parquet(f"{PARQ_DIR}/btcusdt_{label}_full.parquet")
        print(f"Resampled {label} rows:", len(df))

def rebuild_feature_store():
    df = pq.read_table(f"{PARQ_DIR}/btcusdt_5m_full.parquet").to_pandas()
    import ta
    df["rsi_14"]   = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["atr_14"]   = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    bb             = ta.volatility.BollingerBands(df["close"], 20, 2)
    df["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()
    df["macd_diff"]= ta.trend.MACD(df["close"]).macd_diff()
    df = df.dropna().reset_index(drop=True)

    con = duckdb.connect(DB_FILE)
    con.execute(f"DROP TABLE IF EXISTS {TABLE_5M}")
    con.register("tmp_df", df)
    con.execute(f"CREATE TABLE {TABLE_5M} AS SELECT * FROM tmp_df")
    con.close()
    print("Feature table refreshed:", TABLE_5M)

def main():
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    # 1. Download & append per TF
    for tf in TFS:
        df = fetch_zip(tf, yesterday)
        if df is not None and not df.empty:
            append_parquet(tf, df)
    # 2. If 1-minute updated, regenerate the aggregates
    resample_from_1m()
    # 3. Rebuild feature store
    rebuild_feature_store()
    print("Nightly ETL complete.")

if __name__ == "__main__":
    main()
