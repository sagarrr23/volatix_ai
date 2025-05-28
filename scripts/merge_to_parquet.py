#!/usr/bin/env python
"""
Merge all daily Binance 1-minute CSVs (downloaded via data-portal)
into a single clean Parquet + CSV.

Handles:
• corrupted / out-of-range timestamps
• duplicate rows
• automatic directory creation
"""

import os, glob, pandas as pd
from tqdm import tqdm

RAW_DIR  = "data_raw"                           # where daily CSVs live
OUT_DIR  = "parquet"                            # final storage
SYMBOL   = "BTCUSDT"                            # change to ETHUSDT, etc.
TF       = "1m"

os.makedirs(OUT_DIR, exist_ok=True)
OUT_PARQ = f"{OUT_DIR}/{SYMBOL.lower()}_{TF}_full.parquet"
OUT_CSV  = OUT_PARQ.replace(".parquet", ".csv")

def merge_csvs():
    pattern = os.path.join(RAW_DIR, f"{SYMBOL}-{TF}-*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print("❌ No daily CSVs found – run the downloader first.")
        return

    print(f"🔄 Merging {len(files)} daily CSV files …")
    frames = []
    for f in tqdm(files, desc="Loading"):
        df = pd.read_csv(
            f, header=None,
            names=["timestamp","open","high","low","close",
                   "volume","close_time","quote_vol",
                   "num_trades","taker_base","taker_quote","ignore"],
            usecols=["timestamp","open","high","low","close","volume"]
        )
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # ── robust timestamp handling ──────────────────────────
    full["timestamp"] = pd.to_datetime(
        full["timestamp"].astype("int64"),       # ensure int
        unit="ms", utc=True, errors="coerce"     # bad values → NaT
    )
    # drop NaT or absurd future dates
    full = full.dropna(subset=["timestamp"])
    full = full[full["timestamp"] < "2100-01-01"]

    # deduplicate & sort
    full = full.drop_duplicates("timestamp").sort_values("timestamp")
    full.reset_index(drop=True, inplace=True)

    # Save
    full.to_csv(OUT_CSV, index=False)
    full.to_parquet(OUT_PARQ)
    print(f"✅ Merged rows  : {len(full):,}")
    print(f"✅ CSV  saved  → {OUT_CSV}")
    print(f"✅ Parquet saved → {OUT_PARQ}")

if __name__ == "__main__":
    merge_csvs()
