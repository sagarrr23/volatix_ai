import pandas as pd, os, pyarrow

SRC_PARQ = "parquet/btcusdt_1m_full.parquet"
DST_DIR  = "parquet"     # output folder

os.makedirs(DST_DIR, exist_ok=True)
df_1m = pd.read_parquet(SRC_PARQ).set_index("timestamp")
df_1m.index = pd.to_datetime(df_1m.index, utc=True)

def resample(rule, label):
    df_tf = df_1m.resample(rule).agg({
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum"
    }).dropna()

    out_parq = f"{DST_DIR}/btcusdt_{label}.parquet"
    out_csv  = out_parq.replace(".parquet", ".csv")
    df_tf.to_parquet(out_parq)
    df_tf.to_csv(out_csv)
    print(f"✅ {label} → {len(df_tf):,} rows  saved")

if __name__ == "__main__":
    resample("5min", "5m")
    resample("15min", "15m")
    resample("1h",   "1h")
