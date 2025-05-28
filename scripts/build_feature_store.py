import duckdb, pandas as pd, numpy as np, ta, os, pyarrow.parquet as pq


PARQ_5M   = "parquet/btcusdt_5m.parquet"
DB_FILE   = "feature_store/btc_features.duckdb"
TABLE     = "btc_features_5m"
os.makedirs("feature_store", exist_ok=True)

# ── load 5-minute parquet into pandas (arrow back-end is fast) ──
print("📥 Loading 5-minute Parquet …")
df = pq.read_table(PARQ_5M).to_pandas()

# ── compute indicators (examples: add more as desired) ───────────
print("⚙️  Engineering features …")
df["rsi_14"]   = ta.momentum.RSIIndicator(df["close"], 14).rsi()
df["atr_14"]   = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
bb             = ta.volatility.BollingerBands(df["close"], 20, 2)
df["bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()
df["macd_diff"]= ta.trend.MACD(df["close"]).macd_diff()
df["ema_9"]    = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
df["ema_21"]   = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
df["log_vol"]  = (df["volume"]+1).apply(np.log)
df["pct_change"]= df["close"].pct_change()
df["volatility"]= df["pct_change"].rolling(30).std()
# … (add dozens more as needed) …

# drop rows with NaN from rolling calcs
df = df.dropna().reset_index(drop=True)

# ── write to DuckDB ───────────────────────────────────────────────
con = duckdb.connect(DB_FILE)
con.execute(f"DROP TABLE IF EXISTS {TABLE}")
con.register("tmp_df", df)
con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM tmp_df")
con.close()

print(f"✅ Feature table '{TABLE}' saved inside {DB_FILE}")
print(f"📝 Total rows: {len(df):,} | Total columns: {df.shape[1]}")
