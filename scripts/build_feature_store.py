import duckdb, pandas as pd, numpy as np, ta, os, pyarrow.parquet as pq

# ── Configurations ────────────────────────────────
PARQ_5M   = "parquet/btcusdt_5m.parquet"
DB_FILE   = "feature_store/btc_features.duckdb"
TABLE     = "btc_features_5m"
FUTURE_HORIZON = 3         # How many candles to look ahead for reward
REWARD_SCALE   = 100       # To express reward as %
VOL_THRESHOLD  = 0.002     # Volatility threshold for confidence label

os.makedirs("feature_store", exist_ok=True)

# ── Step 1: Load Parquet into DataFrame ────────────
print("📥 Loading 5-minute Parquet …")
df = pq.read_table(PARQ_5M).to_pandas()

# ── Step 2: Feature Engineering ────────────────────
print("⚙️  Engineering features …")
df["rsi_14"]    = ta.momentum.RSIIndicator(df["close"], 14).rsi()
df["atr_14"]    = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
bb              = ta.volatility.BollingerBands(df["close"], 20, 2)
df["bb_width"]  = bb.bollinger_hband() - bb.bollinger_lband()
df["macd_diff"] = ta.trend.MACD(df["close"]).macd_diff()
df["ema_9"]     = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
df["ema_21"]    = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
df["log_vol"]   = (df["volume"] + 1).apply(np.log)
df["pct_change"]= df["close"].pct_change()
df["volatility"]= df["pct_change"].rolling(30).std()

# ── Step 3: Label Generation ───────────────────────
print("🎯 Generating target labels …")
df["future_close"]  = df["close"].shift(-FUTURE_HORIZON)
df["y_reward"]      = (df["future_close"] - df["close"]) / df["close"] * REWARD_SCALE
df["y_direction"]   = df["y_reward"].apply(lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0))
df["y_confidence"]  = df["volatility"].apply(lambda v: 1.0 if v > VOL_THRESHOLD else 0.5)

# ── Step 4: Clean & Organize ───────────────────────
df = df.dropna().reset_index(drop=True)

# Optional: Reorder columns to place labels last
label_cols = ["y_reward", "y_direction", "y_confidence"]
feature_cols = [col for col in df.columns if col not in label_cols and col != "future_close"]
df = df[feature_cols + label_cols]

# ── Step 5: Write to DuckDB ────────────────────────
con = duckdb.connect(DB_FILE)
con.execute(f"DROP TABLE IF EXISTS {TABLE}")
con.register("tmp_df", df)
con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM tmp_df")
con.close()

# ── Final Summary ──────────────────────────────────
print(f"✅ Feature table '{TABLE}' saved inside {DB_FILE}")
print(f"🧾 Total rows: {len(df):,} | Total columns: {df.shape[1]}")
