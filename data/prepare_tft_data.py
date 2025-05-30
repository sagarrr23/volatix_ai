import os
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# ───── CONFIG ──────────────────────────────
SEQ_LEN       = 20         # Time steps for memory
FUTURE_STEP   = 3          # Predict 3 steps ahead (15min)
THRESHOLD     = 0.0015     # 0.15% threshold
DATA_DIR      = "historical_data"
SAVE_PREFIX   = DATA_DIR
# ───────────────────────────────────────────

def load_tf(file, suffix):
    df = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [f"{c}_{suffix}" for c in df.columns]
    return df.sort_index()

def merge_timeframes():
    df_1m  = load_tf(f"{DATA_DIR}/1m.csv",  "1m")
    df_5m  = load_tf(f"{DATA_DIR}/5m.csv",  "5m")
    df_15m = load_tf(f"{DATA_DIR}/15m.csv", "15m")
    df_1h  = load_tf(f"{DATA_DIR}/1h.csv",  "1h")
    return df_5m.join([df_1m, df_15m, df_1h], how="inner")

def add_indicators(df, suffix):
    c, h, l, v = f"close_{suffix}", f"high_{suffix}", f"low_{suffix}", f"volume_{suffix}"
    df[f"rsi_{suffix}"]        = ta.momentum.RSIIndicator(df[c], window=14).rsi()
    df[f"atr_{suffix}"]        = ta.volatility.AverageTrueRange(df[h], df[l], df[c]).average_true_range()
    bb                         = ta.volatility.BollingerBands(close=df[c])
    df[f"bbw_{suffix}"]        = bb.bollinger_hband() - bb.bollinger_lband()
    df[f"macd_diff_{suffix}"]  = ta.trend.MACD(close=df[c]).macd_diff()
    df[f"ema_fast_{suffix}"]   = ta.trend.EMAIndicator(close=df[c], window=9).ema_indicator()
    df[f"ema_slow_{suffix}"]   = ta.trend.EMAIndicator(close=df[c], window=21).ema_indicator()
    df[f"vol_log_{suffix}"]    = np.log1p(df[v])
    df[f"ret_{suffix}"]        = df[c].pct_change()
    df[f"volatility_{suffix}"] = df[f"ret_{suffix}"].rolling(10).std()
    df[f"trend_flag_{suffix}"] = (df[c] > df[f"ema_slow_{suffix}"]).astype(int)
    return df

def generate_labels(df):
    df["future_close"] = df["close_5m"].shift(-FUTURE_STEP)
    df["return_5m"] = (df["future_close"] - df["close_5m"]) / df["close_5m"]
    df = df.dropna(subset=["future_close"])
    df["direction"] = pd.cut(df["return_5m"],
                             bins=[-np.inf, -THRESHOLD, THRESHOLD, np.inf],
                             labels=[0, 1, 2]).astype(int)
    df["confidence"] = (df["return_5m"].abs() / df["return_5m"].abs().max()).clip(0, 1)
    df["reward"]     = df["return_5m"].clip(-0.1, 0.1)
    return df

def balance_classes(df):
    groups = [df[df.direction == i] for i in range(3)]
    target = max(len(g) for g in groups)
    resampled = [resample(g, replace=True, n_samples=target, random_state=42) for g in groups]
    return pd.concat(resampled).sort_index()

def scale_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def make_sequences(df, feature_cols):
    X, y_dir, y_conf, y_rwd = [], [], [], []
    for i in range(SEQ_LEN, len(df)):
        X.append(df[feature_cols].iloc[i - SEQ_LEN:i].values)
        y_dir.append(df["direction"].iloc[i])
        y_conf.append(df["confidence"].iloc[i])
        y_rwd.append(df["reward"].iloc[i])
    return np.array(X), np.array(y_dir), np.array(y_conf), np.array(y_rwd)

def main():
    print("▶️ Merging timeframes...")
    df = merge_timeframes()

    for tf in ["1m", "5m", "15m", "1h"]:
        print(f"⚙️ Adding indicators for {tf}...")
        df = add_indicators(df, tf)

    print("🏷️ Creating labels...")
    df = generate_labels(df)

    print("🧮 Balancing classes...")
    df = balance_classes(df).dropna()

    print("📏 Scaling features...")
    label_cols = ["future_close", "return_5m", "direction", "confidence", "reward"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    df = scale_features(df, feature_cols)

    print("🧠 Creating training sequences...")
    X, y_dir, y_conf, y_rwd = make_sequences(df, feature_cols)

    print(f"✅ Shapes: X={X.shape}, y_dir={y_dir.shape}, y_conf={y_conf.shape}, y_rwd={y_rwd.shape}")
    os.makedirs(SAVE_PREFIX, exist_ok=True)
    np.save(f"{SAVE_PREFIX}/X.npy", X)
    np.save(f"{SAVE_PREFIX}/y_direction.npy", y_dir)
    np.save(f"{SAVE_PREFIX}/y_confidence.npy", y_conf)
    np.save(f"{SAVE_PREFIX}/y_reward.npy", y_rwd)
    print("💾 Saved successfully. Brain ready for training!")

if __name__ == "__main__":
    main()
