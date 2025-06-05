"""
prepare_tft_data.py — Volatix AI v3.7

Multi-timeframe feature engineering + intelligent 5-class labeling for TFT.
Labels: 0 = Strong Sell, 1 = Weak Sell, 2 = Wait, 3 = Weak Buy, 4 = Strong Buy
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import RandomOverSampler

# ───── CONFIG ──────────────────────────────
DATA_FILES = {
    "1m": "historical_data/1m.csv",
    "5m": "historical_data/5m.csv",
    "15m": "historical_data/15m.csv",
    "1h": "historical_data/1h.csv"
}
SAVE_PATH = "historical_data/processed_tft_data.npz"
SEQ_LEN = 30
FUTURE_STEPS = [1, 3, 6]
MIN_MOVE_THRESH = 0.001
HIGH_CONF_THRESH = 0.002
MAX_HOLD = 12

# ───── LOAD + ENGINEER ─────────────────────
def load_and_engineer(file, suffix):
    df = pd.read_csv(file, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]

    df[f"log_return_{suffix}"] = np.log(df["close"] / df["close"].shift(1))
    df[f"volatility_{suffix}"] = df[f"log_return_{suffix}"].rolling(SEQ_LEN).std()
    df[f"vwap_{suffix}"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    df[f"price_vs_vwap_{suffix}"] = df["close"] / df[f"vwap_{suffix}"] - 1
    df[f"volume_z_{suffix}"] = (df["volume"] - df["volume"].rolling(SEQ_LEN).mean()) / (df["volume"].rolling(SEQ_LEN).std() + 1e-9)

    return df.drop(["open", "high", "low", "close", "volume"], axis=1)

# ───── MERGE TIMEFRAMES ─────────────────────
def merge_timeframes(data_files):
    base = load_and_engineer(data_files["1m"], "1m")
    for tf, path in data_files.items():
        if tf == "1m": continue
        df = load_and_engineer(path, tf)
        base = base.join(df, how="inner")
    return base.dropna()

# ───── SMART LABELING LOGIC ────────────────
def generate_smart_labels(close_series):
    y_dir, y_conf, y_rew = [], [], []
    closes = close_series.values

    for i in range(len(closes) - MAX_HOLD):
        window = closes[i:i + MAX_HOLD + 1]
        current = window[0]
        forward = window[1:]
        pct_moves = (forward - current) / current
        reward = np.max(np.abs(pct_moves))
        confidence = np.max(pct_moves) if np.max(pct_moves) > np.abs(np.min(pct_moves)) else np.min(pct_moves)

        if confidence >= 2 * HIGH_CONF_THRESH:
            direction = 4  # Strong Buy
        elif confidence >= MIN_MOVE_THRESH:
            direction = 3  # Weak Buy
        elif confidence <= -2 * HIGH_CONF_THRESH:
            direction = 0  # Strong Sell
        elif confidence <= -MIN_MOVE_THRESH:
            direction = 1  # Weak Sell
        else:
            direction = 2  # Wait

        y_dir.append(direction)
        y_conf.append(abs(confidence))
        y_rew.append(reward)

    return np.array(y_dir), np.array(y_conf), np.array(y_rew)

# ───── SEQUENCE MAKER ─────────────────────
def create_sequences(df, y_dir, y_conf, y_rew):
    X, y_d, y_c, y_r = [], [], [], []
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    for i in range(SEQ_LEN, len(df_scaled) - MAX_HOLD):
        X.append(df_scaled.iloc[i - SEQ_LEN:i].values)
        y_d.append(y_dir[i])
        y_c.append(y_conf[i])
        y_r.append(y_rew[i])

    return np.array(X), np.array(y_d), np.array(y_c), np.array(y_r)

# ───── BALANCE SAMPLES ─────────────────────
def balance_classes(X, y_dir, y_conf, y_rew):
    original_shape = X.shape
    flat_X = X.reshape((X.shape[0], -1))
    ros = RandomOverSampler()
    flat_X_res, y_dir_res = ros.fit_resample(flat_X, y_dir)
    y_conf_res = np.take(y_conf, ros.sample_indices_)
    y_rew_res = np.take(y_rew, ros.sample_indices_)
    X_res = flat_X_res.reshape((-1, original_shape[1], original_shape[2]))
    return X_res, y_dir_res, y_conf_res, y_rew_res

# ───── MAIN ─────────────────────────────────
def main():
    print("🔄 Preparing smart data for TFT Brain...")
    df = merge_timeframes(DATA_FILES)
    print(f"📦 Feature shape: {df.shape}")

    y_dir, y_conf, y_rew = generate_smart_labels(df["log_return_1m"])
    X, y_dir, y_conf, y_rew = create_sequences(df, y_dir, y_conf, y_rew)

    # Balance samples for better learning
    X, y_dir, y_conf, y_rew = balance_classes(X, y_dir, y_conf, y_rew)

    label_dist = dict(Counter(y_dir))
    print(f"📊 Label Distribution (5-class): {label_dist}")
    print(f"✅ Final Shapes: X={X.shape}, y_dir={y_dir.shape}, y_conf={y_conf.shape}, y_rew={y_rew.shape}")

    np.savez_compressed(SAVE_PATH, X=X, y_direction=y_dir, y_confidence=y_conf, y_reward=y_rew)
    print(f"💾 Saved to: {SAVE_PATH}")

# ───── ENTRY ───────────────────────────────
if __name__ == "__main__":
    main()
