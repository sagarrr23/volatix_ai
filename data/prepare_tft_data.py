import os
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# ───── CONFIG ─────────────────────────────────────────────────────────
SEQ_LEN       = 20       # history window
FUTURE_STEP   = 3        # 5m steps ahead
THRESHOLD     = 0.0015   # 0.15%
DATA_DIR      = "historical_data"
SAVE_PREFIX   = DATA_DIR
# ────────────────────────────────────────────────────────────────────────

def load_tf(file, suffix):
    df = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    df.columns = [f"{c}_{suffix}" for c in df.columns]
    return df

def merge_timeframes():
    df_5m  = load_tf(f"{DATA_DIR}/5m.csv",  "5m")
    df_15m = load_tf(f"{DATA_DIR}/15m.csv", "15m")
    df_1h  = load_tf(f"{DATA_DIR}/1h.csv",  "1h")
    df = df_5m.join([df_15m, df_1h], how="inner")
    return df

def add_5m_indicators(df):
    df["rsi_5m"]       = ta.momentum.RSIIndicator(df["close_5m"], window=14).rsi()
    df["atr_5m"]       = ta.volatility.AverageTrueRange(df["high_5m"], df["low_5m"], df["close_5m"]).average_true_range()
    bbands             = ta.volatility.BollingerBands(df["close_5m"], window=20, window_dev=2)
    df["bbw_5m"]       = bbands.bollinger_hband() - bbands.bollinger_lband()
    macd               = ta.trend.MACD(df["close_5m"])
    df["macd_diff_5m"] = macd.macd_diff()
    df["ema_fast_5m"]  = ta.trend.EMAIndicator(df["close_5m"], window=9).ema_indicator()
    df["ema_slow_5m"]  = ta.trend.EMAIndicator(df["close_5m"], window=21).ema_indicator()
    df["vol_log_5m"]   = np.log1p(df["volume_5m"])
    df["ret_5m"]       = df["close_5m"].pct_change()
    df["volatility_5m"]= df["ret_5m"].rolling(10).std()
    return df

def add_higher_tf_context(df, suffix):
    df[f"rsi_{suffix}"]      = ta.momentum.RSIIndicator(df[f"close_{suffix}"], window=14).rsi()
    df[f"ema_slow_{suffix}"] = ta.trend.EMAIndicator(df[f"close_{suffix}"], window=21).ema_indicator()
    df[f"trend_flag_{suffix}"]= (df[f"close_{suffix}"] > df[f"ema_slow_{suffix}"]).astype(int)
    return df

def generate_labels(df):
    df["future_close"] = df["close_5m"].shift(-FUTURE_STEP)
    df["return_5m"]    = (df["future_close"] - df["close_5m"]) / df["close_5m"]
    df = df.dropna(subset=["future_close"])
    # 0=SELL,1=WAIT,2=BUY
    df["direction"]  = pd.cut(
        df["return_5m"],
        bins=[-np.inf, -THRESHOLD, THRESHOLD, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    df["confidence"] = (df["return_5m"].abs() / df["return_5m"].abs().max()).clip(0,1)
    df["reward"]     = df["return_5m"].clip(-0.1, 0.1)
    return df

def balance_classes(df):
    # keep equal n_samples per class
    df_sell = df[df.direction==0]
    df_wait = df[df.direction==1]
    df_buy  = df[df.direction==2]

    target_n = max(len(df_sell), len(df_wait), len(df_buy))
    df_sell_ = resample(df_sell, replace=True, n_samples=target_n, random_state=42)
    df_wait_ = resample(df_wait, replace=True, n_samples=target_n, random_state=42)
    df_buy_  = resample(df_buy,  replace=True, n_samples=target_n, random_state=42)

    return pd.concat([df_sell_, df_wait_, df_buy_]).sort_index()

def scale_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    # persist scaler if needed: joblib.dump(scaler, "scaler.pkl")
    return df

def make_sequences(df, feature_cols):
    X, y_dir, y_conf, y_rwd = [], [], [], []
    for i in range(SEQ_LEN, len(df)):
        X.append(df[feature_cols].iloc[i-SEQ_LEN:i].values)
        y_dir.append(df["direction"].iat[i])
        y_conf.append(df["confidence"].iat[i])
        y_rwd.append(df["reward"].iat[i])
    return np.array(X), np.array(y_dir), np.array(y_conf), np.array(y_rwd)

def main():
    print("▶️  Loading & merging timeframes...")
    df = merge_timeframes()

    print("⚙️  Computing 5m indicators...")
    df = add_5m_indicators(df)

    print("🧠 Adding 15m / 1h context...")
    df = add_higher_tf_context(df, "15m")
    df = add_higher_tf_context(df, "1h")

    print("🏷️ Generating labels...")
    df = generate_labels(df)

    print("🔄 Balancing classes...")
    df = balance_classes(df)

    # drop any remaining NaNs
    df = df.dropna()

    print("🔢 Scaling features...")
    # automatically collect all numeric indicator columns
    feature_cols = [c for c in df.columns
                    if c not in ["future_close","return_5m","direction","confidence","reward"]]
    df = scale_features(df, feature_cols)

    print("📦 Building sequences...")
    X, y_dir, y_conf, y_rwd = make_sequences(df, feature_cols)

    print(f"✅ Shapes → X: {X.shape}, y_dir: {y_dir.shape}, y_conf: {y_conf.shape}, y_rwd: {y_rwd.shape}")
    os.makedirs(SAVE_PREFIX, exist_ok=True)
    np.save(f"{SAVE_PREFIX}/X.npy",         X)
    np.save(f"{SAVE_PREFIX}/y_direction.npy", y_dir)
    np.save(f"{SAVE_PREFIX}/y_confidence.npy", y_conf)
    np.save(f"{SAVE_PREFIX}/y_reward.npy",    y_rwd)
    print("💾 Data saved and ready for TFT training!")

if __name__ == "__main__":
    main()
