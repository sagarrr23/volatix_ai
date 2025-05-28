import pandas as pd
import os

def process_full_1m_csv(input_path, output_dir):
    print("🔄 Loading raw 1m BTC data...")
    df = pd.read_csv(input_path)

    # 🕓 Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    # 🎯 Ensure correct columns exist
    df = df[["open", "high", "low", "close", "volume"]]

    # 🔢 Convert all columns to float
    df = df.astype(float)

    # 🧽 Drop rows with NaNs or zero volume
    df = df.dropna()
    df = df[df["volume"] > 0]

    # 💾 Save 1m
    df.to_csv(os.path.join(output_dir, "1m.csv"))
    print("✅ Saved: 1m.csv")

    # 🔁 Generate 5m, 15m, 1h
    timeframes = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h"
    }

    for name, rule in timeframes.items():
        df_resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        df_resampled.to_csv(os.path.join(output_dir, f"{name}.csv"))
        print(f"✅ Saved: {name}.csv")

    print("🏁 Done. All timeframes ready for TFT.")

# Example usage
if __name__ == "__main__":
    process_full_1m_csv(
        input_path="historical_data/btcusd_bitstamp_1min_latest.csv",
        output_dir="historical_data"
    )
