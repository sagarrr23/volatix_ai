import pandas as pd
import os

def convert_bitstamp_csv(input_path, output_dir):
    print("🔄 Loading raw Bitstamp 1m data...")
    df = pd.read_csv(input_path)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()

    # Scale prices down (prices are multiplied by 100 in original)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] / 100.0

    # Volume stays unchanged

    # Keep only needed columns
    df = df[["open", "high", "low", "close", "volume"]]

    print("✅ Saving 1m.csv...")
    df.to_csv(os.path.join(output_dir, "1m.csv"))

    # Resampling logic for higher timeframes
    for tf, name in zip(["5min", "15min", "1H"], ["5m", "15m", "1h"]):
        df_tf = df.resample(tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        out_path = os.path.join(output_dir, f"{name}.csv")
        df_tf.to_csv(out_path)
        print(f"✅ Saved {name}.csv")

    print("🏁 Conversion complete. All files saved to:", output_dir)


# === Example Usage ===
if __name__ == "__main__":
    convert_bitstamp_csv(
        input_path="historical_data/btcusd_bitstamp_1min_latest.csv",
        output_dir="historical_data"
    )
