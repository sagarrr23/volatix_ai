import ccxt
import pandas as pd
import time
import os
from datetime import datetime

# === Configuration ===
exchange = ccxt.kraken()
symbol = "XXBTZUSD"  # Kraken ticker for BTC/USD
timeframes = ["1m", "5m", "15m", "1h"]
months_back = 7
data_dir = "historical_data"
os.makedirs(data_dir, exist_ok=True)

# Start time
now = exchange.milliseconds()
start_time = int((datetime.now() - pd.DateOffset(months=months_back)).timestamp() * 1000)

for tf in timeframes:
    tf_minutes = exchange.parse_timeframe(tf)
    since = start_time
    all_data = []

    while since < now:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=500)
            if not candles:
                break
            all_data.extend(candles)
            since = candles[-1][0] + tf_minutes * 60 * 1000
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching {tf}: {e}")
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.to_csv(f"{data_dir}/btc_{tf}_7months.csv")
    print(f"[✓] Saved {tf} data with {len(df)} rows.")
