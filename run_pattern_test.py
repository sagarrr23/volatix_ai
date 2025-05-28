import sys
import os
import pandas as pd

# === Path Fix ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# === Imports ===
from core.price_fetcher import fetch_ohlcv
from indicators.candlestick_patterns import detect_candlestick_patterns
from indicators.chart_patterns import detect_chart_patterns
from indicators.regime_classifier import classify_market_regime
from core.strategy_selector import select_strategy, candle_pattern as candle_score, chart_pattern as chart_score, regime_weight
from core.trade_signal_engine import generate_trade_signal

# === Convert API OHLCV response → DataFrame ===
def to_dataframe(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# === Multi-Timeframe Trend Filter ===
def is_higher_tf_bullish(df_15m, df_1h):
    fast = df_15m["close"].ewm(span=5).mean().iloc[-1]
    slow = df_15m["close"].ewm(span=15).mean().iloc[-1]
    long_trend = df_1h["close"].iloc[-1] > df_1h["close"].mean()
    return fast > slow and long_trend

# === Run Signal Engine Test ===
if __name__ == "__main__":
    print("🔄 Running Volatix AI Signal Engine...\n")

    # === 1. Fetch Data ===
    df_5m = to_dataframe(fetch_ohlcv(timeframe="5m", limit=100))
    df_15m = to_dataframe(fetch_ohlcv(timeframe="15m", limit=100))
    df_1h = to_dataframe(fetch_ohlcv(timeframe="1h", limit=100))

    if df_5m.empty or df_15m.empty or df_1h.empty:
        print("❌ Failed to fetch candles from exchange.")
        exit()

    # === 2. Pattern Detection ===
    df_5m = detect_candlestick_patterns(df_5m)
    last_row = df_5m.iloc[-1]
    candle = last_row.get("pattern", None)

    if candle:
        print(f"🧠 Candlestick Pattern: {candle} | Close: {last_row['close']}")
    else:
        print("⚪ No candlestick pattern in last candle.")

    chart = detect_chart_patterns(df_5m)
    print(f"📐 Chart Pattern Detected: {chart if chart else 'None'}")

    # === 3. Regime Detection ===
    regime = classify_market_regime(df_5m)
    print(f"📊 Market Regime: {regime}")

    # === 4. Higher TF Trend Filter ===
    htf_bullish = is_higher_tf_bullish(df_15m, df_1h)
    print(f"📈 Higher Timeframe Bullish: {htf_bullish}")

    # === 5. Strategy Selection ===
    strategy = select_strategy(candle, chart, regime, htf_bullish)
    print(f"🎯 Selected Strategy: {strategy}")

    # === 6. Confidence Score ===
    score = 0.0
    if candle in candle_score:
        score += candle_score[candle]
    if chart in chart_score:
        score += chart_score[chart]
    if regime in regime_weight:
        score += regime_weight[regime]
    if (regime == "trending_up" and htf_bullish) or (regime == "trending_down" and not htf_bullish):
        score += 0.2

    print(f"📊 Confidence Score: {score:.2f}")

    # === 7. Final Trade Signal ===
    signal = generate_trade_signal(
        df=df_5m,
        strategy=strategy,
        candle_pattern=candle,
        chart_pattern=chart,
        regime=regime,
        higher_tf_bullish=htf_bullish
    )

    print(f"🚀 Final Signal: {signal['signal']}")
    print(f"📈 Entry: {signal['entry_price']} | SL: {signal['sl']} | TP: {signal['tp']} | Size: {signal['size']}")
    print(f"📌 Reason: {signal['reason']}")

# === 8. Log Signal to CSV and JSON ===
import json
from datetime import datetime

os.makedirs("logs", exist_ok=True)

log_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "signal": signal["signal"],
    "strategy": strategy,
    "confidence": signal["confidence"],
    "entry_price": signal["entry_price"],
    "sl": signal["sl"],
    "tp": signal["tp"],
    "size": signal["size"],
    "reason": signal["reason"]
}

# ---- CSV Logging ----
csv_path = "logs/signals.csv"
csv_columns = list(log_entry.keys())
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    if write_header:
        writer.writeheader()
    writer.writerow(log_entry)

# ---- JSON Logging ----
json_path = "logs/signals.json"
if os.path.exists(json_path):
    with open(json_path, "r") as jf:
        json_data = json.load(jf)
else:
    json_data = []

json_data.append(log_entry)
with open(json_path, "w") as jf:
    json.dump(json_data, jf, indent=2)
print(f"📂 Signal logged to {csv_path} and {json_path}")