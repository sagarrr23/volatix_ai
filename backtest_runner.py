# backtest_runner.py

import os
import sys
import pandas as pd
from datetime import datetime

# === Add project root to path ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# === Local imports ===
from core.strategy_selector import select_strategy, candle_pattern as candle_score, chart_pattern as chart_score, regime_weight
from core.trade_signal_engine import generate_trade_signal
from indicators.candlestick_patterns import detect_candlestick_patterns
from indicators.chart_patterns import detect_chart_patterns
from indicators.regime_classifier import classify_market_regime

# === Load historical data ===
def load_dataframe(file_path):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

# === Higher timeframe bullish filter ===
def is_higher_tf_bullish(df_15m, df_1h):
    fast = df_15m["close"].ewm(span=5).mean()
    slow = df_15m["close"].ewm(span=15).mean()
    long_trend = df_1h["close"] > df_1h["close"].mean()
    return (fast > slow) & long_trend

# === Run Backtest on historical data ===
def run_backtest(data_dir):
    df_1m = load_dataframe(os.path.join(data_dir, "1m.csv"))
    df_5m = load_dataframe(os.path.join(data_dir, "5m.csv"))
    df_15m = load_dataframe(os.path.join(data_dir, "15m.csv"))
    df_1h = load_dataframe(os.path.join(data_dir, "1h.csv"))

    logs = []

    for i in range(50, min(len(df_5m), len(df_15m), len(df_1h))):
        try:
            slice_5m = df_5m.iloc[:i]
            slice_15m = df_15m.iloc[:i]
            slice_1h = df_1h.iloc[:i]

            # === Detect features ===
            slice_5m = detect_candlestick_patterns(slice_5m)
            last_row = slice_5m.iloc[-1]
            candle = last_row.get("pattern")
            chart = detect_chart_patterns(slice_5m)
            regime = classify_market_regime(slice_5m)
            htf_bullish = is_higher_tf_bullish(slice_15m, slice_1h).iloc[-1]

            # === Strategy selection ===
            strategy = select_strategy(candle, chart, regime, htf_bullish)

            # === Confidence scoring ===
            score = 0.0
            if candle in candle_score:
                score += candle_score[candle]
            if chart in chart_score:
                score += chart_score[chart]
            if regime in regime_weight:
                score += regime_weight[regime]
            if regime == "trending_up" and htf_bullish:
                score += 0.2
            elif regime == "trending_down" and not htf_bullish:
                score += 0.2
            confidence = round(min(score, 1.0), 2)

            # === Final signal generation ===
            signal = generate_trade_signal(
                df=slice_5m,
                strategy=strategy,
                candle_pattern=candle,
                chart_pattern=chart,
                regime=regime,
                higher_tf_bullish=htf_bullish
            )

            logs.append({
                "timestamp": slice_5m.index[-1],
                "strategy": strategy,
                "signal": signal["signal"],
                "entry": signal["entry_price"],
                "sl": signal["sl"],
                "tp": signal["tp"],
                "size": signal["size"],
                "confidence": signal["confidence"],
                "reason": signal["reason"],
                "candle": candle,
                "chart": chart,
                "regime": regime,
                "htf_bullish": htf_bullish
            })

        except Exception as e:
            logs.append({
                "timestamp": df_5m.index[i],
                "signal": "ERROR",
                "reason": str(e)
            })

    return pd.DataFrame(logs)

# === Main execution ===
if __name__ == "__main__":
    data_path = "C:/NEWW/volatix_ai/historical_data"
    output_path = "C:/NEWW/volatix_ai/logs/backtest_results.csv"

    result_df = run_backtest(data_path)
    result_df.to_csv(output_path, index=False)

    print(f"✅ Backtest complete. Output saved to: {output_path}")