
import time
import traceback
import pandas as pd

# === Core Imports ===
from core.logger import logger
from core.price_fetcher import fetch_ohlcv
from indicators.candlestick_patterns import detect_candlestick_patterns
from indicators.chart_patterns import detect_chart_patterns
from indicators.regime_classifier import classify_market_regime
from core.strategy_selector import select_strategy
from core.trade_signal_engine import generate_trade_signal
from core.trade_executor import execute_trade

# === AI Brain Inference ===
from ai.live_brain import update_brain_input

# === Utility: Convert OHLCV to DataFrame ===
def to_dataframe(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# === Trend Filter on Higher TFs ===
def is_higher_tf_bullish(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> bool:
    fast = df_15m["close"].ewm(span=5).mean().iloc[-1]
    slow = df_15m["close"].ewm(span=15).mean().iloc[-1]
    long_trend = df_1h["close"].iloc[-1] > df_1h["close"].mean()
    return fast > slow and long_trend

# === Feature Engineering for Brain ===
def prepare_ai_features(df_1m, df_5m, df_15m, df_1h):
    try:
        last_row = df_5m.iloc[-1]
        features = [
            last_row["open"], last_row["high"], last_row["low"], last_row["close"], last_row["volume"],
            df_1m["close"].pct_change().mean(),
            df_15m["close"].pct_change().mean(),
            df_1h["close"].pct_change().mean(),
            df_5m["volume"].rolling(5).mean().iloc[-1],
            df_5m["close"].rolling(5).std().iloc[-1],
        ]
        # Pad with zeros if needed (should match model's 60 features)
        while len(features) < 60:
            features.append(0.0)
        return pd.Series(features[:60]).values
    except Exception as e:
        logger.warning(f"⚠️ Failed to prepare features: {str(e)}")
        return None

# === MAIN LOOP ===
def run_forever():
    logger.info("🔁 Volatix AI bot loop starting...")
    while True:
        try:
            # 1️⃣ Fetch multi-timeframe candles
            timeframes = {
                "1m": to_dataframe(fetch_ohlcv(timeframe="1m", limit=100)),
                "5m": to_dataframe(fetch_ohlcv(timeframe="5m", limit=100)),
                "15m": to_dataframe(fetch_ohlcv(timeframe="15m", limit=100)),
                "1h": to_dataframe(fetch_ohlcv(timeframe="1h", limit=100))
            }

            if any(df.empty for df in timeframes.values()):
                logger.warning("⚠️ Skipping cycle — one or more timeframes failed to load.")
                time.sleep(30)
                continue

            # 2️⃣ Brain AI Inference
            features = prepare_ai_features(timeframes["1m"], timeframes["5m"], timeframes["15m"], timeframes["1h"])
            if features is not None:
                brain_decision = update_brain_input(features)
                if brain_decision:
                    logger.info(f"🧠 Brain Decision → Direction: {brain_decision['direction']} | Confidence: {brain_decision['confidence']} | Reward: {brain_decision['reward']}")

            # 3️⃣ Strategy Stack
            df_5m = detect_candlestick_patterns(timeframes["5m"])
            candle_pattern = df_5m.iloc[-1].get("pattern")
            chart_pattern = detect_chart_patterns(df_5m)
            regime = classify_market_regime(df_5m)
            higher_tf_bullish = is_higher_tf_bullish(timeframes["15m"], timeframes["1h"])

            logger.info(f"📊 Regime: {regime} | 📈 HTF Bullish: {higher_tf_bullish} | 🧠 Candle: {candle_pattern or 'None'} | 📐 Chart: {chart_pattern or 'None'}")

            # 4️⃣ Strategy Selection
            strategy = select_strategy(
                candle_pattern_name=candle_pattern,
                chart_pattern_name=chart_pattern,
                regime=regime,
                higher_tf_bullish=higher_tf_bullish
            )

            logger.info(f"🎯 Strategy Chosen: {strategy}")

            # 5️⃣ Trade Signal Generation
            signal = generate_trade_signal(
                df=df_5m,
                strategy=strategy,
                candle_pattern=candle_pattern,
                chart_pattern=chart_pattern,
                regime=regime,
                higher_tf_bullish=higher_tf_bullish
            )

            # 6️⃣ Execute or Skip Trade
            if signal["signal"] in ["BUY", "SELL"]:
                logger.info(f"📥 TRADE: {signal['signal']} {signal['size']} @ {signal['entry_price']} | SL: {signal['sl']} | TP: {signal['tp']}")
                execute_trade(signal, strategy=strategy, live_mode=False)
            else:
                logger.info(f"⏳ Signal Skipped: {signal['reason']}")

            time.sleep(30)

        except Exception as e:
            logger.error(f"🔥 Loop Error: {str(e)}")
            traceback.print_exc()
            time.sleep(30)

# === Entry Point ===
if __name__ == "__main__":
    run_forever()
