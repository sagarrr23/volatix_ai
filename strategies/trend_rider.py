import pandas as pd
import numpy as np
import talib

def trend_rider_signal(df: pd.DataFrame, higher_tf_bullish: bool = True) -> dict:
    """
    Advanced trend-following logic using:
    - EMA stack
    - MACD histogram shift
    - Volume spike
    - Higher timeframe confluence
    """

    df = df.copy()
    if len(df) < 60:
        return {"signal": "WAIT", "reason": "Not enough candles"}

    # === EMAs ===
    df["ema_10"] = talib.EMA(df["close"], timeperiod=10)
    df["ema_20"] = talib.EMA(df["close"], timeperiod=20)
    df["ema_50"] = talib.EMA(df["close"], timeperiod=50)

    # === MACD Histogram ===
    macd, macdsignal, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd_hist"] = macdhist

    # === Volume Spike Detection ===
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_ma"] * 1.2  # 20% surge

    # === Latest Values ===
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # === BUY Conditions ===
    ema_up = last["ema_10"] > last["ema_20"] > last["ema_50"]
    macd_up = last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]
    vol_ok = last["vol_spike"]

    # === SELL Conditions ===
    ema_down = last["ema_10"] < last["ema_20"] < last["ema_50"]
    macd_down = last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]

    # === Decision Tree ===
    if ema_up and macd_up and vol_ok and higher_tf_bullish:
        return {
            "signal": "BUY",
            "reason": "Trend up: EMA stack + rising MACD + volume spike"
        }

    elif ema_down and macd_down and vol_ok and not higher_tf_bullish:
        return {
            "signal": "SELL",
            "reason": "Trend down: EMA stack + falling MACD + volume spike"
        }

    return {
        "signal": "WAIT",
        "reason": "No clear trend with volume confirmation"
    }
