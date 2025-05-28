import pandas as pd
import numpy as np
import talib

def reversal_sniper_signal(df: pd.DataFrame, higher_tf_bullish: bool = True) -> dict:
    """
    High-IQ reversal signal with:
    - RSI zones
    - Candle exhaustion
    - Volume fade
    - Psychological trap logic
    """

    df = df.copy()
    if len(df) < 30:
        return {"signal": "WAIT", "reason": "Insufficient data"}

    # === Indicators
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["volume_avg"] = df["volume"].rolling(10).mean()

    # === Last Candle
    last = df.iloc[-1]
    prev = df.iloc[-2]
    candle = None
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"]

    # === Candle type logic
    if body < 0.2 * range_:
        candle = "doji"
    elif last["close"] > last["open"] and (last["open"] - last["low"]) > 2 * body:
        candle = "hammer"
    elif last["open"] > last["close"] and (last["high"] - last["open"]) > 2 * body:
        candle = "shooting_star"
    elif last["close"] > prev["open"] and last["open"] < prev["close"]:
        candle = "bullish_engulfing"
    elif last["close"] < prev["open"] and last["open"] > prev["close"]:
        candle = "bearish_engulfing"

    # === RSI Zones
    rsi = df["rsi"].iloc[-1]
    oversold = rsi < 35
    overbought = rsi > 65

    # === Volume divergence (fading)
    volume_fade = last["volume"] < df["volume_avg"].iloc[-1]

    # === Context logic: exhaustion in last 5 candles
    tail_ratio = (df["high"] - df["close"]).rolling(5).mean() / df["atr"].rolling(5).mean()
    wicks_high = tail_ratio.iloc[-1] > 0.5

    # === BUY Logic
    if oversold and candle in ["hammer", "bullish_engulfing", "doji"] and volume_fade and higher_tf_bullish and not wicks_high:
        return {
            "signal": "BUY",
            "reason": f"{candle} + RSI {rsi:.2f} oversold + vol fade + HTF bullish"
        }

    # === SELL Logic
    wick_low = (df["close"] - df["low"]).rolling(5).mean() / df["atr"].rolling(5).mean()
    wicks_low = wick_low.iloc[-1] > 0.5

    if overbought and candle in ["shooting_star", "bearish_engulfing", "doji"] and volume_fade and not higher_tf_bullish and not wicks_low:
        return {
            "signal": "SELL",
            "reason": f"{candle} + RSI {rsi:.2f} overbought + vol fade + HTF bearish"
        }

    return {"signal": "WAIT", "reason": "No valid reversal setup"}
