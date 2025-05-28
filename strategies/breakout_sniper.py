import pandas as pd
import numpy as np
import pandas_ta as ta

def breakout_sniper_signal(df: pd.DataFrame, higher_tf_bullish: bool = True) -> dict:
    df = df.copy()
    if len(df) < 50:
        return {"signal": "WAIT", "reason": "Not enough data for breakout detection"}

    # === Bollinger Bands for Squeeze Detection ===
    bb = ta.bbands(df["close"], length=20)
    df["bb_width"] = bb["BBU_20_2.0"] - bb["BBL_20_2.0"]
    bb_threshold = df["bb_width"].rolling(50).mean().iloc[-1] * 0.7

    # === Identify Breakout Bar ===
    breakout_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    volume_spike = breakout_bar["volume"] > df["volume"].rolling(20).mean().iloc[-1] * 1.5

    recent_high = df["high"].rolling(20).max().iloc[-2]
    recent_low = df["low"].rolling(20).min().iloc[-2]

    if breakout_bar["close"] > recent_high and breakout_bar["bb_width"] < bb_threshold and volume_spike and higher_tf_bullish:
        return {
            "signal": "BUY",
            "reason": "Breakout above resistance + volume surge + BB squeeze"
        }

    if breakout_bar["close"] < recent_low and breakout_bar["bb_width"] < bb_threshold and volume_spike and not higher_tf_bullish:
        return {
            "signal": "SELL",
            "reason": "Breakdown below support + volume spike + BB squeeze"
        }

    return {"signal": "WAIT", "reason": "No valid breakout setup"}
