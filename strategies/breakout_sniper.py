import pandas as pd
import numpy as np
import talib

def breakout_sniper_signal(df: pd.DataFrame, higher_tf_bullish: bool = True) -> dict:
    """
    Detects explosive breakout entries using:
    - Range compression (Bollinger Band squeeze)
    - High volume breakout bar
    - Price breakout of recent highs/lows
    - Higher TF confirmation
    """

    df = df.copy()
    if len(df) < 50:
        return {"signal": "WAIT", "reason": "Not enough data for breakout detection"}

    # === Bollinger Bands for Squeeze Detection ===
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["bb_width"] = upper - lower
    bb_threshold = df["bb_width"].rolling(50).mean().iloc[-1] * 0.7  # tight squeeze

    # === Identify Breakout Bar ===
    breakout_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    volume_spike = breakout_bar["volume"] > df["volume"].rolling(20).mean().iloc[-1] * 1.5

    # === Resistance and Support Zones ===
    recent_high = df["high"].rolling(20).max().iloc[-2]
    recent_low = df["low"].rolling(20).min().iloc[-2]

    # === BUY Breakout Logic ===
    if (
        breakout_bar["close"] > recent_high and  # price breakout
        breakout_bar["bb_width"] < bb_threshold and  # came from squeeze
        volume_spike and  # volume confirms
        higher_tf_bullish
    ):
        return {
            "signal": "BUY",
            "reason": "Breakout above resistance + volume surge + BB squeeze"
        }

    # === SELL Breakdown Logic ===
    if (
        breakout_bar["close"] < recent_low and  # breakdown
        breakout_bar["bb_width"] < bb_threshold and  # squeeze
        volume_spike and  # strong move
        not higher_tf_bullish
    ):
        return {
            "signal": "SELL",
            "reason": "Breakdown below support + volume spike + BB squeeze"
        }

    # === No Trade ===
    return {
        "signal": "WAIT",
        "reason": "No valid breakout setup"
    }
