import pandas as pd
import numpy as np

def classify_market_regime(df: pd.DataFrame, window=14) -> str:
    """
    Analyzes the price action to classify the market regime.
    Returns one of:
    ['trending_up', 'trending_down', 'sideways_range', 'volatility_spike']
    """

    df = df.copy()

    if len(df) < window + 2:
        return "unknown"

    # Calculate moving averages
    df["ema_fast"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=window, adjust=False).mean()

    # Calculate volatility via ATR (Average True Range)
    df["hl"] = df["high"] - df["low"]
    df["hc"] = np.abs(df["high"] - df["close"].shift(1))
    df["lc"] = np.abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["hl", "hc", "lc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window).mean()

    # Calculate Bollinger Band width
    df["ma"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    df["bb_width"] = (2 * df["std"]) / df["ma"]

    # Conditions
    last = df.iloc[-1]
    price_slope = last["ema_fast"] - last["ema_slow"]

    if price_slope > 0.5 and last["atr"] > df["atr"].mean():
        return "trending_up"
    elif price_slope < -0.5 and last["atr"] > df["atr"].mean():
        return "trending_down"
    elif last["bb_width"] < 0.01:
        return "sideways_range"
    elif last["atr"] > 1.5 * df["atr"].mean():
        return "volatility_spike"

    return "unknown"
