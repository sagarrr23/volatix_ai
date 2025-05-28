import pandas as pd
import numpy as np
import pandas_ta as ta

def trend_rider_signal(df: pd.DataFrame, higher_tf_bullish: bool = True) -> dict:
    df = df.copy()
    if len(df) < 60:
        return {"signal": "WAIT", "reason": "Not enough candles"}

    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)

    macd = ta.macd(df["close"])
    df["macd_hist"] = macd["MACDh_12_26_9"]

    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > df["vol_ma"] * 1.2

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_up = last["ema_10"] > last["ema_20"] > last["ema_50"]
    macd_up = last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]
    ema_down = last["ema_10"] < last["ema_20"] < last["ema_50"]
    macd_down = last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]

    if ema_up and macd_up and last["vol_spike"] and higher_tf_bullish:
        return {"signal": "BUY", "reason": "Trend up: EMA stack + rising MACD + volume spike"}

    if ema_down and macd_down and last["vol_spike"] and not higher_tf_bullish:
        return {"signal": "SELL", "reason": "Trend down: EMA stack + falling MACD + volume spike"}

    return {"signal": "WAIT", "reason": "No clear trend with volume confirmation"}
