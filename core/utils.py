def dynamic_exit_levels(df, direction, atr, risk_reward=1.8):
    """
    Calculates dynamic SL and TP using:
    - Swing high/low structure (last 10 candles)
    - ATR buffer for noise protection
    - Last candle extremes for validation
    - Adjustable RR for TP
    """

    df = df.copy()
    last = df.iloc[-1]
    close = last["close"]
    candle_low = last["low"]
    candle_high = last["high"]

    if direction == "BUY":
        recent_lows = df["low"].tail(10)
        swing_low = recent_lows.min()
        
        # Apply ATR buffer
        sl_base = min(swing_low, candle_low)
        sl = sl_base - atr * 0.5

        # SL must be below close
        if sl >= close:
            sl = close - atr * 1.2

        tp = close + (close - sl) * risk_reward

    elif direction == "SELL":
        recent_highs = df["high"].tail(10)
        swing_high = recent_highs.max()

        # Apply ATR buffer
        sl_base = max(swing_high, candle_high)
        sl = sl_base + atr * 0.5

        # SL must be above close
        if sl <= close:
            sl = close + atr * 1.2

        tp = close - (sl - close) * risk_reward

    else:
        sl, tp = None, None

    return round(sl, 2), round(tp, 2)
