import pandas as pd

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'pattern' column to the DataFrame with detected candlestick reversal patterns.
    """

    df = df.copy()
    df["pattern"] = None

    for i in range(2, len(df)):
        # Current and past two candles (position-based)
        o, h, l, c = df.iloc[i][["open", "high", "low", "close"]]
        po, ph, pl, pc = df.iloc[i - 1][["open", "high", "low", "close"]]
        ppo, pph, ppl, ppc = df.iloc[i - 2][["open", "high", "low", "close"]]

        # 1. Bullish Engulfing
        if pc < po and c > o and o < pc and c > po:
            df.at[df.index[i], "pattern"] = "bullish_engulfing"

        # 2. Bearish Engulfing
        elif pc > po and c < o and o > pc and c < po:
            df.at[df.index[i], "pattern"] = "bearish_engulfing"

        # 3. Hammer
        elif (c > o) and ((o - l) > 2 * (c - o)) and ((h - c) < (c - o)):
            df.at[df.index[i], "pattern"] = "hammer"

        # 4. Shooting Star
        elif (o > c) and ((h - o) > 2 * (o - c)) and ((c - l) < (o - c)):
            df.at[df.index[i], "pattern"] = "shooting_star"

        # 5. Doji
        elif abs(c - o) <= 0.001 * (h - l):
            df.at[df.index[i], "pattern"] = "doji"

        # 6. Morning Star (3 candles)
        elif ppc > ppo and pc < po and c > (ppo + ppc) / 2:
            df.at[df.index[i], "pattern"] = "morning_star"

        # 7. Evening Star (3 candles)
        elif ppc < ppo and pc > po and c < (ppo + ppc) / 2:
            df.at[df.index[i], "pattern"] = "evening_star"

    return df
