import pandas as pd

def detect_chart_patterns(df: pd.DataFrame, window_size=20) -> str:
    """
    Detects major chart patterns over a rolling window.
    Returns one of:
    ['ascending_triangle', 'descending_triangle', 'double_top', 'double_bottom', None]
    """

    if len(df) < window_size:
        return None

    df = df.tail(window_size).copy()
    highs = df["high"].values
    lows = df["low"].values

    pattern = None

    # 1. Ascending Triangle: flat top + rising lows
    if is_flat(highs) and is_rising(lows):
        pattern = "ascending_triangle"

    # 2. Descending Triangle: flat bottom + falling highs
    elif is_flat(lows) and is_falling(highs):
        pattern = "descending_triangle"

    # 3. Double Top: two peaks, similar height
    elif has_double_peak(highs):
        pattern = "double_top"

    # 4. Double Bottom: two valleys, similar depth
    elif has_double_trough(lows):
        pattern = "double_bottom"

    return pattern


def is_flat(series, tolerance=0.02):
    """Checks if values are roughly equal (flat line)"""
    avg = sum(series) / len(series)
    return max(series) < avg * (1 + tolerance) and min(series) > avg * (1 - tolerance)


def is_rising(series):
    """Checks for rising trend in lows"""
    return all(x < y for x, y in zip(series, series[1:]))

def is_falling(series):
    """Checks for falling trend in highs"""
    return all(x > y for x, y in zip(series, series[1:]))

def has_double_peak(series, tolerance=0.01):
    """Detects two highs within ~1% of each other"""
    sorted_highs = sorted(series, reverse=True)
    return abs(sorted_highs[0] - sorted_highs[1]) / sorted_highs[0] <= tolerance

def has_double_trough(series, tolerance=0.01):
    """Detects two lows within ~1% of each other"""
    sorted_lows = sorted(series)
    return abs(sorted_lows[0] - sorted_lows[1]) / sorted_lows[0] <= tolerance
