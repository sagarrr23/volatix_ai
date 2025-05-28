# core/strategy_selector.py

# === Confidence Weights ===
candle_pattern = {
    "doji": 0.15,
    "hammer": 0.20,
    "shooting_star": 0.20,
    "bullish_engulfing": 0.25,
    "bearish_engulfing": 0.25,
    "morning_star": 0.3,
    "evening_star": 0.3
}

chart_pattern = {
    "double_top": 0.25,
    "double_bottom": 0.25,
    "ascending_triangle": 0.3,
    "descending_triangle": 0.3,
    "head_and_shoulders": 0.35,
    "inverse_head_and_shoulders": 0.35
}

regime_weight = {
    "trending_up": 0.2,
    "trending_down": 0.2,
    "sideways_range": 0.15,
    "volatility_spike": 0.1
}

# === Strategy Selector ===
def select_strategy(
    candle_pattern_name: str,
    chart_pattern_name: str,
    regime: str,
    higher_tf_bullish: bool = None
) -> str:
    """
    Dynamically selects strategy using weighted scoring:
    Returns one of:
    ['reversal_sniper', 'trend_rider', 'breakout_hunter', 'wait']
    """

    # Strategy scores
    reversal_score = 0
    trend_score = 0
    breakout_score = 0

    # === Add regime contribution
    if regime == "sideways_range":
        reversal_score += regime_weight.get(regime, 0)
    elif regime.startswith("trending"):
        trend_score += regime_weight.get(regime, 0)
    elif regime == "volatility_spike":
        breakout_score += regime_weight.get(regime, 0)

    # === Candle pattern contribution
    if candle_pattern_name in candle_pattern:
        weight = candle_pattern[candle_pattern_name]
        if regime == "sideways_range":
            reversal_score += weight
        elif regime.startswith("trending"):
            trend_score += weight
        elif regime == "volatility_spike":
            breakout_score += weight * 0.5  # Less important

    # === Chart pattern contribution
    if chart_pattern_name in chart_pattern:
        weight = chart_pattern[chart_pattern_name]
        if regime == "sideways_range":
            reversal_score += weight
        elif regime.startswith("trending"):
            trend_score += weight
        elif regime == "volatility_spike":
            breakout_score += weight

    # === Higher TF Confirmation
    if regime == "trending_up" and higher_tf_bullish:
        trend_score += 0.2
    if regime == "trending_down" and higher_tf_bullish is False:
        trend_score += 0.2

    # === Final selection based on max score
    scores = {
        "reversal_sniper": reversal_score,
        "trend_rider": trend_score,
        "breakout_hunter": breakout_score
    }

    best_strategy = max(scores, key=scores.get)

    # Set a minimum threshold to avoid false trades
    if scores[best_strategy] < 0.45:
        return "wait"

    return best_strategy
