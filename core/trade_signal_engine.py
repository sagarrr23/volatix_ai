from core.wallet_manager import WalletManager
from strategies.breakout_sniper import breakout_sniper_signal
from strategies.reversal_sniper import reversal_sniper_signal
from strategies.trend_rider import trend_rider_signal
from core.utils import dynamic_exit_levels
from ai.tft_predictor import predict_next_move

import numpy as np

# ========================================
# 🔁 Central Signal Engine
# ========================================
def generate_trade_signal(
    df,
    strategy,
    candle_pattern,
    chart_pattern,
    regime,
    higher_tf_bullish=True,
    backtest=False
):
    """
    Unified signal decision engine combining:
    - AI prediction (TFT brain)
    - Strategy signals
    - Capital control
    - Pattern confluence
    - Risk-managed trade construction
    """

    wallet = WalletManager()

    # 🛑 1. Capital protection check
    if not wallet.can_trade_today():
        return _no_trade("🛑 Daily loss limit hit. Trading disabled for today.")

    # 🚫 2. Strategy block
    if strategy == "wait":
        return _no_trade("⚠️ Strategy is wait.")

    # ✅ 3. Extract latest price
    last = df.iloc[-1]
    close_price = last["close"]
    atr = df["atr"].iloc[-1] if "atr" in df.columns else df["high"].sub(df["low"]).rolling(14).mean().iloc[-1]

    # 🎯 4. Base confidence scoring from patterns and strategy type
    score = 0
    if candle_pattern: score += 0.4
    if chart_pattern: score += 0.3
    if regime.startswith("trending") and strategy == "trend_rider": score += 0.3
    elif regime == "sideways_range" and strategy == "reversal_sniper": score += 0.3
    elif regime == "volatility_spike" and strategy == "breakout_hunter": score += 0.3
    confidence = round(min(score, 1.0), 2)

    # 🧠 5. AI Prediction – Optional Boost or Override
    if len(df) >= 20:
        try:
            sequence = df.tail(20).values  # shape: (20, N)
            ai_result = predict_next_move(sequence)
            ai_signal = ai_result["direction"]
            ai_conf = ai_result["confidence"]
            ai_reward = ai_result["reward"]

            # Optional: Combine AI confidence into final score
            confidence = round((confidence + ai_conf) / 2, 2)
        except Exception as e:
            ai_signal = "WAIT"
            ai_conf = 0.0
            ai_reward = 0.0
    else:
        ai_signal = "WAIT"
        ai_conf = 0.0
        ai_reward = 0.0

    # 🔁 6. Strategy-based signal routing
    if strategy == "breakout_hunter":
        result = breakout_sniper_signal(df, higher_tf_bullish=higher_tf_bullish)
    elif strategy == "reversal_sniper":
        result = reversal_sniper_signal(df, higher_tf_bullish=higher_tf_bullish)
    elif strategy == "trend_rider":
        result = trend_rider_signal(df, higher_tf_bullish=higher_tf_bullish)
    else:
        return _no_trade(f"❓ Unknown strategy: {strategy}", price=close_price, confidence=confidence)

    # 🤖 AI filter enforcement (optional strict rule)
    if ai_signal != "WAIT" and ai_conf > 0.5:
        result["reason"] += f" | AI confirms: {ai_signal}, confidence={ai_conf}, reward={ai_reward}"
    else:
        result["reason"] += f" | AI blocked or weak ({ai_signal}, {ai_conf})"
        result["signal"] = "WAIT"

    return _route_result(result, close_price, atr, confidence, wallet, df)

# ========================================
# ✅ Trade Builder with Strong SL/TP Fallback
# ========================================
def _build_trade(direction, price, atr, confidence, reason, wallet, df):
    try:
        sl, tp = dynamic_exit_levels(df, direction, atr)

        min_sl_dist = price * 0.0025
        min_tp_dist = price * 0.005

        if sl is None or np.isnan(sl):
            sl = price - min_sl_dist if direction == "BUY" else price + min_sl_dist
        if tp is None or np.isnan(tp):
            tp = price + min_tp_dist if direction == "BUY" else price - min_tp_dist

        if abs(price - sl) < min_sl_dist:
            sl = price - min_sl_dist if direction == "BUY" else price + min_sl_dist
        if abs(tp - price) < min_tp_dist:
            tp = price + min_tp_dist if direction == "BUY" else price - min_tp_dist

        size = wallet.get_smart_trade_size(price, sl)

        return {
            "signal": direction,
            "confidence": confidence,
            "entry_price": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "size": round(size, 2),
            "reason": reason + " (with SL/TP fallback)"
        }

    except Exception as e:
        return _no_trade(f"SL/TP generation error: {str(e)}", price=price, confidence=confidence)

# ========================================
# ❌ No-Trade Signal Builder
# ========================================
def _no_trade(reason, price=None, confidence=0.0):
    return {
        "signal": "WAIT",
        "confidence": confidence,
        "entry_price": price,
        "sl": None,
        "tp": None,
        "size": 0.0,
        "reason": reason
    }

# ========================================
# 🔁 Signal Router
# ========================================
def _route_result(result, price, atr, confidence, wallet, df):
    if result["signal"] != "WAIT":
        return _build_trade(
            direction=result["signal"],
            price=price,
            atr=atr,
            confidence=confidence,
            reason=result["reason"],
            wallet=wallet,
            df=df
        )
    else:
        return _no_trade(
            reason=result["reason"],
            price=price,
            confidence=confidence
        )