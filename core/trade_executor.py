import sys
import os
import random
from datetime import datetime

# Project root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.logger import logger
from core.wallet_manager import WalletManager
from core.trade_logger import TradeLogger
from core.telegram_notifier import send_telegram_message
from core.config_loader import BotConfig

# Optional: Only import ccxt if live mode is used
if BotConfig.TRADE_MODE == "live":
    import ccxt

wallet = WalletManager()
logger_engine = TradeLogger()

def execute_trade(signal: dict, strategy: str, live_mode=False) -> bool:
    """
    Executes a trade based on strategy signal.
    Supports live (via CCXT) and simulated (backtest) modes.
    """
    if signal["signal"] == "WAIT":
        logger.warning("🟡 No trade executed — WAIT signal.")
        return False

    side = signal["signal"]
    entry_price = signal["entry_price"]
    sl = signal["sl"]
    tp = signal["tp"]
    size = signal["size"]
    confidence = signal["confidence"]
    reason = signal.get("reason", "AI Strategy Signal")

    # === LIVE EXECUTION MODE ===
    if live_mode and BotConfig.TRADE_MODE == "live":
        try:
            exchange_class = getattr(ccxt, BotConfig.TRADE_EXCHANGE)
            exchange = exchange_class({
                'apiKey': BotConfig.EXCHANGE_API_KEY,
                'secret': BotConfig.EXCHANGE_SECRET_KEY,
                'enableRateLimit': True
            })

            if side == "BUY":
                order = exchange.create_market_buy_order(BotConfig.TRADE_SYMBOL, size)
            else:
                order = exchange.create_market_sell_order(BotConfig.TRADE_SYMBOL, size)

            actual_fill = order['average'] or order['price']
            pnl = round((tp - actual_fill) * size if side == "BUY" else (actual_fill - tp) * size, 2)

            logger.success(f"🟢 LIVE Trade Executed: {side} {size} @ {actual_fill}")
            _finalize_trade(
                entry=actual_fill,
                exit_price=tp,
                pnl=pnl,
                side=side,
                qty=size,
                strategy=strategy,
                confidence=confidence,
                reason="LIVE Market Order Filled",
                mode="live"
            )
            return True

        except Exception as e:
            logger.error(f"❌ LIVE trade failed: {e}")
            return False

    # === SIMULATED (BACKTEST) MODE ===
    exit_price = tp if random.random() > 0.5 else sl
    pnl = (exit_price - entry_price) * size if side == "BUY" else (entry_price - exit_price) * size
    pnl = round(pnl, 2)

    logger.info("🧪 Simulated trade triggered.")
    return _finalize_trade(
        entry=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        side=side,
        qty=size,
        strategy=strategy,
        confidence=confidence,
        reason=reason,
        mode="backtest"
    )

# ================================
# 🔧 Finalization + Logging
# ================================
def _finalize_trade(entry, exit_price, pnl, side, qty, strategy, confidence, reason, mode):
    wallet.apply_trade_result(pnl)

    logger_engine.log_trade(
        symbol=BotConfig.TRADE_SYMBOL,
        side=side,
        qty=qty,
        entry_price=entry,
        exit_price=exit_price,
        pnl=pnl,
        strategy=strategy,
        confidence=confidence,
        mode=mode
    )

    emoji = "✅" if pnl > 0 else "❌"
    send_telegram_message(
        f"{emoji} *TRADE EXECUTED ({mode.upper()})*\n\n"
        f"*Strategy:* `{strategy}`\n"
        f"*Direction:* `{side}`\n"
        f"*Entry:* `{entry}`\n"
        f"*Exit:* `{exit_price}`\n"
        f"*PnL:* `{pnl}`\n"
        f"*Confidence:* `{confidence}`\n"
        f"*Reason:* `{reason}`"
    )
    return True


# === Test Block ===
if __name__ == "__main__":
    test_signal = {
        "signal": "BUY",
        "entry_price": 100.0,
        "sl": 95.0,
        "tp": 112.0,
        "size": 1.5,
        "confidence": 0.91,
        "reason": "Test signal"
    }

    executed = execute_trade(test_signal, strategy="Breakout Sniper", live_mode=False)
    print("Trade executed:", executed)
