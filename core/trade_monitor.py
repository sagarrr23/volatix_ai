import sys
import os
from datetime import datetime

# Set import path for cross-folder modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.wallet_manager import WalletManager
from core.trade_logger import TradeLogger
from core.config_loader import BotConfig
from core.telegram_notifier import send_telegram_message  # ✅ NEW

wallet = WalletManager()
logger = TradeLogger()


def monitor_trade(open_trade: dict, current_price: float) -> bool:
    """
    Monitors the current trade for TP or SL trigger.

    Args:
        open_trade (dict): The open trade object
        current_price (float): The current market price

    Returns:
        bool: True if trade is closed (TP/SL hit), False otherwise
    """
    if not open_trade:
        return False

    side = open_trade["signal"]
    entry = open_trade["entry_price"]
    sl = open_trade["sl"]
    tp = open_trade["tp"]
    qty = open_trade["size"]
    strategy = open_trade["strategy"]
    confidence = open_trade["confidence"]

    # === Check TP ===
    if side == "BUY" and current_price >= tp:
        pnl = round((tp - entry) * qty, 2)
        return _finalize_trade(entry, tp, pnl, side, qty, strategy, confidence, reason="TP HIT")

    if side == "SELL" and current_price <= tp:
        pnl = round((entry - tp) * qty, 2)
        return _finalize_trade(entry, tp, pnl, side, qty, strategy, confidence, reason="TP HIT")

    # === Check SL ===
    if side == "BUY" and current_price <= sl:
        pnl = round((sl - entry) * qty, 2)
        return _finalize_trade(entry, sl, pnl, side, qty, strategy, confidence, reason="SL HIT")

    if side == "SELL" and current_price >= sl:
        pnl = round((entry - sl) * qty, 2)
        return _finalize_trade(entry, sl, pnl, side, qty, strategy, confidence, reason="SL HIT")

    return False


def _finalize_trade(entry, exit_price, pnl, side, qty, strategy, confidence, reason):
    """
    Handles closing trade, applying wallet logic, logging, and Telegram alert.
    """
    # Update wallet balance
    wallet.apply_trade_result(pnl)

    # Log trade to CSV + JSON
    logger.log_trade(
        symbol=BotConfig.TRADE_SYMBOL,
        side=side,
        qty=qty,
        entry_price=entry,
        exit_price=exit_price,
        pnl=pnl,
        strategy=strategy,
        confidence=confidence,
        mode="live"
    )

    # Send Telegram alert
    emoji = "✅" if pnl > 0 else "❌"
    message = (
        f"{emoji} *TRADE CLOSED ({reason})*\n\n"
        f"*Direction:* `{side}`\n"
        f"*Entry:* `{entry}`\n"
        f"*Exit:* `{exit_price}`\n"
        f"*PnL:* `{pnl}`\n"
        f"*Strategy:* `{strategy}`\n"
        f"*Confidence:* `{confidence}`"
    )

    send_telegram_message(message)

    return True
