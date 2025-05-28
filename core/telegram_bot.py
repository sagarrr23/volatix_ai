import os
import sys
import json
import traceback
from datetime import datetime
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)

# Extend import path to access core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.wallet_manager import WalletManager
from core.config_loader import BotConfig
from core.logger import logger
from core.trade_logger import TradeLogger
from core.trade_executor import execute_trade

# Instantiate shared objects
wallet = WalletManager()
trade_logger = TradeLogger()
bot_token = BotConfig.TELEGRAM_BOT_TOKEN
admin_id = int(BotConfig.TELEGRAM_CHAT_ID)


# =============================
# 🛡️ Admin Access Decorator
# =============================
def admin_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != admin_id:
            await update.message.reply_text("🚫 Access Denied.")
            return
        return await func(update, context)
    return wrapper


# =============================
# 📋 Command Handlers
# =============================

@admin_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome to Volatix AI.\nUse /help to see all commands.")

@admin_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
🧠 Volatix AI – Admin Commands:
/wallet - Show wallet balance
/resetloss - Reset today's loss counter
/setrisk X - Set trade risk % (e.g., /setrisk 0.05)
/setlosslimit X - Set max daily drawdown (e.g., /setlosslimit 0.1)
/lasttrade - Show details of last trade
/logs - Send trade CSV log file
/manualtrade BUY/SELL entry sl tp size - Simulate trade
/strategy - Show current strategy name
/stopbot - ⛔ Coming soon
/restartbot - 🔄 Coming soon
    """)

@admin_only
async def wallet_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"👜 Balance: ${wallet.get_balance()}\n"
        f"📉 Daily Loss: ${wallet.get_daily_loss()}\n"
        f"💼 Max Per Trade: ${wallet.get_trade_amount()}"
    )

@admin_only
async def reset_daily_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wallet.reset_daily_loss()
    await update.message.reply_text("🔄 Daily loss tracker reset.")

@admin_only
async def set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        risk = float(context.args[0])
        BotConfig.MAX_TRADE_PERCENT = risk
        await update.message.reply_text(f"📊 Max trade risk set to {risk*100:.2f}%.")
    except:
        await update.message.reply_text("⚠️ Usage: /setrisk 0.05")

@admin_only
async def set_loss_limit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        loss = float(context.args[0])
        BotConfig.DAILY_LOSS_LIMIT = loss
        await update.message.reply_text(f"🚫 Max daily loss set to {loss*100:.2f}%.")
    except:
        await update.message.reply_text("⚠️ Usage: /setlosslimit 0.1")

@admin_only
async def last_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open("logs/trades.json", "r") as f:
            trades = json.load(f)
            last = trades[-1]
            msg = (
                f"📈 Last Trade ({last['mode']}):\n"
                f"{last['timestamp']}\n"
                f"{last['side']} {last['qty']} {last['symbol']}\n"
                f"Entry: {last['entry_price']} | Exit: {last['exit_price']}\n"
                f"PnL: {last['pnl']} | Strategy: {last['strategy']}\n"
                f"Confidence: {last['confidence']}"
            )
            await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text("❌ Could not read last trade.")
        logger.error(f"Failed to load last trade: {e}")

@admin_only
async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=open("logs/trades.csv", "rb")
        )
    except Exception as e:
        await update.message.reply_text("❌ Error sending logs.")
        logger.error(f"Log send error: {e}")

@admin_only
async def manual_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        side, entry, sl, tp, size = context.args
        signal = {
            "signal": side.upper(),
            "entry_price": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "size": float(size),
            "confidence": 0.99,
            "reason": "manual"
        }
        execute_trade(signal, strategy="Manual")
        await update.message.reply_text("📥 Manual trade executed.")
    except:
        await update.message.reply_text("⚠️ Usage: /manualtrade BUY 100 95 112 1.5")

@admin_only
async def strategy_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open("logs/strategy_status.txt", "r") as f:
            status = f.read().strip()
        await update.message.reply_text(f"🎯 Active Strategy: {status}")
    except:
        await update.message.reply_text("⚠️ Strategy status not found.")

# Future placeholder commands
@admin_only
async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🛑 Bot stop not implemented in this version.")

@admin_only
async def restart_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Restart not implemented in this version.")


# =============================
# 🔌 Bot Setup and Launch
# =============================
def main():
    app = ApplicationBuilder().token(bot_token).build()

    # Command bindings
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("wallet", wallet_status))
    app.add_handler(CommandHandler("resetloss", reset_daily_loss))
    app.add_handler(CommandHandler("setrisk", set_risk))
    app.add_handler(CommandHandler("setlosslimit", set_loss_limit))
    app.add_handler(CommandHandler("lasttrade", last_trade))
    app.add_handler(CommandHandler("logs", logs))
    app.add_handler(CommandHandler("manualtrade", manual_trade))
    app.add_handler(CommandHandler("strategy", strategy_status))
    app.add_handler(CommandHandler("stopbot", stop_bot))
    app.add_handler(CommandHandler("restartbot", restart_bot))

    app.run_polling()


# ENTRY
if __name__ == "__main__":
    main()
