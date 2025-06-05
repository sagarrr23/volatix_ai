import os
from dotenv import load_dotenv

# Load environment variables from .env file located at project root
load_dotenv()

# === Main Configuration Class ===
class BotConfig:
    # Exchange Settings
    EXCHANGE_API_KEY     = os.getenv("EXCHANGE_API_KEY")
    EXCHANGE_SECRET_KEY  = os.getenv("EXCHANGE_SECRET_KEY")
    TRADE_SYMBOL         = os.getenv("TRADE_SYMBOL", "BTC/USDT")
    TRADE_EXCHANGE       = os.getenv("TRADE_EXCHANGE", "kraken")
    TRADE_MODE           = os.getenv("TRADE_MODE", "backtest")  # or "live"

    # Wallet Settings
    START_BALANCE        = float(os.getenv("WALLET_START_BALANCE", 300))
    MAX_TRADE_PERCENT    = float(os.getenv("MAX_TRADE_PERCENT", 0.05))  # 5%
    DAILY_LOSS_LIMIT     = float(os.getenv("DAILY_LOSS_LIMIT", 0.06))   # 6%

    # Telegram Bot Integration
    TELEGRAM_TOKEN       = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")
    TELEGRAM_BOT_NAME    = os.getenv("TELEGRAM_BOT_NAME", "VolatixBot")

    # UI Dashboard Security
    DASHBOARD_USER       = os.getenv("DASHBOARD_USER", "admin")
    DASHBOARD_PASS       = os.getenv("DASHBOARD_PASS", "volatix123")

    # TFT Model Settings
    TFT_FEATURE_DIM      = int(os.getenv("TFT_FEATURE_DIM", 60))
    SEQ_LEN              = int(os.getenv("SEQ_LEN", 20))
    FUTURE_STEPS         = [1, 3, 6]
    MODEL_PATH           = os.getenv("MODEL_PATH", "models/best_tft_model.pth")


# === Expose as Module-Level Constants ===
TFT_FEATURE_DIM = BotConfig.TFT_FEATURE_DIM
SEQ_LEN         = BotConfig.SEQ_LEN
FUTURE_STEPS    = BotConfig.FUTURE_STEPS
MODEL_PATH      = BotConfig.MODEL_PATH
