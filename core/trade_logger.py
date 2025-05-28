import os
import sys
import json
import csv
from datetime import datetime, timezone

# Allow imports from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.logger import logger  # Emoji-style logger

# Constants
TRADE_LOG_DIR = "logs"
CSV_FILE = os.path.join(TRADE_LOG_DIR, "trades.csv")
JSON_FILE = os.path.join(TRADE_LOG_DIR, "trades.json")
CURRENT_TRADE_FILE = os.path.join(TRADE_LOG_DIR, "current_trade.json")

class TradeLogger:
    def __init__(self):
        os.makedirs(TRADE_LOG_DIR, exist_ok=True)
        self._init_csv()
        self._init_json()

    def _init_csv(self):
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "side", "quantity",
                    "entry_price", "exit_price", "pnl",
                    "strategy", "confidence", "mode"
                ])

    def _init_json(self):
        if not os.path.exists(JSON_FILE):
            with open(JSON_FILE, "w") as f:
                json.dump([], f, indent=2)

    def log_trade(self, symbol, side, qty, entry_price, exit_price, pnl, strategy, confidence, mode="live"):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "strategy": strategy,
            "confidence": round(confidence, 2),
            "mode": mode
        }

        logger.info(
            f"📈 Trade Logged | {side.upper()} {qty} {symbol} | "
            f"Entry: {entry_price} → Exit: {exit_price} | "
            f"PnL: {pnl:.2f} | Strategy: {strategy} | Confidence: {confidence:.2f}"
        )

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, symbol, side.upper(), qty,
                entry_price, exit_price, round(pnl, 2),
                strategy, round(confidence, 2), mode
            ])

        with open(JSON_FILE, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(trade)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)

        # 🧠 Clear current trade after it's finalized
        if os.path.exists(CURRENT_TRADE_FILE):
            os.remove(CURRENT_TRADE_FILE)

    def save_current_trade(self, signal: dict):
        with open(CURRENT_TRADE_FILE, "w", encoding="utf-8") as f:
            json.dump(signal, f, indent=2)

    def load_current_trade(self):
        if os.path.exists(CURRENT_TRADE_FILE):
            with open(CURRENT_TRADE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def load_last_n(self, n=5):
        if not os.path.exists(JSON_FILE):
            return []
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data[-n:] if len(data) >= n else data
            except json.JSONDecodeError:
                return []

    def load_all(self):
        """✅ Returns full trade history as list"""
        if not os.path.exists(JSON_FILE):
            return []
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

# Optional CLI test
if __name__ == "__main__":
    logger = TradeLogger()
    logger.log_trade(
        symbol="BTC/USDT",
        side="buy",
        qty=0.01,
        entry_price=30000,
        exit_price=30750,
        pnl=75.0,
        strategy="Breakout Sniper",
        confidence=0.91,
        mode="backtest"
    )
