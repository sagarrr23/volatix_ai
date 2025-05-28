import os
import json
from datetime import datetime

TRADE_LOG_PATH = "data/live_trades.json"
MAX_ACTIVE_TRADES = 3  # 🔒 Adjustable limit

def get_active_trades():
    if os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "r") as f:
            try:
                trades = json.load(f)
                return [t for t in trades if t["status"] == "OPEN"]
            except:
                return []
    return []

def can_open_new_trade():
    active = get_active_trades()
    return len(active) < MAX_ACTIVE_TRADES
