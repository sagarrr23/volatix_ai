import os
import sys
import json
import datetime

# Import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config_loader import BotConfig


class WalletManager:
    def __init__(self, wallet_file="wallet.json"):
        self.wallet_file = wallet_file
        self.max_daily_loss_percent = BotConfig.DAILY_LOSS_LIMIT
        self.max_trade_percent = BotConfig.MAX_TRADE_PERCENT
        self.wallet = self._load_wallet()
        self._check_daily_reset()

    def _load_wallet(self):
        if os.path.exists(self.wallet_file):
            with open(self.wallet_file, "r") as f:
                data = json.load(f)
                # Patch for backward compatibility
                data.setdefault("daily_loss", 0.0)
                data.setdefault("start_of_day", data.get("balance", BotConfig.START_BALANCE))
                data.setdefault("last_reset", str(datetime.date.today()))
                data.setdefault("accumulated_profit", 0.0)
                data.setdefault("trading_pool", data.get("balance", BotConfig.START_BALANCE))
                return data
        else:
            return {
                "balance": BotConfig.START_BALANCE,
                "daily_loss": 0.0,
                "start_of_day": BotConfig.START_BALANCE,
                "last_reset": str(datetime.date.today()),
                "accumulated_profit": 0.0,
                "trading_pool": BotConfig.START_BALANCE
            }

    def _save_wallet(self):
        with open(self.wallet_file, "w") as f:
            json.dump(self.wallet, f, indent=2)

    def _check_daily_reset(self):
        today = str(datetime.date.today())
        if self.wallet["last_reset"] != today:
            self.wallet["daily_loss"] = 0.0
            self.wallet["start_of_day"] = self.wallet["balance"]
            self.wallet["last_reset"] = today
            self._save_wallet()

    def get_trade_amount(self):
        return round(self.wallet["trading_pool"] * self.max_trade_percent, 2)

    def get_smart_trade_size(self, entry_price, stop_loss_price, risk_percent=0.015):
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            return 0.0
        risk_amount = self.wallet["trading_pool"] * risk_percent
        size = risk_amount / stop_distance
        return round(size, 4)

    def apply_trade_result(self, profit_or_loss):
        self.wallet["trading_pool"] += profit_or_loss
        self.wallet["balance"] += profit_or_loss

        if profit_or_loss > 0:
            self.wallet["accumulated_profit"] += profit_or_loss
        elif profit_or_loss < 0:
            self.wallet["daily_loss"] += abs(profit_or_loss)

        self._save_wallet()

    def reset_daily_loss(self):
        self.wallet["daily_loss"] = 0.0
        self.wallet["start_of_day"] = self.wallet["balance"]
        self.wallet["last_reset"] = str(datetime.date.today())
        self._save_wallet()

    def reset_accumulated_profit(self):
        profit = self.wallet["accumulated_profit"]
        self.wallet["accumulated_profit"] = 0.0
        self._save_wallet()
        return round(profit, 2)

    def can_trade_today(self):
        max_loss = self.wallet["start_of_day"] * self.max_daily_loss_percent
        return self.wallet["daily_loss"] < max_loss

    def get_balance(self):
        return round(self.wallet["balance"], 2)

    def get_daily_loss(self):
        return round(self.wallet["daily_loss"], 2)

    @property
    def trading_pool(self):
        return round(self.wallet.get("trading_pool", 0.0), 2)

    @property
    def accumulated_profit(self):
        return round(self.wallet.get("accumulated_profit", 0.0), 2)


# Manual test
if __name__ == "__main__":
    wallet = WalletManager()
    print("──────────── WALLET STATUS ────────────")
    print(f"👜 Total Balance: ${wallet.get_balance()}")
    print(f"💼 Trading Pool: ${wallet.trading_pool}")
    print(f"📦 Accumulated Profit: ${wallet.accumulated_profit}")
    print(f"📉 Daily Loss: ${wallet.get_daily_loss()}")
    print(f"📊 Max Trade Size: ${wallet.get_trade_amount()}")
    print("───────────────────────────────────────")

    # Simulate a win
    wallet.apply_trade_result(+30)
    print(f"✅ Balance after win: ${wallet.get_balance()}")

    # Simulate a loss
    wallet.apply_trade_result(-20)
    print(f"❌ Balance after loss: ${wallet.get_balance()}")
    print("🚦 Can Trade Today?", wallet.can_trade_today())

    # Withdraw profits
    withdrawn = wallet.reset_accumulated_profit()
    print(f"💸 Withdrawn Profit: ${withdrawn}")
