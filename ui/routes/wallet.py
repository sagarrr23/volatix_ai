import os
import sys
import json
from datetime import datetime
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_303_SEE_OTHER
from dotenv import load_dotenv

# Import fixes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from core.wallet_manager import WalletManager
from core.auth import authenticate

# Setup
router = APIRouter()
templates = Jinja2Templates(directory="ui/templates")
load_dotenv()

WALLET_LOG_FILE = "logs/wallet_actions.json"

# Load recent wallet action logs
def load_action_log(limit=10):
    if not os.path.exists(WALLET_LOG_FILE):
        return []
    try:
        with open(WALLET_LOG_FILE, "r") as f:
            data = json.load(f)
            return data[-limit:]
    except:
        return []

# Save action log entry
def log_wallet_action(action: str):
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action
    }
    if os.path.exists(WALLET_LOG_FILE):
        try:
            with open(WALLET_LOG_FILE, "r") as f:
                logs = json.load(f)
        except:
            logs = []
    else:
        logs = []
    logs.append(log_entry)
    with open(WALLET_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# Simple AI-based wallet suggestion
def get_ai_suggestion(wallet: WalletManager):
    if not wallet.can_trade_today():
        return "You’ve reached the daily risk limit. Consider pausing trades."
    elif wallet.get_daily_loss() > 0.5 * wallet.get_trade_amount():
        return "Your drawdown is approaching 50% of your limit. Monitor volatility."
    elif wallet.get_trade_amount() < 10:
        return "Trade risk is set too low. Increase to capture profit spikes."
    else:
        return "All systems nominal. Monitor market volume."

# === GET Wallet Page ===
@router.get("/wallet", response_class=HTMLResponse)
async def get_wallet(request: Request, user: str = Depends(authenticate)):
    wallet = WalletManager()

    return templates.TemplateResponse("wallet.html", {
        "request": request,
        "balance": wallet.get_balance(),
        "daily_loss": wallet.get_daily_loss(),
        "max_trade": wallet.get_trade_amount(),
        "can_trade": wallet.can_trade_today(),
        "trading_pool": wallet.trading_pool,
        "profit_pool": wallet.accumulated_profit,
        "suggestion": get_ai_suggestion(wallet),
        "action_log": load_action_log()
    })

# === POST Wallet Actions ===
@router.post("/wallet", response_class=HTMLResponse)
async def wallet_action(request: Request, action: str = Form(...), user: str = Depends(authenticate)):
    wallet = WalletManager()

    if action == "reset":
        wallet.reset_daily_loss()
        log_wallet_action("🧹 Reset Daily Loss")

    elif action == "simulate_gain":
        wallet.apply_trade_result(+25.0)
        log_wallet_action("📈 Simulated +$25 Gain")

    elif action == "simulate_loss":
        wallet.apply_trade_result(-50.0)
        log_wallet_action("📉 Simulated -$50 Loss")

    elif action == "withdraw_profits":
        amount = wallet.reset_accumulated_profit()
        log_wallet_action(f"💸 Withdrew ${amount} Profits")

    return RedirectResponse(url="/wallet", status_code=HTTP_303_SEE_OTHER)
