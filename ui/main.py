import os
import sys
import json
from datetime import datetime
from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_401_UNAUTHORIZED
from dotenv import load_dotenv

# Path fix to access core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.wallet_manager import WalletManager
from core.trade_logger import TradeLogger
from core.config_loader import BotConfig

# ========== SETUP ==========
load_dotenv()
DASHBOARD_USER = os.getenv("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.getenv("DASHBOARD_PASS", "volatix123")

app = FastAPI()

# Mount static folder for CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="ui/templates")
templates.env.globals["now"] = datetime.now
templates.env.globals["config"] = BotConfig

# Files
USER_PROFILE_FILE = "logs/user_profile.json"
CSV_EXPORT_PATH = "logs/trades.csv"
JSON_EXPORT_PATH = "logs/trades.json"

# Static AI Diagnostic (Mock)
AI_INFO = {
    "lstm_accuracy": "84.6%",
    "lstm_confidence": "0.89",
    "lstm_latency": "148ms",
    "rl_mode": "active",
    "rl_avg_reward": "+0.33",
    "rl_decision_rate": "1.8/sec"
}

# ========== AUTH ==========
def authenticate(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    if credentials.username != DASHBOARD_USER or credentials.password != DASHBOARD_PASS:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return credentials.username

# ========== PROFILE STORAGE ==========
def load_user_profile():
    if os.path.exists(USER_PROFILE_FILE):
        try:
            with open(USER_PROFILE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_profile(data):
    os.makedirs("logs", exist_ok=True)
    with open(USER_PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ========== ROUTES ==========

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")

# Dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(authenticate)):
    wallet = WalletManager()
    logger = TradeLogger()
    profile = load_user_profile()

    trades = logger.load_last_n(5)
    current_trade = logger.load_current_trade()
    strategy = "N/A"
    try:
        with open("logs/strategy_status.txt", "r") as f:
            strategy = f.read().strip()
    except:
        pass

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "balance": wallet.get_balance(),
        "daily_loss": wallet.get_daily_loss(),
        "max_risk": wallet.get_trade_amount(),
        "trades": trades,
        "current_trade": current_trade,
        "current_strategy": strategy,
        "telegram_bot": getattr(BotConfig, "TELEGRAM_BOT_NAME", "volatixbot"),
        "telegram_chat": getattr(BotConfig, "TELEGRAM_CHAT_ID", "N/A"),
        "profile": profile,
        "can_trade": wallet.can_trade_today(),
        "ai": AI_INFO
    })

# Wallet Page
@app.get("/wallet", response_class=HTMLResponse)
async def wallet_page(request: Request, user=Depends(authenticate)):
    wallet = WalletManager()
    return templates.TemplateResponse("wallet.html", {
        "request": request,
        "balance": wallet.get_balance(),
        "daily_loss": wallet.get_daily_loss(),
        "max_trade": wallet.get_trade_amount(),
        "can_trade": wallet.can_trade_today()
    })

@app.post("/wallet", response_class=HTMLResponse)
async def wallet_actions(request: Request, action: str = Form(...), user=Depends(authenticate)):
    wallet = WalletManager()
    if action == "reset":
        wallet.reset_daily_loss()
    elif action == "simulate_gain":
        wallet.apply_trade_result(+25)
    elif action == "simulate_loss":
        wallet.apply_trade_result(-50)
    return RedirectResponse(url="/wallet", status_code=302)

# Trades Page
@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request, user=Depends(authenticate)):
    logger = TradeLogger()
    trades = logger.load_all()
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["pnl"] > 0])
    losing_trades = len([t for t in trades if t["pnl"] < 0])
    net_pnl = round(sum(t["pnl"] for t in trades), 2)
    return templates.TemplateResponse("trades.html", {
        "request": request,
        "trades": trades,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "net_pnl": net_pnl
    })

# Downloads
@app.get("/trades/download/csv", response_class=FileResponse)
async def download_csv(user=Depends(authenticate)):
    if os.path.exists(CSV_EXPORT_PATH):
        return FileResponse(path=CSV_EXPORT_PATH, filename="volatix_trades.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="CSV export not found")

@app.get("/trades/download/json", response_class=FileResponse)
async def download_json(user=Depends(authenticate)):
    if os.path.exists(JSON_EXPORT_PATH):
        return FileResponse(path=JSON_EXPORT_PATH, filename="volatix_trades.json", media_type="application/json")
    raise HTTPException(status_code=404, detail="JSON export not found")

# Settings
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user=Depends(authenticate)):
    profile = load_user_profile()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "strategy": open("logs/strategy_status.txt").read().strip() if os.path.exists("logs/strategy_status.txt") else "N/A",
        "telegram_bot": getattr(BotConfig, "TELEGRAM_BOT_NAME", "volatixbot"),
        "telegram_chat": getattr(BotConfig, "TELEGRAM_CHAT_ID", "N/A"),
        "profile": profile,
        "ai": AI_INFO
    })

@app.post("/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    name: str = Form(""),
    email: str = Form(""),
    telegram: str = Form(""),
    org: str = Form(""),
    tagline: str = Form(""),
    max_risk: float = Form(...),
    loss_limit: float = Form(...),
    user=Depends(authenticate)
):
    BotConfig.MAX_TRADE_PERCENT = max_risk
    BotConfig.DAILY_LOSS_LIMIT = loss_limit

    profile_data = {
        "name": name.strip(),
        "email": email.strip(),
        "telegram": telegram.strip(),
        "organization": org.strip(),
        "tagline": tagline.strip()
    }
    save_user_profile(profile_data)

    return RedirectResponse(url="/settings", status_code=302)
