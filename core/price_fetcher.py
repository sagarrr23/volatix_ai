# core/price_fetcher.py

import os
import sys
import time
import ccxt

# Add root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config_loader import BotConfig

# === Exchange Setup ===
def get_exchange():
    exchange_id = BotConfig.TRADE_EXCHANGE
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({
        'apiKey': BotConfig.EXCHANGE_API_KEY,
        'secret': BotConfig.EXCHANGE_SECRET_KEY,
        'enableRateLimit': True,
    })

# === Kraken-Compatible Timeframe Mapping (if needed)
TIMEFRAME_MAP = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "1440"
}

# === Fetch Function (Universal)
def fetch_ohlcv(symbol=BotConfig.TRADE_SYMBOL, timeframe="5m", limit=100):
    exchange = get_exchange()

    try:
        exchange.load_markets()
    except Exception as e:
        print(f"[ERROR] Could not load markets: {e}")
        return []

    try:
        since = exchange.milliseconds() - limit * 60 * 1000  # 60 seconds * 1000ms
        print(f"[DEBUG] Fetching {symbol} at {timeframe}")
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    except Exception as e:
        print(f"[ERROR] Failed to fetch candles for {symbol} ({timeframe}): {e}")
        return []



# === Multi-Timeframe Helpers ===
def fetch_1m(symbol=BotConfig.TRADE_SYMBOL): return fetch_ohlcv(symbol, "1m")
def fetch_5m(symbol=BotConfig.TRADE_SYMBOL): return fetch_ohlcv(symbol, "5m")
def fetch_15m(symbol=BotConfig.TRADE_SYMBOL): return fetch_ohlcv(symbol, "15m")
def fetch_1h(symbol=BotConfig.TRADE_SYMBOL): return fetch_ohlcv(symbol, "1h")

# === Debug Mode ===
if __name__ == "__main__":
    candles = fetch_5m()
    if candles:
        print("\n📊 Last 5 Candles (5m OHLCV):")
        for c in candles[-5:]:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c[0] / 1000))
            print(f"Time: {ts} | O: {c[1]} | H: {c[2]} | L: {c[3]} | C: {c[4]} | V: {c[5]}")
    else:
        print("❌ No candle data fetched.")
