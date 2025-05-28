from core.trade_monitor import monitor_trade

# Sample open trade
open_position = {
    "signal": "BUY",
    "entry_price": 100.0,
    "sl": 95.0,
    "tp": 112.0,
    "size": 1.5,
    "strategy": "Breakout Sniper",
    "confidence": 0.91
}

# Simulate price movement loop
test_prices = [101.2, 105.5, 110.0, 112.5, 113.2]

print("📡 Simulating live price feed...\n")
for price in test_prices:
    print(f"Current Price: {price}")
    closed = monitor_trade(open_position, price)
    if closed:
        print("📉 Trade Closed. Exiting loop.\n")
        break
    else:
        print("⏳ Trade still open.\n")
