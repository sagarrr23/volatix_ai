import sys
import os

# Add the project root (one level up from 'test') to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.breakout_sniper import breakout_sniper_signal
from strategies.reversal_sniper import reversal_sniper_signal
from strategies.trend_rider import trend_rider_signal


# Load example 1H BTC/USDT CSV (your own file)
df = pd.read_csv("historical_data/BTCUSDT_1h.csv")
df.columns = [c.lower() for c in df.columns]

print(breakout_sniper_signal(df.tail(100)))
print(reversal_sniper_signal(df.tail(100)))
print(trend_rider_signal(df.tail(100)))
