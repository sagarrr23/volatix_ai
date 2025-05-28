import os, io, zipfile, requests, pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

# ── CONFIG ──────────────────────────────────────────
SYM      = "BTCUSDT"          # change to ETHUSDT for ETH
TF       = "1m"               # 1-minute bars
START    = "2017-08-17"       # first Binance trading day
DST_DIR  = "data_raw"         # where zip/CSV will live
BASE_URL = ("https://data.binance.vision/data/spot/daily/klines/"
            "{sym}/{tf}/{sym}-{tf}-{date}.zip")
# ────────────────────────────────────────────────────

os.makedirs(DST_DIR, exist_ok=True)
start_dt  = datetime.strptime(START, "%Y-%m-%d").date()
today_dt  = datetime.utcnow().date()

def daterange(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def fetch_one(day):
    url  = BASE_URL.format(sym=SYM, tf=TF, date=day)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return None            # day missing (exchange maintenance)
    zf   = zipfile.ZipFile(io.BytesIO(resp.content))
    csv  = zf.namelist()[0]
    zf.extract(csv, DST_DIR)   # extract into raw folder
    return csv

print(f"⬇️  Downloading {SYM} {TF} from {START} to {today_dt} …")
for d in tqdm(list(daterange(start_dt, today_dt))):
    ymd = d.strftime("%Y-%m-%d")
    out = f"{SYM}-{TF}-{ymd}.csv"
    if os.path.exists(f"{DST_DIR}/{out}"):
        continue               # already downloaded
    fetch_one(ymd)

print("✅ All available ZIPs downloaded & extracted.")
