import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    """
    Sends a formatted text message to the configured Telegram chat.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram credentials missing.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("❌ Telegram API error:", response.text)
        return response.status_code == 200
    except Exception as e:
        print("❌ Telegram send failed:", e)
        return False
