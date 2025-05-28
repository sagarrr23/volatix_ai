import os
import logging
from datetime import datetime

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Format log filename by date
log_filename = os.path.join(LOG_DIR, f"volatix_{datetime.now().strftime('%Y-%m-%d')}.log")

# Create logger
logger = logging.getLogger("VolatixAI")
logger.setLevel(logging.DEBUG)

# === FILE HANDLER ===
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_format)

# === CONSOLE HANDLER ===
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_format)

# Add both handlers (once only)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Test block
if __name__ == "__main__":
    try:
        logger.info("🟢 Logger initialized successfully!")
        logger.warning("⚠️ Example warning message.")
        logger.error("❌ Example error message.")
        logger.critical("🚨 Critical system failure simulation.")
    except UnicodeEncodeError:
        logger.info("Logger initialized (no emoji fallback).")
