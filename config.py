from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATABASE_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATABASE_DIR / "market.db"

DEFAULT_PERIOD = "1y"
NEWS_LIMIT = 10
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.0"))

# Optional. yfinance works without these.
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
