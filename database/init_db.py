from pathlib import Path
import sqlite3

from config import DB_PATH


def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stocks (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS market_data (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );

            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                headline TEXT NOT NULL,
                source TEXT,
                published TEXT,
                url TEXT,
                sentiment TEXT,
                confidence REAL,
                UNIQUE(ticker, headline, published)
            );

            CREATE TABLE IF NOT EXISTS technical_indicators (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                sma20 REAL,
                sma50 REAL,
                sma200 REAL,
                PRIMARY KEY(ticker, date)
            );

            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                report_path TEXT NOT NULL
            );
            """
        )


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {Path(DB_PATH).resolve()}")
