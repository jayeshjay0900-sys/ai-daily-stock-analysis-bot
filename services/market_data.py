from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd
import yfinance as yf

from config import DB_PATH


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    df = _clean_columns(df)

    if df.empty:
        raise ValueError(
            f"No historical data was returned for {ticker}. "
            "Check the ticker symbol and market."
        )

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing market-data columns: {missing}")

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(
        subset=["Close"]
    )


def get_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).fast_info
        return dict(info)
    except Exception:
        return {}


def save_market_data(ticker: str, df: pd.DataFrame) -> None:
    ticker = ticker.upper()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
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
            )
            """
        )
        rows = []
        for idx, row in df.iterrows():
            rows.append(
                (
                    ticker,
                    pd.Timestamp(idx).strftime("%Y-%m-%d"),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row["Adj Close"]),
                    float(row["Volume"]),
                )
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO market_data
            (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def load_market_data(ticker: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT date, open AS Open, high AS High, low AS Low,
                   close AS Close, adj_close AS "Adj Close", volume AS Volume
            FROM market_data
            WHERE ticker = ?
            ORDER BY date
            """,
            conn,
            params=[ticker.upper()],
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def get_history(ticker: str, period: str = "1y", refresh: bool = False) -> pd.DataFrame:
    if not refresh:
        cached = load_market_data(ticker)
        if not cached.empty:
            # Refresh if cached data is older than today's date.
            last_date = cached.index.max().date()
            today = pd.Timestamp.now().date()
            if last_date >= today:
                return cached

    fresh = download_history(ticker, period)
    save_market_data(ticker, fresh)
    return fresh
