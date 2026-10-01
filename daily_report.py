from __future__ import annotations

import argparse
import sqlite3

from analysis.fundamental import get_fundamentals
from analysis.risk import calculate_risk
from analysis.sentiment import analyze_news_sentiment, sentiment_summary
from analysis.technical import analyze_technicals
from config import DB_PATH, NEWS_LIMIT
from database.init_db import initialize_database
from models.prediction_model import run_direction_model
from services.market_data import get_history
from services.news import fetch_news
from services.report_generator import generate_markdown_report, save_report


def save_news(ticker: str, articles: list[dict]):
    with sqlite3.connect(DB_PATH) as conn:
        for article in articles:
            conn.execute(
                """
                INSERT OR REPLACE INTO news
                (ticker, headline, source, published, url, sentiment, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker.upper(),
                    article.get("headline"),
                    article.get("source"),
                    article.get("published"),
                    article.get("url"),
                    article.get("sentiment"),
                    article.get("confidence"),
                ),
            )


def run(ticker: str, period: str = "1y", refresh: bool = True):
    initialize_database()

    print(f"Downloading market data for {ticker}...")
    df = get_history(ticker, period=period, refresh=refresh)

    technical = analyze_technicals(df)

    benchmark_ticker = "^NSEI" if ticker.upper().endswith(".NS") else "^GSPC"
    try:
        benchmark = get_history(benchmark_ticker, period=period, refresh=False)
    except Exception:
        benchmark = None

    risk = calculate_risk(df, benchmark)

    fundamentals = get_fundamentals(ticker)
    company_name = fundamentals.get("longName") or fundamentals.get("shortName") or ticker

    print("Fetching news...")
    try:
        articles = fetch_news(ticker, company_name, NEWS_LIMIT)
        articles = analyze_news_sentiment(articles)
    except Exception as exc:
        print(f"News unavailable: {exc}")
        articles = []

    save_news(ticker, articles)
    sentiment = sentiment_summary(articles)

    print("Running ML direction model...")
    ml_result = run_direction_model(df)

    report = generate_markdown_report(
        ticker=ticker,
        df=df,
        technical=technical,
        risk=risk,
        sentiment=sentiment,
        ml_result=ml_result,
        fundamentals=fundamentals,
    )

    path = save_report(ticker, report)

    print("\n" + report)
    print(f"\nSaved report: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a daily stock analysis report.")
    parser.add_argument("--ticker", required=True, help="Example: AAPL or RELIANCE.NS")
    parser.add_argument("--period", default="1y", help="yfinance period, e.g. 1y, 2y, 5y")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use cached market data when available.",
    )
    args = parser.parse_args()

    run(args.ticker, args.period, refresh=not args.no_refresh)
