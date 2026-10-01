from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3

from config import DB_PATH, REPORTS_DIR


def _fmt(value, digits=2, suffix=""):
    if value is None:
        return "N/A"
    try:
        if value != value:
            return "N/A"
        return f"{value:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def performance_summary(df):
    close = df["Close"]
    periods = {
        "1D": 1,
        "5D": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }
    result = {}
    for label, n in periods.items():
        if len(close) > n:
            result[label] = float((close.iloc[-1] / close.iloc[-n - 1] - 1) * 100)
        else:
            result[label] = None
    return result


def generate_markdown_report(
    ticker,
    df,
    technical,
    risk,
    sentiment,
    ml_result,
    fundamentals=None,
):
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    perf = performance_summary(df)

    report = f"""# Daily Stock Analysis

**Generated:** {today}  
**Ticker:** `{ticker}`  
**Latest market-data date:** {df.index[-1].strftime("%Y-%m-%d")}

> This report is an analytical dashboard, not financial advice. Historical
> indicators and model outputs do not guarantee future results.

## Current Price

**{_fmt(technical["price"])}**

## Price Performance

| Period | Change |
|---|---:|
| 1D | {_fmt(perf["1D"], suffix="%")} |
| 5D | {_fmt(perf["5D"], suffix="%")} |
| 1M | {_fmt(perf["1M"], suffix="%")} |
| 3M | {_fmt(perf["3M"], suffix="%")} |
| 6M | {_fmt(perf["6M"], suffix="%")} |
| 1Y | {_fmt(perf["1Y"], suffix="%")} |

## Technical Indicators

| Indicator | Value |
|---|---:|
| RSI 14 | {_fmt(technical["rsi"])} |
| MACD | {_fmt(technical["macd"])} |
| MACD Signal | {_fmt(technical["macd_signal"])} |
| SMA 20 | {_fmt(technical["sma20"])} |
| SMA 50 | {_fmt(technical["sma50"])} |
| SMA 200 | {_fmt(technical["sma200"])} |
| ATR 14 | {_fmt(technical["atr"])} |
| Bollinger Upper | {_fmt(technical["bb_upper"])} |
| Bollinger Lower | {_fmt(technical["bb_lower"])} |
| Support (recent) | {_fmt(technical["support"])} |
| Resistance (recent) | {_fmt(technical["resistance"])} |

### Technical Interpretation

- Long-term trend: {technical["long_term"]}
- Momentum: {technical["momentum"]}
- MACD: {technical["macd_interpretation"]}

## Risk

| Metric | Value |
|---|---:|
| Annualized volatility | {_fmt(risk["annualized_volatility"] * 100 if risk["annualized_volatility"] == risk["annualized_volatility"] else None, suffix="%")} |
| Average daily return | {_fmt(risk["average_daily_return"] * 100 if risk["average_daily_return"] == risk["average_daily_return"] else None, suffix="%")} |
| Maximum drawdown | {_fmt(risk["max_drawdown"] * 100 if risk["max_drawdown"] == risk["max_drawdown"] else None, suffix="%")} |
| Sharpe ratio | {_fmt(risk["sharpe"])} |
| Recent annualized volatility | {_fmt(risk["recent_volatility"] * 100 if risk["recent_volatility"] == risk["recent_volatility"] else None, suffix="%")} |
| Beta | {_fmt(risk["beta"])} |

## News Sentiment

- Positive: {_fmt(sentiment.get("positive"), suffix="%")}
- Neutral: {_fmt(sentiment.get("neutral"), suffix="%")}
- Negative: {_fmt(sentiment.get("negative"), suffix="%")}
- Articles analyzed: {sentiment.get("total", 0)}

## ML Direction Model

"""

    if ml_result.get("available"):
        report += f"""- Model: {ml_result["model_name"]}
- Historical test classification: **{ml_result["next_direction"]}**
- Model probability/confidence: {_fmt((ml_result["confidence"] or 0) * 100, suffix="%") if ml_result["confidence"] is not None else "N/A"}
- Accuracy: {_fmt(ml_result["accuracy"] * 100, suffix="%")}
- Precision: {_fmt(ml_result["precision"] * 100, suffix="%")}
- Recall: {_fmt(ml_result["recall"] * 100, suffix="%")}
- F1: {_fmt(ml_result["f1"] * 100, suffix="%")}

**Important:** {ml_result["warning"]}
"""
    else:
        report += f'- Model unavailable: {ml_result.get("message", "Unknown reason.")}\n'

    report += """
## Key Observations

1. Review the price trend together with the 20/50/200-day moving averages.
2. Treat RSI, MACD and Bollinger Bands as indicators requiring context, not standalone decisions.
3. Compare news sentiment with the actual news headlines and source dates.

## Risks / Uncertainties

1. Market conditions can change rapidly.
2. News sentiment models can misclassify financial language.
3. Historical backtests and classification metrics do not establish future performance.
4. Market-data providers may have delays, outages, or licensing limitations.

## Data Classification

- **Observed facts:** market prices, volumes, dates and retrieved headlines.
- **Calculated indicators:** technical and risk metrics derived from observed data.
- **Model outputs:** ML direction classification and its historical test metrics.
- **Interpretations:** explanatory text derived from the indicators.

"""

    if fundamentals:
        report += "## Selected Fundamentals\n\n"
        for key, value in fundamentals.items():
            if value is not None:
                report += f"- **{key}:** {value}\n"

    return report


def save_report(ticker: str, markdown: str) -> Path:
    path = REPORTS_DIR / f"{datetime.now():%Y-%m-%d}_{ticker.upper().replace('/', '_')}.md"
    path.write_text(markdown, encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                report_path TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO daily_reports (ticker, generated_at, report_path) VALUES (?, ?, ?)",
            (ticker.upper(), datetime.now().isoformat(), str(path)),
        )

    return path
