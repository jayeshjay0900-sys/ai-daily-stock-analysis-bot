from __future__ import annotations

import math
import yfinance as yf


def get_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return {}

    fields = [
        "shortName",
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "returnOnEquity",
        "profitMargins",
        "revenueGrowth",
        "earningsGrowth",
    ]

    result = {field: info.get(field) for field in fields}

    # Normalize NaN-like values for JSON/UI friendliness.
    for key, value in result.items():
        if isinstance(value, float) and math.isnan(value):
            result[key] = None

    return result
