from __future__ import annotations

import pandas as pd

from utils.indicators import add_indicators, support_resistance


def analyze_technicals(df: pd.DataFrame) -> dict:
    enriched = add_indicators(df)
    latest = enriched.iloc[-1]

    price = float(latest["Close"])
    sma50 = latest["SMA_50"]
    sma200 = latest["SMA_200"]
    rsi_value = float(latest["RSI_14"])
    macd_value = float(latest["MACD"])
    signal = float(latest["MACD_Signal"])

    if pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 and price > sma200:
            long_term = "Price is above the 50-day and 200-day SMA."
        elif price < sma50 and price < sma200:
            long_term = "Price is below the 50-day and 200-day SMA."
        else:
            long_term = "Price is mixed relative to the 50-day and 200-day SMA."
    else:
        long_term = "Not enough history for the 50/200-day trend comparison."

    if rsi_value > 70:
        momentum = "RSI is above 70, which can indicate potentially overbought conditions."
    elif rsi_value < 30:
        momentum = "RSI is below 30, which can indicate potentially oversold conditions."
    else:
        momentum = "RSI is between 30 and 70."

    if macd_value > signal:
        macd_interpretation = "MACD is above its signal line."
    else:
        macd_interpretation = "MACD is below its signal line."

    support, resistance = support_resistance(enriched)

    return {
        "data": enriched,
        "price": price,
        "rsi": rsi_value,
        "macd": macd_value,
        "macd_signal": signal,
        "sma20": latest["SMA_20"],
        "sma50": latest["SMA_50"],
        "sma200": latest["SMA_200"],
        "ema20": latest["EMA_20"],
        "ema50": latest["EMA_50"],
        "atr": latest["ATR_14"],
        "bb_upper": latest["BB_Upper"],
        "bb_middle": latest["BB_Middle"],
        "bb_lower": latest["BB_Lower"],
        "volume": latest["Volume"],
        "volume_change_pct": latest["Volume_Change_Pct"],
        "support": support,
        "resistance": resistance,
        "long_term": long_term,
        "momentum": momentum,
        "macd_interpretation": macd_interpretation,
    }
