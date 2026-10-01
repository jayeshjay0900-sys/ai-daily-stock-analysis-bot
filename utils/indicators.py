import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def macd(series: pd.Series):
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    histogram = line - signal
    return line, signal, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2):
    middle = sma(series, window)
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["Close"]

    result["SMA_20"] = sma(close, 20)
    result["SMA_50"] = sma(close, 50)
    result["SMA_200"] = sma(close, 200)
    result["EMA_20"] = ema(close, 20)
    result["EMA_50"] = ema(close, 50)
    result["RSI_14"] = rsi(close)

    result["MACD"], result["MACD_Signal"], result["MACD_Hist"] = macd(close)
    (
        result["BB_Upper"],
        result["BB_Middle"],
        result["BB_Lower"],
    ) = bollinger_bands(close)
    result["ATR_14"] = atr(result)

    result["Volume_SMA_20"] = result["Volume"].rolling(20).mean()
    result["Volume_Change_Pct"] = result["Volume"].pct_change() * 100
    result["Daily_Return"] = close.pct_change()

    return result


def support_resistance(df: pd.DataFrame, lookback: int = 60):
    recent = df.tail(lookback)
    if recent.empty:
        return None, None
    return float(recent["Low"].min()), float(recent["High"].max())
