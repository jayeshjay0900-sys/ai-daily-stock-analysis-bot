from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_risk(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict:
    returns = df["Close"].pct_change().dropna()

    annualized_volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else np.nan
    average_daily_return = float(returns.mean()) if not returns.empty else np.nan

    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    sharpe = np.nan
    if len(returns) > 1 and returns.std() != 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))

    recent = returns.tail(20)
    recent_volatility = (
        float(recent.std() * np.sqrt(252)) if len(recent) > 1 else np.nan
    )

    beta = np.nan
    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat(
            [returns.rename("stock"), benchmark["Close"].pct_change().rename("benchmark")],
            axis=1,
        ).dropna()
        if len(aligned) > 2 and aligned["benchmark"].var() != 0:
            beta = float(aligned["stock"].cov(aligned["benchmark"]) / aligned["benchmark"].var())

    return {
        "annualized_volatility": annualized_volatility,
        "average_daily_return": average_daily_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "recent_volatility": recent_volatility,
        "beta": beta,
    }
