"""Metricas estadisticas de portfolio.

Funciones puras: reciben listas/series, no tocan disco ni IO.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def returns_from_equity(equity_curve: Sequence[float]) -> np.ndarray:
    """Retornos simples a partir de una curva de equity."""
    arr = np.asarray(list(equity_curve), dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    return np.diff(arr) / arr[:-1]


def sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.04,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sharpe anualizado.

    Convencion: returns son retornos por periodo (ej. diarios). Se descuenta
    la tasa libre de riesgo prorrateada y se anualiza con sqrt(periods).
    Devuelve 0.0 cuando la volatilidad es cero o hay menos de 2 puntos.
    """
    r = np.asarray(list(returns), dtype=float)
    if r.size < 2:
        return 0.0
    excess = r - (risk_free_rate / periods_per_year)
    std = np.std(excess, ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Maximo drawdown (positivo, ej 0.15 = 15%)."""
    arr = np.asarray(list(equity_curve), dtype=float)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    dd = (peaks - arr) / peaks
    return float(np.max(dd))


def cagr(equity_curve: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    arr = np.asarray(list(equity_curve), dtype=float)
    if arr.size < 2 or arr[0] <= 0:
        return 0.0
    total_return = arr[-1] / arr[0]
    years = (arr.size - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def summarize(equity_curve: Sequence[float]) -> dict[str, float]:
    rets = returns_from_equity(equity_curve)
    return {
        "sharpe_ratio": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(equity_curve),
        "cagr": cagr(equity_curve),
        "total_return": float(equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) >= 2 else 0.0,
        "n_periods": int(len(equity_curve)),
    }
