"""Genera fixtures CSV deterministas para AAPL y SPY.

Camino aleatorio con tendencia + ruido, semilla fija. No usa APIs externas.
Se ejecuta una sola vez para producir data/fixtures/*.csv reproducibles.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _business_days(start: date, n: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def generate(symbol: str, start_price: float, drift: float, vol: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = _business_days(date(2024, 1, 2), n)
    log_returns = rng.normal(loc=drift, scale=vol, size=n)
    # Inyectar un par de regimenes para forzar cruces y SELL signals.
    log_returns[60:90] -= vol * 1.8
    log_returns[120:150] += vol * 1.5

    closes = start_price * np.exp(np.cumsum(log_returns))
    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.01, size=n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.01, size=n))
    volumes = rng.integers(1_000_000, 5_000_000, size=n)

    return pd.DataFrame(
        {
            "date": [d.isoformat() for d in days],
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(closes, 4),
            "volume": volumes,
        }
    )


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    aapl = generate("AAPL", start_price=180.0, drift=0.0006, vol=0.018, n=200, seed=11)
    spy = generate("SPY", start_price=470.0, drift=0.0004, vol=0.012, n=200, seed=23)

    aapl.to_csv(out / "AAPL.csv", index=False)
    spy.to_csv(out / "SPY.csv", index=False)
    print(f"Fixtures escritos en {out}")


if __name__ == "__main__":
    main()
