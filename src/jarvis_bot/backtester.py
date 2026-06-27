"""Backtester con walk-forward validation.

Flujo:
    load fixtures → enrich indicators → split ventanas walk-forward
    → para cada ventana: simular con PaperBroker → calcular métricas
    → comparar estrategias → reportar la mejor.

Walk-forward:
    Divide la serie histórica en N ventanas solapadas.
    Cada ventana: train_pct (ej. 70%) para "ajustar" parámetros visualmente,
    test_pct (ej. 30%) para evaluar out-of-sample.
    Sólo las métricas del período TEST cuentan para la evaluación.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import stats_engine
from .brokers import Order, OrderSide, PaperBroker
from .indicators import enrich
from .modules.risk_manager import RiskManager
from .strategies import Strategy
from .strategies.base import Signal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado de una sola simulación
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    strategy_name: str
    window_id: int
    period_start: datetime
    period_end: datetime
    n_trades: int
    final_equity: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    cagr: float
    win_rate: float
    equity_curve: list[float] = field(default_factory=list, repr=False)
    trades: list[dict] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        return (
            f"[{self.strategy_name} | W{self.window_id}] "
            f"return={self.total_return:+.2%} sharpe={self.sharpe_ratio:.2f} "
            f"dd={self.max_drawdown:.2%} trades={self.n_trades} wr={self.win_rate:.2%}"
        )


@dataclass
class BacktestReport:
    results: list[SimResult]
    best_strategy: str
    avg_sharpe_by_strategy: dict[str, float]
    avg_return_by_strategy: dict[str, float]

    def print_summary(self) -> None:
        print("\n=== Backtest Walk-Forward Report ===")
        for name, sharpe in sorted(self.avg_sharpe_by_strategy.items(), key=lambda x: -x[1]):
            ret = self.avg_return_by_strategy[name]
            print(f"  {name:20s}  avg_sharpe={sharpe:+.3f}  avg_return={ret:+.2%}")
        print(f"\n  Mejor estrategia: {self.best_strategy}")
        print("=" * 40)


# ---------------------------------------------------------------------------
# Simulación de una ventana sobre un DataFrame ya enriquecido
# ---------------------------------------------------------------------------


def _simulate_window(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float,
    risk_per_trade: float,
    stop_loss_pct: float,
    max_drawdown_limit: float,
    max_position_pct: float,
    symbol: str,
    window_id: int,
) -> SimResult:
    """Corre un backtest sobre las filas del DataFrame dado."""
    broker = PaperBroker(starting_cash=initial_capital)
    risk = RiskManager(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        stop_loss_pct=stop_loss_pct,
        max_drawdown_limit=max_drawdown_limit,
        max_position_pct=max_position_pct,
    )

    equity_curve: list[float] = []
    trades: list[dict] = []

    df = df.reset_index(drop=True)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        price = float(curr["close"])
        equity = broker.cash() + broker.positions().get(symbol, 0.0) * price
        equity_curve.append(equity)

        drawdown_locked = risk.drawdown_breach(equity_curve)
        sig = strategy.signal(prev, curr)

        if sig == Signal.BUY and not drawdown_locked:
            if symbol in broker.positions():
                continue
            sizing = risk.position_size(equity=equity, price=price)
            if not sizing.approved or sizing.quantity * price > broker.cash():
                continue
            ts = curr["date"] if hasattr(curr["date"], "to_pydatetime") else curr["date"]
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            fill = broker.submit(Order(symbol=symbol, side=OrderSide.BUY, quantity=sizing.quantity, timestamp=ts, reference_price=price))
            trades.append({"side": "buy", "price": fill.price, "qty": fill.quantity, "equity": equity})

        elif sig == Signal.SELL:
            qty = broker.positions().get(symbol, 0.0)
            if qty <= 0:
                continue
            ts = curr["date"] if hasattr(curr["date"], "to_pydatetime") else curr["date"]
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            fill = broker.submit(Order(symbol=symbol, side=OrderSide.SELL, quantity=qty, timestamp=ts, reference_price=price))
            trades.append({"side": "sell", "price": fill.price, "qty": fill.quantity, "equity": equity})

    final_equity = broker.cash() + sum(
        qty * float(df.iloc[-1]["close"]) for sym, qty in broker.positions().items()
    )

    stats = stats_engine.summarize(equity_curve) if len(equity_curve) >= 2 else {}
    win_rate = _calc_win_rate(trades)

    period_start = df.iloc[0]["date"]
    period_end = df.iloc[-1]["date"]
    if hasattr(period_start, "to_pydatetime"):
        period_start = period_start.to_pydatetime()
    if hasattr(period_end, "to_pydatetime"):
        period_end = period_end.to_pydatetime()

    return SimResult(
        strategy_name=strategy.name,
        window_id=window_id,
        period_start=period_start,
        period_end=period_end,
        n_trades=len([t for t in trades if t["side"] == "buy"]),
        final_equity=final_equity,
        total_return=stats.get("total_return", 0.0),
        sharpe_ratio=stats.get("sharpe_ratio", 0.0),
        max_drawdown=stats.get("max_drawdown", 0.0),
        cagr=stats.get("cagr", 0.0),
        win_rate=win_rate,
        equity_curve=equity_curve,
        trades=trades,
    )


def _calc_win_rate(trades: list[dict]) -> float:
    """Calcula el porcentaje de trades ganadores (buy→sell con ganancia)."""
    buys, sells = [], []
    for t in trades:
        if t["side"] == "buy":
            buys.append(t["price"])
        elif t["side"] == "sell" and buys:
            buy_price = buys.pop(0)
            sells.append(t["price"] > buy_price)
    return sum(sells) / len(sells) if sells else 0.0


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def walk_forward(
    df: pd.DataFrame,
    strategies: Sequence[Strategy],
    symbol: str,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = 0.04,
    max_drawdown_limit: float = 0.15,
    max_position_pct: float = 0.25,
    n_splits: int = 3,
    train_pct: float = 0.70,
) -> BacktestReport:
    """Valida múltiples estrategias con walk-forward sobre un DataFrame enriquecido.

    Args:
        df:             DataFrame OHLCV + indicadores (output de indicators.enrich()).
        strategies:     Lista de estrategias a comparar.
        symbol:         Ticker del activo.
        n_splits:       Número de ventanas walk-forward.
        train_pct:      Fracción de cada ventana usada como período de calentamiento.
                        Solo el período TEST (1-train_pct) cuenta para métricas.

    Returns:
        BacktestReport con resultados y la mejor estrategia por Sharpe promedio OOS.
    """
    n = len(df)
    window_size = n // n_splits
    all_results: list[SimResult] = []

    log.info(
        "Walk-forward: %d splits, window=%d barras, train=%.0f%%, test=%.0f%%",
        n_splits, window_size, train_pct * 100, (1 - train_pct) * 100,
    )

    for split_idx in range(n_splits):
        start = split_idx * window_size
        end = start + window_size if split_idx < n_splits - 1 else n
        window_df = df.iloc[start:end].copy()

        train_end = start + int(window_size * train_pct)
        test_df = df.iloc[train_end:end].copy()

        if len(test_df) < 20:
            log.warning("Ventana %d: periodo test muy corto (%d barras), omitiendo.", split_idx, len(test_df))
            continue

        log.info(
            "Ventana %d: test %s → %s (%d barras)",
            split_idx,
            test_df.iloc[0]["date"],
            test_df.iloc[-1]["date"],
            len(test_df),
        )

        for strategy in strategies:
            result = _simulate_window(
                df=test_df,
                strategy=strategy,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                stop_loss_pct=stop_loss_pct,
                max_drawdown_limit=max_drawdown_limit,
                max_position_pct=max_position_pct,
                symbol=symbol,
                window_id=split_idx,
            )
            log.info(result.summary())
            all_results.append(result)

    # Calcular promedios OOS por estrategia
    strategy_names = list({r.strategy_name for r in all_results})
    avg_sharpe = {}
    avg_return = {}
    for name in strategy_names:
        subset = [r for r in all_results if r.strategy_name == name]
        avg_sharpe[name] = sum(r.sharpe_ratio for r in subset) / len(subset)
        avg_return[name] = sum(r.total_return for r in subset) / len(subset)

    best = max(avg_sharpe, key=lambda k: avg_sharpe[k]) if avg_sharpe else "none"

    return BacktestReport(
        results=all_results,
        best_strategy=best,
        avg_sharpe_by_strategy=avg_sharpe,
        avg_return_by_strategy=avg_return,
    )


# ---------------------------------------------------------------------------
# Función de conveniencia: backtest desde fixtures locales
# ---------------------------------------------------------------------------


def backtest_from_fixtures(
    fixtures_dir: Path,
    symbol: str,
    strategies: Sequence[Strategy],
    fast_ema: int = 9,
    slow_ema: int = 21,
    rsi_period: int = 14,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = 0.04,
    n_splits: int = 3,
) -> BacktestReport:
    """Carga fixture CSV, enriquece indicadores y corre walk-forward."""
    path = Path(fixtures_dir) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fixture no encontrado: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = enrich(df, fast_ema=fast_ema, slow_ema=slow_ema, rsi_period=rsi_period)

    return walk_forward(
        df=df,
        strategies=strategies,
        symbol=symbol,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        stop_loss_pct=stop_loss_pct,
        n_splits=n_splits,
    )
