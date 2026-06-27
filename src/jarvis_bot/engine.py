"""Pipeline de paper trading.

Flujo:
    load_historical_data
      -> calculate_indicators
      -> generate_signals
      -> apply_risk_manager + simulate_order via PaperBroker
      -> persist portfolio snapshots + history

Cada bar genera como mucho una operacion por simbolo (long-only en el MVP).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .brokers import Broker, Order, OrderSide, PaperBroker
from .config import AppConfig
from .modules.risk_manager import RiskManager
from .modules.sentiment import SentimentAnalyzer
from .modules.macro import MacroAnalyzer
from .modules.alpaca_executor import AlpacaExecutor
from .modules.alpaca_data import AlpacaDataIngestor

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


# -- Carga de datos ----------------------------------------------------------


def load_historical_data(fixtures_dir: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Carga CSVs por simbolo desde una carpeta local.

    Espera archivos {SYMBOL}.csv con columnas: date, open, high, low, close, volume.
    """
    data: dict[str, pd.DataFrame] = {}
    fixtures_dir = Path(fixtures_dir)
    for symbol in symbols:
        path = fixtures_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Fixture no encontrado: {path}")
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"{path} le faltan columnas: {missing}")
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        data[symbol] = df
    return data


# -- Indicadores y senales ---------------------------------------------------


def calculate_indicators(
    df: pd.DataFrame, fast: int, slow: int, rsi_period: int
) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    out["rsi"] = _rsi(out["close"], rsi_period)
    return out


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def generate_signal(
    row_prev: pd.Series,
    row: pd.Series,
    rsi_overbought: float,
    rsi_oversold: float,
    sentiment_score: float = 0.5,
    macro_risk: str = "normal",
) -> str:
    """Estrategia MVP: cruce de medias EMA confirmado por RSI, Sentimiento y Macro.

    - BUY cuando ema_fast cruza por encima de ema_slow, RSI no esta sobrecomprado, sentimiento no es bajista extremo y macro normal.
    - SELL cuando ema_fast cruza por debajo de ema_slow, RSI sobrecomprado, o sentimiento bajista extremo forzando salida.
    - HOLD en cualquier otro caso.
    """
    if any(pd.isna(v) for v in (row_prev.ema_fast, row_prev.ema_slow, row.ema_fast, row.ema_slow)):
        return "HOLD"

    crossed_up = row_prev.ema_fast <= row_prev.ema_slow and row.ema_fast > row.ema_slow
    crossed_down = row_prev.ema_fast >= row_prev.ema_slow and row.ema_fast < row.ema_slow

    # Incorporamos el sentimiento como filtro externo
    is_bearish_extreme = sentiment_score < 0.35
    is_macro_danger = macro_risk == "high_risk"

    if crossed_up and row.rsi < rsi_overbought and not is_bearish_extreme and not is_macro_danger:
        return "BUY"
    if crossed_down or row.rsi > rsi_overbought or (row.rsi > 50 and is_bearish_extreme) or (is_macro_danger and row.ema_fast < row.ema_slow):
        return "SELL"
    if row.rsi < rsi_oversold and row.ema_fast > row.ema_slow and not is_bearish_extreme and not is_macro_danger:
        return "BUY"
    return "HOLD"


# -- Pipeline ----------------------------------------------------------------


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    positions_value: float
    equity: float
    positions: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    snapshots: list[PortfolioSnapshot]
    trades: list[dict]
    final_equity: float
    symbols: list[str]


def _portfolio_value(broker: Broker, prices: dict[str, float]) -> tuple[float, float]:
    positions = broker.positions()
    positions_value = sum(qty * prices.get(sym, 0.0) for sym, qty in positions.items())
    return broker.cash(), positions_value


def run_pipeline(cfg: AppConfig, broker: Broker | None = None) -> PipelineResult:
    if not cfg.is_paper():
        raise RuntimeError("run_pipeline solo se ejecuta en modo paper.")

    broker = broker or PaperBroker(starting_cash=cfg.risk.initial_capital)
    risk = RiskManager(
        initial_capital=cfg.risk.initial_capital,
        risk_per_trade=cfg.risk.risk_per_trade,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        max_drawdown_limit=cfg.risk.max_drawdown_limit,
        max_position_pct=cfg.risk.max_position_pct,
    )
    sentiment_analyzer = SentimentAnalyzer(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
    macro_analyzer = MacroAnalyzer(api_key=os.getenv("FRED_API_KEY"))
    alpaca_executor = AlpacaExecutor()
    data_ingestor = AlpacaDataIngestor()

    log.info(
        "Obteniendo datos %s (lookback=%dd, feed=%s) desde Alpaca...",
        cfg.data.timeframe,
        cfg.data.lookback_days,
        data_ingestor.feed,
    )
    raw = data_ingestor.fetch_historical_data(
        cfg.data.symbols,
        timeframe=cfg.data.timeframe,
        lookback_days=cfg.data.lookback_days,
    )

    # Timeframes adicionales (ej. 1Day para contexto de tendencia) — solo se logean por ahora
    if cfg.data.extra_timeframes:
        log.info("Descargando timeframes adicionales: %s", list(cfg.data.extra_timeframes))
        data_ingestor.fetch_multi_timeframe(
            cfg.data.symbols,
            timeframes=list(cfg.data.extra_timeframes),
            lookback_days=cfg.data.lookback_days,
        )

    enriched = {
        sym: calculate_indicators(
            df,
            fast=cfg.signals.fast_ma,
            slow=cfg.signals.slow_ma,
            rsi_period=cfg.signals.rsi_period,
        )
        for sym, df in raw.items()
        if not df.empty
    }

    if not enriched:
        log.error("Sin datos enriquecidos — verifica las API keys y los símbolos configurados.")
        return PipelineResult(snapshots=[], trades=[], final_equity=cfg.risk.initial_capital, symbols=list(cfg.data.symbols))

    # Alinea por fecha (intersección). Mantiene reproducibilidad.
    common_dates = sorted(
        set.intersection(*(set(df["date"].tolist()) for df in enriched.values()))
    )

    snapshots: list[PortfolioSnapshot] = []
    trades: list[dict] = []
    equity_curve: list[float] = []

    indexed = {sym: df.set_index("date") for sym, df in enriched.items() if not df.empty}


    for i, date in enumerate(common_dates):
        if i == 0:
            continue  # necesitamos previa para detectar cruces

        prev_date = common_dates[i - 1]
        prices_today = {sym: float(indexed[sym].loc[date, "close"]) for sym in indexed}
        equity = broker.cash() + sum(
            qty * prices_today[sym] for sym, qty in broker.positions().items()
        )
        equity_curve.append(equity)
        drawdown_locked = risk.drawdown_breach(equity_curve)

        # En backtest llamariamos un mock historico, pero para paper trading simularemos o llamaremos a la API real
        # Optimizacion: Para no agotar API limits por cada bar, simulamos el score basado en mock salvo el ultimo dia
        is_latest_bar = (i == len(common_dates) - 1)
        if is_latest_bar and cfg.is_paper():
            # fetch current real sentiment for forward testing execution
            global_sentiment = sentiment_analyzer.fetch_news_sentiment(list(indexed.keys()))
            macro_state = macro_analyzer.fetch_macro_state()
        else:
            global_sentiment = 0.5  # Neutral para las fechas pasadas del backtest inicial
            macro_state = "normal"

        for sym, df in indexed.items():
            row_prev = df.loc[prev_date]
            row = df.loc[date]
            signal = generate_signal(
                row_prev, row, cfg.signals.rsi_overbought, cfg.signals.rsi_oversold, global_sentiment, macro_state
            )
            price = float(row["close"])

            if signal == "BUY" and not drawdown_locked:
                if sym in broker.positions():
                    continue  # ya largo: no piramidamos en el MVP
                sizing = risk.position_size(equity=equity, price=price)
                if not sizing.approved:
                    continue
                qty = sizing.quantity
                if qty * price > broker.cash():
                    continue
                fill = broker.submit(
                    Order(
                        symbol=sym,
                        side=OrderSide.BUY,
                        quantity=qty,
                        timestamp=date.to_pydatetime() if hasattr(date, "to_pydatetime") else date,
                        reference_price=price,
                    )
                )
                trades.append(_trade_dict(fill, signal, equity))
                
                # Ejecución en Broker Real (Alpaca) si es el último día (operación actual)
                if is_latest_bar and cfg.is_paper():
                    alpaca_executor.submit_order(sym, "buy", qty)

            elif signal == "SELL":
                qty = broker.positions().get(sym, 0.0)
                if qty <= 0:
                    continue
                fill = broker.submit(
                    Order(
                        symbol=sym,
                        side=OrderSide.SELL,
                        quantity=qty,
                        timestamp=date.to_pydatetime() if hasattr(date, "to_pydatetime") else date,
                        reference_price=price,
                    )
                )
                trades.append(_trade_dict(fill, signal, equity))
                
                # Ejecución en Broker Real (Alpaca) si es el último día (operación actual)
                if is_latest_bar and cfg.is_paper():
                    alpaca_executor.submit_order(sym, "sell", qty)

        cash, positions_value = _portfolio_value(broker, prices_today)
        snapshots.append(
            PortfolioSnapshot(
                timestamp=date.to_pydatetime() if hasattr(date, "to_pydatetime") else date,
                cash=cash,
                positions_value=positions_value,
                equity=cash + positions_value,
                positions=broker.positions(),
            )
        )

    final_equity = snapshots[-1].equity if snapshots else cfg.risk.initial_capital
    log.info(
        "Pipeline completo: %d snapshots, %d trades, equity final=%.2f",
        len(snapshots),
        len(trades),
        final_equity,
    )
    return PipelineResult(
        snapshots=snapshots,
        trades=trades,
        final_equity=final_equity,
        symbols=list(cfg.data.symbols),
    )


def _trade_dict(fill, signal: str, equity_pre: float) -> dict:
    return {
        "timestamp": fill.timestamp,
        "symbol": fill.symbol,
        "side": fill.side.value,
        "quantity": fill.quantity,
        "price": fill.price,
        "signal": signal,
        "equity_pre_trade": equity_pre,
        "commission": fill.commission,
    }
