"""Pipeline de paper trading.

Flujo:
    load_historical_data
      -> calculate_indicators (enrich)
      -> generate_signals (strategy pattern)
      -> apply_risk_manager + FXRiskManager + simulate_order via PaperBroker
      -> persist portfolio snapshots + history

Contexto Colombia:
  - FXRiskManager ajusta el tamaño de posición según volatilidad COP/USD
  - market_hours guard previene operar fuera del horario NYSE
  - PnL se registra en USD y COP para reporting

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
from .indicators import enrich
from .modules.alpaca_data import AlpacaDataIngestor
from .modules.alpaca_executor import AlpacaExecutor, BracketConfig
from .modules.fx_risk import FXRiskManager
from .modules.macro import MacroAnalyzer
from .modules.market_hours import is_market_open, market_status
from .modules.position_reconciler import PositionReconciler
from .modules.risk_manager import RiskManager
from .modules.sentiment import SentimentAnalyzer
from .strategies import get_strategy
from .strategies.base import Signal

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
    cop_equity: float | None = None       # Valor del portafolio en COP (si FX disponible)


@dataclass
class PipelineResult:
    snapshots: list[PortfolioSnapshot]
    trades: list[dict]
    final_equity: float
    symbols: list[str]
    fx_summary: dict = field(default_factory=dict)   # Resumen FX al final del pipeline


def _portfolio_value(broker: Broker, prices: dict[str, float]) -> tuple[float, float]:
    positions = broker.positions()
    positions_value = sum(qty * prices.get(sym, 0.0) for sym, qty in positions.items())
    return broker.cash(), positions_value


def run_pipeline(cfg: AppConfig, broker: Broker | None = None) -> PipelineResult:
    if not cfg.is_paper():
        raise RuntimeError("run_pipeline solo se ejecuta en modo paper.")

    # -- Guard de horario de mercado (Colombia context) --
    if cfg.colombia.check_market_hours and not is_market_open():
        status = market_status()
        log.warning(
            "Mercado cerrado (sesión: '%s'). "
            "Hora ET: %s / Bogotá: %s. "
            "Próxima apertura: %s (%s Bogotá). "
            "El pipeline continuará en modo histórico sin enviar órdenes.",
            status["session"],
            status["et_time"],
            status["bogota_time"],
            status["next_open_et"],
            status["next_open_bogota"],
        )

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

    # Executor con bracket orders (stop-loss + take-profit automáticos)
    bracket = BracketConfig(
        stop_loss_pct=cfg.risk.stop_loss_pct,
        take_profit_pct=cfg.risk.take_profit_pct,
    )
    alpaca_executor = AlpacaExecutor(bracket=bracket)

    # Reconciliador de posiciones (detecta drift entre estado local y Alpaca)
    reconciler = PositionReconciler(executor=alpaca_executor)

    data_ingestor = AlpacaDataIngestor()

    # Reconciliar posiciones al inicio (restaura estado si el bot se reinició)
    if alpaca_executor.is_configured():
        remote_positions = reconciler.positions_from_alpaca()
        if remote_positions and isinstance(broker, PaperBroker):
            log.info("Restaurando %d posiciones desde Alpaca.", len(remote_positions))
            for sym, qty in remote_positions.items():
                broker._positions[sym] = qty  # type: ignore[attr-defined]

    # FX Risk Manager (contexto Colombia)
    fx_risk = FXRiskManager(
        fx_vol_caution_pct=cfg.colombia.fx_vol_caution_pct,
        fx_vol_block_pct=cfg.colombia.fx_vol_block_pct,
    )

    # Instancia la estrategia según config
    strategy = get_strategy(
        cfg.signals.strategy,
        rsi_overbought=cfg.signals.rsi_overbought,
        rsi_oversold=cfg.signals.rsi_oversold,
    )
    log.info("Estrategia activa: %s", strategy.name)

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

    # Timeframes adicionales (ej. 1Day para contexto de tendencia)
    if cfg.data.extra_timeframes:
        log.info("Descargando timeframes adicionales: %s", list(cfg.data.extra_timeframes))
        data_ingestor.fetch_multi_timeframe(
            cfg.data.symbols,
            timeframes=list(cfg.data.extra_timeframes),
            lookback_days=cfg.data.lookback_days,
        )

    enriched = {
        sym: enrich(
            df,
            fast_ema=cfg.signals.fast_ma,
            slow_ema=cfg.signals.slow_ma,
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

    # Estado FX inicial (se actualiza en la última barra con datos reales)
    fx_state = None
    current_usd_cop: float | None = None

    for i, date in enumerate(common_dates):
        if i == 0:
            continue  # necesitamos fila previa para señales

        prev_date = common_dates[i - 1]
        prices_today = {sym: float(indexed[sym].loc[date, "close"]) for sym in indexed}
        equity = broker.cash() + sum(
            qty * prices_today[sym] for sym, qty in broker.positions().items()
        )
        equity_curve.append(equity)
        drawdown_locked = risk.drawdown_breach(equity_curve)

        # Sentimiento, macro y FX solo en la última barra (evita agotar API limits)
        is_latest_bar = (i == len(common_dates) - 1)
        if is_latest_bar and cfg.is_paper():
            global_sentiment = sentiment_analyzer.fetch_news_sentiment(list(indexed.keys()))
            macro_full = macro_analyzer.fetch_full_state()
            macro_state = macro_full.risk_level if hasattr(macro_full, "risk_level") else macro_analyzer.fetch_macro_state()

            # Actualizar FX state con datos reales
            if cfg.colombia.apply_fx_risk and hasattr(macro_full, "usd_cop") and macro_full.usd_cop:
                current_usd_cop = macro_full.usd_cop
                change_pct = macro_full.usd_cop_30d_change_pct or 0.0
                fx_state = fx_risk.evaluate(current_usd_cop, change_pct)
                log.info("FX Colombia: %s", fx_state)
        else:
            global_sentiment = 0.5
            macro_state = "normal"

        # Calcular multiplicador FX para ajustar tamaño de posición
        fx_multiplier = fx_risk.position_size_multiplier(fx_state) if fx_state else 1.0
        fx_allows_entry = fx_risk.allows_new_entry(fx_state) if fx_state else True

        for sym, df in indexed.items():
            row_prev = df.loc[prev_date]
            row = df.loc[date]
            sig = strategy.signal(row_prev, row, global_sentiment, macro_state)
            price = float(row["close"])

            if sig == Signal.BUY and not drawdown_locked and fx_allows_entry:
                if sym in broker.positions():
                    continue  # ya largo: no piramidamos en el MVP
                sizing = risk.position_size(equity=equity, price=price)
                if not sizing.approved:
                    continue
                # Aplicar multiplicador FX al tamaño de posición
                qty = max(0, int(sizing.quantity * fx_multiplier))
                if qty <= 0:
                    log.info("FX cautela: posición para %s reducida a 0 — omitida.", sym)
                    continue
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
                # Registrar tasa FX de entrada para calcular PnL en COP después
                if current_usd_cop:
                    fx_risk.record_entry(sym, current_usd_cop)

                trade = _trade_dict(fill, sig.value, equity)
                trade["usd_cop_entry"] = current_usd_cop
                trades.append(trade)

                # Ejecución en Broker Real (Alpaca) solo en la última barra y si mercado abierto
                if is_latest_bar and cfg.is_paper() and is_market_open():
                    fill_result = alpaca_executor.submit_order(
                        sym, "buy", qty, ref_price=price, use_bracket=True
                    )
                    if fill_result and fill_result.is_filled:
                        log.info("Fill real Alpaca: %s", fill_result)
                        trade["alpaca_fill_price"] = fill_result.avg_fill_price
                        trade["alpaca_order_id"] = fill_result.order_id

            elif sig == Signal.SELL:
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
                # Calcular PnL ajustado por FX en COP
                pnl_usd = fill.price * fill.quantity - (
                    fx_risk.entry_rate(sym) * 0 if not current_usd_cop else 0
                )
                trade = _trade_dict(fill, sig.value, equity)
                trade["usd_cop_exit"] = current_usd_cop
                if current_usd_cop and fx_risk.entry_rate(sym):
                    pnl_adj = fx_risk.adjusted_pnl(
                        sym,
                        pnl_usd=fill.price * fill.quantity,  # bruto, proxy
                        position_usd=fill.price * fill.quantity,
                        current_usd_cop=current_usd_cop,
                    )
                    trade["pnl_cop_approx"] = round(pnl_adj.pnl_cop, 0)
                    trade["fx_impact_cop"] = round(pnl_adj.fx_impact_cop, 0)
                fx_risk.record_exit(sym)
                trades.append(trade)

                # Ejecución en Broker Real (Alpaca) solo en la última barra y si mercado abierto
                if is_latest_bar and cfg.is_paper() and is_market_open():
                    fill_result = alpaca_executor.submit_order(sym, "sell", qty, use_bracket=False)
                    if fill_result and fill_result.is_filled:
                        log.info("Fill real Alpaca: %s", fill_result)
                        trade["alpaca_fill_price"] = fill_result.avg_fill_price
                        trade["alpaca_order_id"] = fill_result.order_id

        cash, positions_value = _portfolio_value(broker, prices_today)
        cop_equity = fx_risk.portfolio_value_cop(cash + positions_value) if current_usd_cop else None
        snapshots.append(
            PortfolioSnapshot(
                timestamp=date.to_pydatetime() if hasattr(date, "to_pydatetime") else date,
                cash=cash,
                positions_value=positions_value,
                equity=cash + positions_value,
                positions=broker.positions(),
                cop_equity=cop_equity,
            )
        )

    final_equity = snapshots[-1].equity if snapshots else cfg.risk.initial_capital
    final_fx_summary = fx_risk.fx_summary(final_equity)

    # Reconciliación final: verifica que el estado local coincide con Alpaca
    if alpaca_executor.is_configured():
        reconciler.reconcile(broker.positions())

    log.info(
        "Pipeline completo: %d snapshots, %d trades, equity=%.2f USD%s",
        len(snapshots),
        len(trades),
        final_equity,
        f" / {final_fx_summary.get('cop_value', 0):,.0f} COP" if final_fx_summary.get("cop_value") else "",
    )
    return PipelineResult(
        snapshots=snapshots,
        trades=trades,
        final_equity=final_equity,
        symbols=list(cfg.data.symbols),
        fx_summary=final_fx_summary,
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
        # FX (Fase 3)
        "usd_cop_entry": None,
        "usd_cop_exit": None,
        "pnl_cop_approx": None,
        "fx_impact_cop": None,
        # Alpaca live fill (Fase 4)
        "alpaca_fill_price": None,
        "alpaca_order_id": None,
    }
