"""Carga configuracion desde YAML + variables de entorno.

Sin secretos hardcodeados. Cualquier API key se lee de env (no se persiste).
El modo trading queda fijado a 'paper' por defecto y solo cambia si se exige
explicitamente via env JARVIS_TRADING_MODE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class RiskConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_drawdown_limit: float = 0.15
    max_position_pct: float = 0.25


@dataclass(frozen=True)
class SignalConfig:
    fast_ma: int = 9
    slow_ma: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    strategy: str = "combined"   # trend_follow | mean_reversion | combined


@dataclass(frozen=True)
class DataConfig:
    fixtures_dir: Path = field(default_factory=lambda: Path("data/fixtures"))
    symbols: tuple[str, ...] = ("AAPL", "VOO", "QQQ")
    # Parámetros de ingesta live (Alpaca)
    timeframe: str = "1Hour"          # resolución principal de datos
    lookback_days: int = 365          # días de historia a pedir a Alpaca
    extra_timeframes: tuple[str, ...] = ()  # timeframes adicionales (ej. "1Day")


@dataclass(frozen=True)
class ColombiaConfig:
    """Configuración para el contexto de inversión desde Colombia."""

    apply_fx_risk: bool = True            # Activar módulo de riesgo FX COP/USD
    fx_vol_caution_pct: float = 5.0      # % cambio 30d que activa cautela (reducir posición)
    fx_vol_block_pct: float = 10.0       # % cambio 30d que bloquea nuevas entradas
    check_market_hours: bool = True       # Guard: no operar fuera de horario NYSE
    timezone: str = "America/Bogota"     # Zona horaria del usuario


@dataclass(frozen=True)
class MLConfig:
    """Configuración del módulo de Machine Learning (Fase 6)."""

    enabled: bool = False               # Activar estrategia ML en el pipeline
    model_path: str = "models/signal_classifier.pkl"  # Ruta del modelo entrenado
    confidence_threshold: float = 0.45  # Umbral mínimo de confianza para BUY/SELL
    n_splits: int = 3                   # Splits para walk-forward training
    train_pct: float = 0.70             # % de cada split usado para entrenamiento
    forward_horizon: int = 5            # Barras hacia adelante para el target
    target_threshold_pct: float = 1.0  # % mínimo de movimiento para BUY/SELL target


@dataclass(frozen=True)
class MarketRegimeConfig:
    """Configuración del análisis de régimen de mercado (VIX + WTI)."""

    enabled: bool = True
    vix_calm_threshold: float = 20.0      # VIX < este valor → sin restricción
    vix_elevated_threshold: float = 25.0  # VIX 20-25 → reducir al 75%
    vix_extreme_threshold: float = 35.0   # VIX > este valor → bloquear
    wti_caution_pct: float = -10.0        # WTI -10% en 30d → reducir al 75%
    wti_warning_pct: float = -20.0        # WTI -20% en 30d → reducir al 50%


@dataclass(frozen=True)
class OutputConfig:
    state_path: Path = field(default_factory=lambda: Path("data/portfolio_state.csv"))
    history_path: Path = field(default_factory=lambda: Path("data/history.csv"))
    report_path: Path = field(default_factory=lambda: Path("logs/report.md"))


@dataclass(frozen=True)
class AppConfig:
    trading_mode: str = "paper"
    log_level: str = "INFO"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    colombia: ColombiaConfig = field(default_factory=ColombiaConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    market_regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def is_paper(self) -> bool:
        return self.trading_mode.lower() == "paper"


def _coerce_path(value: Any, default: Path) -> Path:
    return Path(value) if value is not None else default


def _build_config(raw: Mapping[str, Any]) -> AppConfig:
    data_raw = raw.get("data", {}) or {}
    signals_raw = raw.get("signals", {}) or {}
    risk_raw = raw.get("risk", {}) or {}
    colombia_raw = raw.get("colombia", {}) or {}
    output_raw = raw.get("output", {}) or {}

    _data_defaults = DataConfig()
    data = DataConfig(
        fixtures_dir=_coerce_path(data_raw.get("fixtures_dir"), _data_defaults.fixtures_dir),
        symbols=tuple(data_raw.get("symbols") or _data_defaults.symbols),
        timeframe=str(data_raw.get("timeframe", _data_defaults.timeframe)),
        lookback_days=int(data_raw.get("lookback_days", _data_defaults.lookback_days)),
        extra_timeframes=tuple(data_raw.get("extra_timeframes") or _data_defaults.extra_timeframes),
    )
    signals = SignalConfig(
        fast_ma=int(signals_raw.get("fast_ma", SignalConfig.fast_ma)),
        slow_ma=int(signals_raw.get("slow_ma", SignalConfig.slow_ma)),
        rsi_period=int(signals_raw.get("rsi_period", SignalConfig.rsi_period)),
        rsi_overbought=float(signals_raw.get("rsi_overbought", SignalConfig.rsi_overbought)),
        rsi_oversold=float(signals_raw.get("rsi_oversold", SignalConfig.rsi_oversold)),
        strategy=str(signals_raw.get("strategy", SignalConfig.strategy)),
    )
    risk = RiskConfig(
        initial_capital=float(risk_raw.get("initial_capital", RiskConfig.initial_capital)),
        risk_per_trade=float(risk_raw.get("risk_per_trade", RiskConfig.risk_per_trade)),
        stop_loss_pct=float(risk_raw.get("stop_loss_pct", RiskConfig.stop_loss_pct)),
        take_profit_pct=float(risk_raw.get("take_profit_pct", RiskConfig.take_profit_pct)),
        max_drawdown_limit=float(
            risk_raw.get("max_drawdown_limit", RiskConfig.max_drawdown_limit)
        ),
        max_position_pct=float(risk_raw.get("max_position_pct", RiskConfig.max_position_pct)),
    )
    _co_defaults = ColombiaConfig()
    colombia = ColombiaConfig(
        apply_fx_risk=bool(colombia_raw.get("apply_fx_risk", _co_defaults.apply_fx_risk)),
        fx_vol_caution_pct=float(
            colombia_raw.get("fx_vol_caution_pct", _co_defaults.fx_vol_caution_pct)
        ),
        fx_vol_block_pct=float(
            colombia_raw.get("fx_vol_block_pct", _co_defaults.fx_vol_block_pct)
        ),
        check_market_hours=bool(
            colombia_raw.get("check_market_hours", _co_defaults.check_market_hours)
        ),
        timezone=str(colombia_raw.get("timezone", _co_defaults.timezone)),
    )
    regime_raw = raw.get("market_regime", {}) or {}
    _regime_defaults = MarketRegimeConfig()
    market_regime = MarketRegimeConfig(
        enabled=bool(regime_raw.get("enabled", _regime_defaults.enabled)),
        vix_calm_threshold=float(regime_raw.get("vix_calm_threshold", _regime_defaults.vix_calm_threshold)),
        vix_elevated_threshold=float(regime_raw.get("vix_elevated_threshold", _regime_defaults.vix_elevated_threshold)),
        vix_extreme_threshold=float(regime_raw.get("vix_extreme_threshold", _regime_defaults.vix_extreme_threshold)),
        wti_caution_pct=float(regime_raw.get("wti_caution_pct", _regime_defaults.wti_caution_pct)),
        wti_warning_pct=float(regime_raw.get("wti_warning_pct", _regime_defaults.wti_warning_pct)),
    )
    ml_raw = raw.get("ml", {}) or {}
    _ml_defaults = MLConfig()
    ml = MLConfig(
        enabled=bool(ml_raw.get("enabled", _ml_defaults.enabled)),
        model_path=str(ml_raw.get("model_path", _ml_defaults.model_path)),
        confidence_threshold=float(
            ml_raw.get("confidence_threshold", _ml_defaults.confidence_threshold)
        ),
        n_splits=int(ml_raw.get("n_splits", _ml_defaults.n_splits)),
        train_pct=float(ml_raw.get("train_pct", _ml_defaults.train_pct)),
        forward_horizon=int(ml_raw.get("forward_horizon", _ml_defaults.forward_horizon)),
        target_threshold_pct=float(
            ml_raw.get("target_threshold_pct", _ml_defaults.target_threshold_pct)
        ),
    )
    output = OutputConfig(
        state_path=_coerce_path(output_raw.get("state_path"), OutputConfig().state_path),
        history_path=_coerce_path(output_raw.get("history_path"), OutputConfig().history_path),
        report_path=_coerce_path(output_raw.get("report_path"), OutputConfig().report_path),
    )
    return AppConfig(
        trading_mode=str(raw.get("trading_mode", "paper")),
        log_level=str(raw.get("log_level", "INFO")),
        seed=int(raw.get("seed", 42)),
        data=data,
        signals=signals,
        risk=risk,
        colombia=colombia,
        ml=ml,
        market_regime=market_regime,
        output=output,
    )


def load_config(path: Path | str | None = None) -> AppConfig:
    load_dotenv(override=False)

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    env_mode = os.getenv("JARVIS_TRADING_MODE")
    if env_mode:
        raw["trading_mode"] = env_mode
    env_log = os.getenv("JARVIS_LOG_LEVEL")
    if env_log:
        raw["log_level"] = env_log

    cfg = _build_config(raw)

    # Guardrail: ningun otro modo soportado en el MVP.
    if cfg.trading_mode.lower() != "paper":
        raise RuntimeError(
            f"trading_mode='{cfg.trading_mode}' no soportado. MVP solo permite 'paper'."
        )
    return cfg
