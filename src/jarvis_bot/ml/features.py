"""Ingeniería de features para el clasificador de señales — Fase 6.

Diseño:
  - Entrada: DataFrame enriquecido (salida de indicators.enrich())
  - Salida: feature matrix X (n_samples × n_features) y targets y

Features incluidas (todas numéricas, sin look-ahead):
  Indicadores técnicos:
    rsi, macd_hist, macd_line, bb_pct_b, bb_width
    volume_ratio, atr_pct
    ema_crossover: (ema_fast - ema_slow) / close  (relativo, sin escala de precio)

  Features lag (t-1, t-2, t-5):
    rsi_lag1, rsi_lag2, rsi_lag5
    macd_hist_lag1, macd_hist_lag2
    return_1d, return_5d, return_10d  (retorno de precio)

  Features de volatilidad:
    close_std_5d: std rolling 5 barras del retorno
    atr_pct (ya incluido en enrich)

Target:
  Retorno forward a N barras clasificado:
    BUY  (+1): retorno > +threshold
    SELL (-1): retorno < -threshold
    HOLD  (0): dentro del rango ±threshold

  threshold por defecto: 1% (configurable)
  horizon por defecto: 5 barras (configurable)

Precauciones:
  - Elimina filas con NaN en features O target (primeras N filas)
  - No filtra por fecha: el caller es responsable del split train/test
  - No normaliza: RandomForest no lo necesita, pero se puede activar
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columnas requeridas en el DataFrame de entrada
REQUIRED_INDICATOR_COLS = {
    "close", "rsi", "macd_hist", "macd_line", "bb_pct_b", "bb_width",
    "volume_ratio", "ema_fast", "ema_slow",
}

# Nombres de las features en el orden que se construyen
FEATURE_NAMES = [
    # Indicadores técnicos
    "rsi",
    "macd_hist",
    "macd_line",
    "bb_pct_b",
    "bb_width",
    "volume_ratio",
    "ema_crossover",         # (ema_fast - ema_slow) / close
    # Retornos
    "return_1d",             # retorno 1 barra
    "return_5d",             # retorno 5 barras
    "return_10d",            # retorno 10 barras
    # Lags RSI
    "rsi_lag1",
    "rsi_lag2",
    "rsi_lag5",
    # Lags MACD histogram
    "macd_hist_lag1",
    "macd_hist_lag2",
    # Volatilidad realizada
    "close_std_5d",          # std rolling de return_1d en 5 barras
]

# Opcionales (presentes solo si enrich() calculó ATR)
_OPTIONAL_FEATURES = ["atr_pct"]


@dataclass
class FeatureConfig:
    """Parámetros para la construcción de features y targets."""

    forward_horizon: int = 5       # barras hacia adelante para calcular el target
    target_threshold_pct: float = 1.0   # % mínimo de movimiento para BUY/SELL
    include_optional: bool = True  # incluir atr_pct si está disponible
    drop_na: bool = True           # eliminar filas con NaN


@dataclass
class FeatureMatrix:
    """Resultado de build_feature_matrix()."""

    X: pd.DataFrame              # features (sin NaN si drop_na=True)
    y: pd.Series                 # targets: -1, 0, +1
    feature_names: list[str]
    valid_index: pd.Index        # índice original de las filas incluidas
    n_buy: int = 0
    n_sell: int = 0
    n_hold: int = 0

    def class_balance(self) -> dict:
        total = self.n_buy + self.n_sell + self.n_hold
        if total == 0:
            return {}
        return {
            "BUY": f"{self.n_buy} ({self.n_buy / total * 100:.1f}%)",
            "HOLD": f"{self.n_hold} ({self.n_hold / total * 100:.1f}%)",
            "SELL": f"{self.n_sell} ({self.n_sell / total * 100:.1f}%)",
        }


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Construye la feature matrix sin targets (para inferencia en tiempo real).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame enriquecido (debe tener columnas de indicators.enrich()).
    cfg : FeatureConfig | None
        Configuración de features. Se usan valores por defecto si es None.

    Retorna
    -------
    pd.DataFrame con las features calculadas, mismo índice que df.
    """
    cfg = cfg or FeatureConfig()
    _validate_columns(df)
    feat = pd.DataFrame(index=df.index)

    # Indicadores técnicos directos
    feat["rsi"] = df["rsi"]
    feat["macd_hist"] = df["macd_hist"]
    feat["macd_line"] = df["macd_line"]
    feat["bb_pct_b"] = df["bb_pct_b"]
    feat["bb_width"] = df["bb_width"]
    feat["volume_ratio"] = df["volume_ratio"]
    feat["ema_crossover"] = (df["ema_fast"] - df["ema_slow"]) / df["close"].replace(0, np.nan)

    # Retornos de precio (log-return para estabilidad numérica)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    feat["return_1d"] = log_ret
    feat["return_5d"] = np.log(df["close"] / df["close"].shift(5))
    feat["return_10d"] = np.log(df["close"] / df["close"].shift(10))

    # Lags
    feat["rsi_lag1"] = df["rsi"].shift(1)
    feat["rsi_lag2"] = df["rsi"].shift(2)
    feat["rsi_lag5"] = df["rsi"].shift(5)
    feat["macd_hist_lag1"] = df["macd_hist"].shift(1)
    feat["macd_hist_lag2"] = df["macd_hist"].shift(2)

    # Volatilidad realizada (std de log-returns en ventana de 5)
    feat["close_std_5d"] = log_ret.rolling(5).std()

    # Opcionales
    if cfg.include_optional and "atr_pct" in df.columns:
        feat["atr_pct"] = df["atr_pct"]

    return feat


def build_targets(
    df: pd.DataFrame,
    forward_horizon: int = 5,
    threshold_pct: float = 1.0,
) -> pd.Series:
    """Calcula el target de clasificación basado en retorno forward.

    Target:
      +1 (BUY)  si close[t+horizon] / close[t] - 1 > +threshold_pct/100
      -1 (SELL) si close[t+horizon] / close[t] - 1 < -threshold_pct/100
       0 (HOLD) en otro caso

    Las últimas `forward_horizon` filas tendrán target NaN (no hay futuro).
    """
    future_close = df["close"].shift(-forward_horizon)
    forward_return = (future_close / df["close"]) - 1.0
    threshold = threshold_pct / 100.0

    target = pd.Series(0, index=df.index, dtype=int)
    target[forward_return > threshold] = 1
    target[forward_return < -threshold] = -1
    target[forward_return.isna()] = np.nan  # type: ignore[call-overload]
    return target.astype("Int64")  # nullable int para NaN


def build_feature_matrix(
    df: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> FeatureMatrix:
    """Construye features + targets listos para entrenamiento.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame enriquecido (salida de indicators.enrich()).
    cfg : FeatureConfig | None
        Configuración de features/targets.

    Retorna
    -------
    FeatureMatrix con X, y, feature_names, valid_index y estadísticas de clase.
    """
    cfg = cfg or FeatureConfig()

    feat = build_features(df, cfg)
    target = build_targets(df, cfg.forward_horizon, cfg.target_threshold_pct)

    # Combinar y eliminar NaN
    combined = feat.copy()
    combined["__target__"] = target

    if cfg.drop_na:
        combined = combined.dropna()

    X = combined.drop(columns=["__target__"])
    y = combined["__target__"].astype(int)

    feature_names = list(X.columns)
    n_buy = int((y == 1).sum())
    n_sell = int((y == -1).sum())
    n_hold = int((y == 0).sum())

    log.info(
        "Feature matrix: %d muestras × %d features | BUY=%d HOLD=%d SELL=%d",
        len(X), len(feature_names), n_buy, n_hold, n_sell,
    )

    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=feature_names,
        valid_index=X.index,
        n_buy=n_buy,
        n_sell=n_sell,
        n_hold=n_hold,
    )


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_INDICATOR_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"El DataFrame le faltan columnas requeridas para features ML: {missing}. "
            f"Asegúrate de pasar la salida de indicators.enrich()."
        )
