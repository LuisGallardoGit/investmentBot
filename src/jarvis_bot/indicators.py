"""Indicadores técnicos — funciones puras sobre pandas Series/DataFrame.

Todos los indicadores reciben Series numéricas y devuelven Series.
Sin efectos secundarios, sin IO, sin estado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Medias móviles
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Media móvil simple."""
    return series.rolling(window=window).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing via EWM).

    Rellena NaN iniciales con 50 (neutral).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # Cuando avg_loss == 0 (solo ganancias), RSI = 100; NaN iniciales → 50 (neutral)
    rsi_val = np.where(
        avg_loss == 0,
        100.0,
        100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan))),
    )
    result = pd.Series(rsi_val, index=series.index, dtype=float)
    return result.fillna(50.0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, Signal line y Histograma.

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger: upper, middle (SMA), lower.

    Returns:
        (upper, middle, lower)
    """
    middle = sma(series, window)
    std = series.rolling(window=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bb_pct_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """%B de Bollinger: posición del precio dentro de las bandas.

    0 = en la banda inferior, 1 = en la superior, 0.5 = en la media.
    Valores fuera de [0,1] indican precio más allá de las bandas.
    """
    upper, middle, lower = bollinger_bands(series, window, num_std)
    band_width = upper - lower
    return ((series - lower) / band_width).where(band_width > 0, 0.5)


def bb_width(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Ancho de las bandas normalizado por la media (proxy de volatilidad)."""
    upper, middle, lower = bollinger_bands(series, window, num_std)
    return ((upper - lower) / middle).where(middle > 0, np.nan)


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — medida de volatilidad absoluta.

    Requiere columnas: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR expresado como porcentaje del precio de cierre (volatilidad relativa)."""
    return atr(df, period) / df["close"]


# ---------------------------------------------------------------------------
# Volumen
# ---------------------------------------------------------------------------

def volume_ratio(series: pd.Series, window: int = 20) -> pd.Series:
    """Ratio volumen actual / promedio móvil de volumen.

    > 1.5 = volumen alto (confirma movimiento), < 0.5 = volumen bajo (señal débil).
    """
    avg = series.rolling(window=window).mean()
    return (series / avg).where(avg > 0, 1.0)


# ---------------------------------------------------------------------------
# Función de enriquecimiento completo del DataFrame
# ---------------------------------------------------------------------------

def enrich(
    df: pd.DataFrame,
    fast_ema: int = 9,
    slow_ema: int = 21,
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Añade todos los indicadores al DataFrame OHLCV.

    Columnas añadidas:
        ema_fast, ema_slow, rsi,
        macd_line, macd_signal, macd_hist,
        bb_upper, bb_mid, bb_lower, bb_pct_b, bb_width,
        atr, atr_pct, volume_ratio
    """
    out = df.copy()
    close = out["close"]

    out["ema_fast"] = ema(close, fast_ema)
    out["ema_slow"] = ema(close, slow_ema)
    out["rsi"] = rsi(close, rsi_period)

    ml, sl, mh = macd(close, macd_fast, macd_slow, macd_signal)
    out["macd_line"] = ml
    out["macd_signal"] = sl
    out["macd_hist"] = mh

    upper, mid, lower = bollinger_bands(close, bb_window, bb_std)
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["bb_pct_b"] = bb_pct_b(close, bb_window, bb_std)
    out["bb_width"] = bb_width(close, bb_window, bb_std)

    out["atr"] = atr(out, atr_period)
    out["atr_pct"] = atr_pct(out, atr_period)
    out["volume_ratio"] = volume_ratio(out["volume"], vol_window)

    return out
