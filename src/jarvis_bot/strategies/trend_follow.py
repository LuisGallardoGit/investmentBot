"""Estrategia de seguimiento de tendencia (mejorada respecto al MVP).

Señales:
  BUY  — cruce EMA fast > slow + MACD histograma positivo + RSI no sobrecomprado
          + volumen confirmatorio + sentimiento y macro favorables.
  SELL — cruce EMA fast < slow O MACD hist negativo cruzando cero O RSI sobrecomprado
          O sentimiento extremo bajista O macro high_risk con tendencia bajista.
  HOLD — cualquier otro caso.

Mejoras sobre el MVP:
  - Confirmación MACD: reduce falsas señales en mercados laterales.
  - Confirmación de volumen: filtra rupturas de baja convicción.
  - ATR dinámico disponible para uso futuro en stop-loss real.
"""

from __future__ import annotations

import math

import pandas as pd

from .base import Signal, Strategy

_NAN = float("nan")


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


class TrendFollowStrategy(Strategy):
    """EMA crossover confirmado por MACD, RSI, volumen, sentimiento y macro."""

    name = "trend_follow"

    def __init__(
        self,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_volume_ratio: float = 0.8,
        sentiment_bearish_threshold: float = 0.35,
        sentiment_bullish_threshold: float = 0.65,
    ) -> None:
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.min_volume_ratio = min_volume_ratio
        self.sentiment_bearish_threshold = sentiment_bearish_threshold
        self.sentiment_bullish_threshold = sentiment_bullish_threshold

    def signal(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> Signal:
        # Guardia: indicadores mínimos presentes
        required = ("ema_fast", "ema_slow", "rsi")
        if any(_is_nan(curr.get(c, _NAN)) for c in required):
            return Signal.HOLD
        if any(_is_nan(prev.get(c, _NAN)) for c in ("ema_fast", "ema_slow")):
            return Signal.HOLD

        ema_crossed_up = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
        ema_crossed_down = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

        rsi_val = float(curr["rsi"])
        is_overbought = rsi_val > self.rsi_overbought
        is_oversold = rsi_val < self.rsi_oversold

        # Confirmación MACD (opcional: si no está calculado se ignora)
        macd_hist = curr.get("macd_hist", _NAN)
        macd_hist_prev = prev.get("macd_hist", _NAN)
        macd_ok = _is_nan(macd_hist) or float(macd_hist) > 0
        macd_turning_negative = (
            not _is_nan(macd_hist)
            and not _is_nan(macd_hist_prev)
            and float(macd_hist_prev) >= 0
            and float(macd_hist) < 0
        )

        # Confirmación de volumen
        vol_ratio = curr.get("volume_ratio", _NAN)
        volume_ok = _is_nan(vol_ratio) or float(vol_ratio) >= self.min_volume_ratio

        # Filtros externos
        is_bearish_extreme = sentiment < self.sentiment_bearish_threshold
        is_macro_danger = macro_risk == "high_risk"

        # --- BUY ---
        if (
            ema_crossed_up
            and not is_overbought
            and macd_ok
            and volume_ok
            and not is_bearish_extreme
            and not is_macro_danger
        ):
            return Signal.BUY

        # BUY refuerzo: RSI sobrevendido con tendencia alcista
        ema_uptrend = curr["ema_fast"] > curr["ema_slow"]
        if (
            is_oversold
            and ema_uptrend
            and macd_ok
            and not is_bearish_extreme
            and not is_macro_danger
        ):
            return Signal.BUY

        # --- SELL ---
        if ema_crossed_down:
            return Signal.SELL
        if is_overbought and macd_turning_negative:
            return Signal.SELL
        if is_bearish_extreme and rsi_val > 50 and ema_uptrend:
            return Signal.SELL
        if is_macro_danger and curr["ema_fast"] < curr["ema_slow"]:
            return Signal.SELL

        return Signal.HOLD
