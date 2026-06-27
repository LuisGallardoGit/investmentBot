"""Estrategia de reversión a la media usando Bollinger Bands.

Lógica:
  BUY  — precio cruza por encima de la banda inferior de Bollinger (%B < 0.2)
          con RSI sobrevendido, sentimiento y macro no extremos.
  SELL — precio cruza por debajo de la banda superior (%B > 0.8)
          con RSI sobrecomprado, o condiciones adversas externas.
  HOLD — precio dentro de las bandas, sin señal fuerte.

Esta estrategia funciona mejor en mercados laterales/rangos.
El Combined la pondera junto con TrendFollow para cubrir ambos regímenes.
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


class MeanReversionStrategy(Strategy):
    """Bollinger %B + RSI para reversión a la media."""

    name = "mean_reversion"

    def __init__(
        self,
        bb_buy_threshold: float = 0.2,
        bb_sell_threshold: float = 0.8,
        rsi_overbought: float = 65.0,
        rsi_oversold: float = 35.0,
        sentiment_bearish_threshold: float = 0.30,
        require_volume_confirmation: bool = True,
        min_volume_ratio: float = 1.0,
    ) -> None:
        self.bb_buy_threshold = bb_buy_threshold
        self.bb_sell_threshold = bb_sell_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sentiment_bearish_threshold = sentiment_bearish_threshold
        self.require_volume_confirmation = require_volume_confirmation
        self.min_volume_ratio = min_volume_ratio

    def signal(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> Signal:
        bb_pct = curr.get("bb_pct_b", _NAN)
        bb_pct_prev = prev.get("bb_pct_b", _NAN)
        rsi_val = curr.get("rsi", _NAN)

        if _is_nan(bb_pct) or _is_nan(rsi_val):
            return Signal.HOLD

        bb_pct = float(bb_pct)
        rsi_val = float(rsi_val)
        is_bearish_extreme = sentiment < self.sentiment_bearish_threshold
        is_macro_danger = macro_risk == "high_risk"

        # Confirmación de volumen (spike = liquidez real)
        vol_ratio = curr.get("volume_ratio", _NAN)
        volume_spike = _is_nan(vol_ratio) or (
            not self.require_volume_confirmation or float(vol_ratio) >= self.min_volume_ratio
        )

        # BUY: precio rebota desde la banda inferior
        # El precio debe haber estado por debajo y ahora cruzar hacia arriba
        was_below = not _is_nan(bb_pct_prev) and float(bb_pct_prev) < self.bb_buy_threshold
        now_recovering = bb_pct > self.bb_buy_threshold or bb_pct < self.bb_buy_threshold

        price_near_lower = bb_pct < self.bb_buy_threshold
        if (
            price_near_lower
            and rsi_val < self.rsi_oversold
            and volume_spike
            and not is_bearish_extreme
            and not is_macro_danger
        ):
            return Signal.BUY

        # SELL: precio en banda superior
        price_near_upper = bb_pct > self.bb_sell_threshold
        if price_near_upper and rsi_val > self.rsi_overbought:
            return Signal.SELL

        # SELL defensivo por condiciones externas adversas
        if is_macro_danger and bb_pct > 0.5:
            return Signal.SELL
        if is_bearish_extreme and bb_pct > 0.6:
            return Signal.SELL

        return Signal.HOLD
