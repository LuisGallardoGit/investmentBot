"""Estrategia combinada (ensemble) de TrendFollow + MeanReversion.

Lógica de votación:
  - Cada sub-estrategia emite un voto: BUY=+1, SELL=-1, HOLD=0.
  - Los votos se ponderan (por defecto 60% trend, 40% mean_reversion).
  - Score > threshold_buy  → BUY
  - Score < threshold_sell → SELL
  - En caso contrario      → HOLD

Permite aprovechar mercados tendenciales (trend_follow) y laterales
(mean_reversion) sin depender de un solo régimen.
"""

from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy
from .mean_reversion import MeanReversionStrategy
from .trend_follow import TrendFollowStrategy


class CombinedStrategy(Strategy):
    """Ensemble ponderado de TrendFollow y MeanReversion."""

    name = "combined"

    def __init__(
        self,
        trend_weight: float = 0.60,
        reversion_weight: float = 0.40,
        threshold_buy: float = 0.35,
        threshold_sell: float = -0.35,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_volume_ratio: float = 0.8,
        sentiment_bearish_threshold: float = 0.35,
    ) -> None:
        total = trend_weight + reversion_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0, suma actual: {total}")

        self.trend_weight = trend_weight
        self.reversion_weight = reversion_weight
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell

        self._trend = TrendFollowStrategy(
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            min_volume_ratio=min_volume_ratio,
            sentiment_bearish_threshold=sentiment_bearish_threshold,
        )
        self._reversion = MeanReversionStrategy(
            rsi_overbought=rsi_overbought - 5,
            rsi_oversold=rsi_oversold + 5,
            sentiment_bearish_threshold=sentiment_bearish_threshold,
        )

    def signal(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> Signal:
        _VOTE = {Signal.BUY: 1, Signal.HOLD: 0, Signal.SELL: -1}

        trend_vote = _VOTE[self._trend.signal(prev, curr, sentiment, macro_risk)]
        reversion_vote = _VOTE[self._reversion.signal(prev, curr, sentiment, macro_risk)]

        score = trend_vote * self.trend_weight + reversion_vote * self.reversion_weight

        if score >= self.threshold_buy:
            return Signal.BUY
        if score <= self.threshold_sell:
            return Signal.SELL
        return Signal.HOLD

    def explain(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> dict:
        """Retorna el desglose de votos para debugging/reporting."""
        trend_sig = self._trend.signal(prev, curr, sentiment, macro_risk)
        reversion_sig = self._reversion.signal(prev, curr, sentiment, macro_risk)
        _VOTE = {Signal.BUY: 1, Signal.HOLD: 0, Signal.SELL: -1}
        score = _VOTE[trend_sig] * self.trend_weight + _VOTE[reversion_sig] * self.reversion_weight
        final = self.signal(prev, curr, sentiment, macro_risk)
        return {
            "trend_signal": trend_sig.value,
            "reversion_signal": reversion_sig.value,
            "trend_weight": self.trend_weight,
            "reversion_weight": self.reversion_weight,
            "score": round(score, 3),
            "final_signal": final.value,
        }
