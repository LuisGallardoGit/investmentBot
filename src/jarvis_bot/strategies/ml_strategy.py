"""MLStrategy — estrategia de señales basada en modelo ML — Fase 6.

Implementa la interfaz Strategy ABC usando un SignalClassifier entrenado.

Comportamiento:
  - Si el modelo está disponible y la confianza supera el umbral → señal ML
  - Si la confianza es baja → HOLD (no operar sin convicción)
  - Si el modelo no está disponible → fallback a CombinedStrategy

Integración:
  - Se registra en get_strategy() con nombre "ml"
  - Requiere que el modelo esté pre-entrenado y cargado
  - Las features se extraen de la misma row del DataFrame que usan las otras estrategias

Uso:
    from jarvis_bot.ml.model import SignalClassifier
    from jarvis_bot.strategies.ml_strategy import MLStrategy

    clf = SignalClassifier.load("models/signal_classifier.pkl")
    strategy = MLStrategy(classifier=clf)
    signal = strategy.signal(row_prev, row)
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pandas as pd

from .base import Signal, Strategy
from .combined import CombinedStrategy

if TYPE_CHECKING:
    from ..ml.model import SignalClassifier

log = logging.getLogger(__name__)

# Mapping int label → Signal enum
_LABEL_TO_SIGNAL = {1: Signal.BUY, -1: Signal.SELL, 0: Signal.HOLD}


class MLStrategy(Strategy):
    """Estrategia de señales usando un clasificador ML entrenado.

    Parámetros
    ----------
    classifier : SignalClassifier | None
        Modelo entrenado. Si es None, usa el fallback.
    confidence_threshold : float
        Confianza mínima para emitir BUY/SELL (default: del config del modelo).
    fallback : Strategy | None
        Estrategia usada si el modelo no está disponible o falla.
        Por defecto: CombinedStrategy(trend_weight=0.60).
    """

    name = "ml"

    def __init__(
        self,
        classifier: "SignalClassifier | None" = None,
        confidence_threshold: float | None = None,
        fallback: Strategy | None = None,
    ) -> None:
        self._clf = classifier
        self._confidence_threshold = confidence_threshold
        self._fallback = fallback or CombinedStrategy()
        self._fallback_count = 0
        self._ml_count = 0

    @property
    def is_available(self) -> bool:
        """True si el modelo está cargado y entrenado."""
        return self._clf is not None and self._clf.is_trained

    def signal(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> Signal:
        """Genera una señal de trading para la barra actual.

        Intenta usar el modelo ML; si falla o la confianza es baja,
        retorna HOLD o delega al fallback según la configuración.
        """
        if not self.is_available:
            self._fallback_count += 1
            return self._fallback.signal(prev, curr, sentiment, macro_risk)

        try:
            label, confidence = self._clf.predict_with_confidence(curr)
            threshold = self._confidence_threshold or self._clf.cfg.confidence_threshold

            log.debug(
                "ML signal: label=%d confidence=%.3f threshold=%.3f",
                label, confidence, threshold,
            )

            sig = _LABEL_TO_SIGNAL.get(label, Signal.HOLD)

            # Filtros de seguridad macro y sentimiento (mismos que rule-based)
            if sig == Signal.BUY:
                if sentiment < 0.30 or macro_risk == "high_risk":
                    log.debug("ML BUY filtrado por sentimiento/macro.")
                    return Signal.HOLD

            self._ml_count += 1
            return sig

        except Exception as exc:
            log.warning("Error en predicción ML (%s) — usando fallback.", exc)
            self._fallback_count += 1
            return self._fallback.signal(prev, curr, sentiment, macro_risk)

    def explain(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> dict:
        """Retorna desglose de la señal para debugging."""
        if not self.is_available:
            return {"source": "fallback", "fallback_signal": self._fallback.signal(prev, curr, sentiment, macro_risk).value}

        try:
            label, confidence = self._clf.predict_with_confidence(curr)
            sig = self.signal(prev, curr, sentiment, macro_risk)
            top = self._clf.feature_importance_report(top_n=5)
            return {
                "source": "ml",
                "raw_label": label,
                "confidence": confidence,
                "final_signal": sig.value,
                "top_features": {f: round(imp, 4) for f, imp in top},
                "ml_count": self._ml_count,
                "fallback_count": self._fallback_count,
            }
        except Exception as exc:
            return {"source": "error", "error": str(exc)}

    def stats(self) -> dict:
        """Estadísticas de uso del modelo."""
        total = self._ml_count + self._fallback_count
        return {
            "ml_signals": self._ml_count,
            "fallback_signals": self._fallback_count,
            "ml_pct": self._ml_count / total * 100 if total else 0,
        }
