"""Interfaz base para todas las estrategias."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy(ABC):
    """Contrato mínimo que toda estrategia debe cumplir.

    Una estrategia recibe dos filas consecutivas de un DataFrame enriquecido
    (con todos los indicadores ya calculados) más contexto externo, y devuelve
    una señal BUY / SELL / HOLD.
    """

    name: str = "base"

    @abstractmethod
    def signal(
        self,
        prev: pd.Series,
        curr: pd.Series,
        sentiment: float = 0.5,
        macro_risk: str = "normal",
    ) -> Signal:
        """Evalúa dos barras consecutivas y devuelve una señal."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
