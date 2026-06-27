"""Interfaz de broker. La capa concreta es PaperBroker.

Cualquier broker real (Alpaca, IBKR, etc.) debe implementar esta interfaz.
El pipeline solo conoce esta abstraccion para que los guardrails de modo paper
queden centralizados en la fabrica/orquestador.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    timestamp: datetime
    reference_price: float  # precio usado por la estrategia para decidir


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0

    @property
    def signed_qty(self) -> float:
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def cash_flow(self) -> float:
        # BUY consume cash (negativo), SELL libera cash (positivo).
        sign = -1.0 if self.side is OrderSide.BUY else 1.0
        return sign * self.quantity * self.price - self.commission


class Broker(ABC):
    """Contrato minimo que cualquier broker (paper o real) debe cumplir."""

    is_paper: bool = True

    @abstractmethod
    def submit(self, order: Order) -> Fill:
        ...

    @abstractmethod
    def positions(self) -> dict[str, float]:
        ...

    @abstractmethod
    def cash(self) -> float:
        ...

    @abstractmethod
    def fills(self) -> Iterable[Fill]:
        ...
