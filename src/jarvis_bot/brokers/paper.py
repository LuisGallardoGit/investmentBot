"""Broker simulado en memoria.

No conecta a ningun exchange. Ejecuta ordenes contra el precio de referencia
con un slippage parametrizable y registra cada fill para auditoria.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .base import Broker, Fill, Order, OrderSide

log = logging.getLogger(__name__)


class InsufficientCashError(RuntimeError):
    pass


class InsufficientPositionError(RuntimeError):
    pass


class PaperBroker(Broker):
    is_paper = True

    def __init__(
        self,
        starting_cash: float,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        self._cash = float(starting_cash)
        self._positions: dict[str, float] = {}
        self._fills: list[Fill] = []
        self._commission = float(commission_per_trade)
        self._slippage = float(slippage_pct)

    def submit(self, order: Order) -> Fill:
        if order.quantity <= 0:
            raise ValueError("quantity debe ser > 0")

        price = order.reference_price * (
            1 + self._slippage if order.side is OrderSide.BUY else 1 - self._slippage
        )

        if order.side is OrderSide.BUY:
            cost = price * order.quantity + self._commission
            if cost > self._cash + 1e-9:
                raise InsufficientCashError(
                    f"Cash {self._cash:.2f} insuficiente para {order.symbol} qty={order.quantity}"
                )
            self._cash -= cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.quantity
        else:
            current = self._positions.get(order.symbol, 0.0)
            if order.quantity > current + 1e-9:
                raise InsufficientPositionError(
                    f"Posicion {current} insuficiente para vender {order.quantity} de {order.symbol}"
                )
            self._cash += price * order.quantity - self._commission
            remaining = current - order.quantity
            if remaining <= 1e-9:
                self._positions.pop(order.symbol, None)
            else:
                self._positions[order.symbol] = remaining

        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=order.timestamp,
            commission=self._commission,
        )
        self._fills.append(fill)
        log.info(
            "[PAPER] %s %.4f %s @ %.2f cash=%.2f",
            order.side.value.upper(),
            order.quantity,
            order.symbol,
            price,
            self._cash,
        )
        return fill

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def fills(self) -> Iterable[Fill]:
        return list(self._fills)
