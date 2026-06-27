"""Gestion de riesgo y position sizing.

Reglas:
- Riesgo fijo por trade como % del equity actual.
- Stop loss obligatorio para dimensionar (Kelly-lite).
- Cap por posicion como % del equity total (evita all-in).
- Si el drawdown excede el limite, deja de abrir nuevas posiciones.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .. import stats_engine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingResult:
    quantity: float
    notional: float
    risk_amount: float
    approved: bool
    reason: str = ""


class RiskManager:
    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.01,
        stop_loss_pct: float = 0.05,
        max_drawdown_limit: float = 0.15,
        max_position_pct: float = 0.25,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital debe ser > 0")
        if not (0 < risk_per_trade < 1):
            raise ValueError("risk_per_trade debe estar en (0,1)")
        if not (0 < stop_loss_pct < 1):
            raise ValueError("stop_loss_pct debe estar en (0,1)")
        self.initial_capital = float(initial_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.stop_loss_pct = float(stop_loss_pct)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.max_position_pct = float(max_position_pct)

    def calculate_sharpe_ratio(self, returns) -> float:
        return stats_engine.sharpe_ratio(returns)

    def calculate_max_drawdown(self, equity_curve) -> float:
        return stats_engine.max_drawdown(equity_curve)

    def position_size(self, equity: float, price: float) -> SizingResult:
        """Devuelve la cantidad sugerida (acciones) para una entrada al precio dado.

        Formula: risk_amount = equity * risk_per_trade
                 qty_by_risk = risk_amount / (price * stop_loss_pct)
                 qty_by_cap  = (equity * max_position_pct) / price
                 qty = floor(min(qty_by_risk, qty_by_cap))
        """
        if equity <= 0 or price <= 0:
            return SizingResult(0.0, 0.0, 0.0, approved=False, reason="equity/price <= 0")

        risk_amount = equity * self.risk_per_trade
        qty_by_risk = risk_amount / (price * self.stop_loss_pct)
        qty_by_cap = (equity * self.max_position_pct) / price
        qty = math.floor(min(qty_by_risk, qty_by_cap))

        if qty <= 0:
            return SizingResult(0.0, 0.0, risk_amount, approved=False, reason="qty calculada < 1")

        notional = qty * price
        return SizingResult(
            quantity=float(qty),
            notional=float(notional),
            risk_amount=float(risk_amount),
            approved=True,
        )

    def drawdown_breach(self, equity_curve) -> bool:
        if not equity_curve:
            return False
        return self.calculate_max_drawdown(equity_curve) >= self.max_drawdown_limit
