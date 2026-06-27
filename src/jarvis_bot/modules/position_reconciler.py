"""Reconciliación de posiciones: estado local (PaperBroker) vs Alpaca real.

Problema:
  En paper trading, el PaperBroker mantiene posiciones en memoria. Si el bot
  se reinicia, las posiciones en memoria se pierden pero las de Alpaca persisten.
  También puede haber diferencias si una orden se fills parcialmente, o si
  se hacen ajustes manuales en la cuenta Alpaca.

Solución:
  PositionReconciler compara ambos estados y clasifica las diferencias:
    - MATCH: coinciden qty
    - LOCAL_ONLY: posición en memoria pero no en Alpaca (orden no llegó a hacer fill)
    - REMOTE_ONLY: posición en Alpaca pero no en memoria (bot reiniciado, orden manual)
    - MISMATCH: ambos tienen el símbolo pero la qty difiere

Acciones disponibles:
  - report_only: solo logea diferencias (default, no modifica nada)
  - sync_from_alpaca: sincroniza el estado local con lo que dice Alpaca (más confiable)

Nota: en Fase 4 paper trading solo, la "posición real" es la de Alpaca paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .alpaca_executor import AlpacaExecutor

log = logging.getLogger(__name__)


class DiscrepancyType(str, Enum):
    MATCH = "match"
    LOCAL_ONLY = "local_only"   # solo en memoria, no en Alpaca
    REMOTE_ONLY = "remote_only" # solo en Alpaca, no en memoria
    MISMATCH = "mismatch"       # ambos tienen pero qty difiere


@dataclass
class PositionDiscrepancy:
    symbol: str
    local_qty: float | None
    remote_qty: float | None
    discrepancy_type: DiscrepancyType
    remote_market_value: float | None = None
    remote_avg_entry_price: float | None = None

    def __str__(self) -> str:
        if self.discrepancy_type == DiscrepancyType.MATCH:
            return f"[OK] {self.symbol}: qty={self.local_qty}"
        if self.discrepancy_type == DiscrepancyType.LOCAL_ONLY:
            return f"[LOCAL_ONLY] {self.symbol}: memoria={self.local_qty}, Alpaca=0"
        if self.discrepancy_type == DiscrepancyType.REMOTE_ONLY:
            return f"[REMOTE_ONLY] {self.symbol}: Alpaca={self.remote_qty}, memoria=0"
        return f"[MISMATCH] {self.symbol}: memoria={self.local_qty}, Alpaca={self.remote_qty}"


@dataclass
class ReconciliationReport:
    local_positions: dict[str, float]
    remote_positions: dict[str, float]
    discrepancies: list[PositionDiscrepancy] = field(default_factory=list)
    synced: bool = False

    @property
    def has_discrepancies(self) -> bool:
        return any(d.discrepancy_type != DiscrepancyType.MATCH for d in self.discrepancies)

    @property
    def n_matches(self) -> int:
        return sum(1 for d in self.discrepancies if d.discrepancy_type == DiscrepancyType.MATCH)

    @property
    def n_issues(self) -> int:
        return len(self.discrepancies) - self.n_matches

    def summary(self) -> str:
        lines = [
            f"Reconciliación: {self.n_matches} OK, {self.n_issues} discrepancias",
        ]
        for d in self.discrepancies:
            if d.discrepancy_type != DiscrepancyType.MATCH:
                lines.append(f"  {d}")
        return "\n".join(lines)


class PositionReconciler:
    """Compara posiciones locales (PaperBroker) con las de Alpaca.

    Parámetros
    ----------
    executor : AlpacaExecutor
        Instancia del executor para consultar posiciones remotas.
    qty_tolerance : float
        Diferencia máxima de qty considerada aceptable (default 0.001 acciones).
    """

    def __init__(
        self,
        executor: "AlpacaExecutor",
        qty_tolerance: float = 0.001,
    ) -> None:
        self.executor = executor
        self.qty_tolerance = qty_tolerance

    def _fetch_remote_positions(self) -> dict[str, dict]:
        """Obtiene posiciones de Alpaca. Retorna dict[symbol → raw_position]."""
        if not self.executor.is_configured():
            log.debug("Alpaca no configurado — no se pueden obtener posiciones remotas.")
            return {}
        raw = self.executor.get_positions()
        result: dict[str, dict] = {}
        for pos in raw:
            sym = pos.get("symbol", "").upper()
            if sym:
                result[sym] = pos
        return result

    def reconcile(
        self,
        local_positions: dict[str, float],
        sync_from_alpaca: bool = False,
    ) -> ReconciliationReport:
        """Compara posiciones locales con Alpaca y genera un reporte.

        Parámetros
        ----------
        local_positions : dict[str, float]
            Posiciones actuales del PaperBroker (symbol → qty).
        sync_from_alpaca : bool
            Si True, retorna el estado correcto de Alpaca para que el caller
            actualice el estado local. No modifica nada directamente.

        Retorna
        -------
        ReconciliationReport con discrepancias clasificadas.
        """
        remote_raw = self._fetch_remote_positions()
        remote_positions: dict[str, float] = {
            sym: float(pos.get("qty") or pos.get("quantity") or 0)
            for sym, pos in remote_raw.items()
        }

        # Normalizar local: solo símbolos con qty > 0
        local = {sym: qty for sym, qty in local_positions.items() if qty > 0}

        all_symbols = set(local) | set(remote_positions)
        discrepancies: list[PositionDiscrepancy] = []

        for sym in sorted(all_symbols):
            local_qty = local.get(sym)
            remote_qty = remote_positions.get(sym)
            remote_info = remote_raw.get(sym, {})

            if local_qty is not None and remote_qty is not None:
                diff = abs(local_qty - remote_qty)
                if diff <= self.qty_tolerance:
                    dtype = DiscrepancyType.MATCH
                else:
                    dtype = DiscrepancyType.MISMATCH
            elif local_qty is not None:
                dtype = DiscrepancyType.LOCAL_ONLY
            else:
                dtype = DiscrepancyType.REMOTE_ONLY

            discrepancies.append(
                PositionDiscrepancy(
                    symbol=sym,
                    local_qty=local_qty,
                    remote_qty=remote_qty,
                    discrepancy_type=dtype,
                    remote_market_value=float(remote_info.get("market_value") or 0) or None,
                    remote_avg_entry_price=float(remote_info.get("avg_entry_price") or 0) or None,
                )
            )

        report = ReconciliationReport(
            local_positions=dict(local),
            remote_positions=dict(remote_positions),
            discrepancies=discrepancies,
            synced=sync_from_alpaca,
        )

        # Log del reporte
        if report.has_discrepancies:
            log.warning(report.summary())
        else:
            log.info(
                "Reconciliación OK: %d posiciones coinciden con Alpaca.",
                report.n_matches,
            )

        return report

    def positions_from_alpaca(self) -> dict[str, float]:
        """Retorna el estado de posiciones según Alpaca (para restaurar estado local).

        Útil al reiniciar el bot: en lugar de empezar con 0 posiciones, lee
        las posiciones reales de la cuenta.
        """
        remote_raw = self._fetch_remote_positions()
        result = {}
        for sym, pos in remote_raw.items():
            qty = float(pos.get("qty") or pos.get("quantity") or 0)
            if qty > 0:
                result[sym] = qty
        if result:
            log.info(
                "Posiciones cargadas desde Alpaca: %s",
                {s: f"{q:.4f}" for s, q in result.items()},
            )
        else:
            log.info("No hay posiciones abiertas en Alpaca.")
        return result

    def account_equity(self) -> float | None:
        """Retorna el equity total de la cuenta Alpaca. None si no disponible."""
        account = self.executor.get_account()
        equity = account.get("equity") or account.get("portfolio_value")
        if equity is not None:
            return float(equity)
        return None
