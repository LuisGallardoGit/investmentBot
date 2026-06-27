"""Módulo de riesgo cambiario COP/USD para traders colombianos.

Contexto:
  - Un inversionista colombiano que opera mercados USD tiene exposición doble:
    ganancia en el activo + movimiento del peso.
  - Si el COP se aprecia (USD/COP baja), las ganancias en USD valen MENOS en pesos.
  - Si el COP se deprecia, las ganancias en USD valen MÁS en pesos.
  - Volatilidad alta en COP/USD es un riesgo adicional que debe modular
    el tamaño de posición o bloquear entradas.

Fuentes de datos:
  - Tasa spot: FRED DEXCOUS (USD/COP) — ya disponible en MacroAnalyzer
  - Cálculo de PnL en COP a partir de PnL en USD × tasa_entrada / tasa_actual

Reglas de negocio:
  - FX vol > fx_vol_threshold (default 5% en 30d) → reducir exposición
  - FX vol > 10% → bloquear nuevas entradas (solo mantener posiciones abiertas)
  - PnL en COP se registra en cada trade para reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class FXState:
    """Estado del mercado FX COP/USD en el momento del análisis."""

    usd_cop: float          # Tasa actual USD/COP (ej: 4200.0 = 1 USD = 4200 COP)
    change_30d_pct: float   # Variación porcentual de la tasa en 30 días
    risk_level: str         # "normal" | "elevated" | "high"
    source: str = "FRED/DEXCOUS"

    @property
    def cop_per_usd(self) -> float:
        """Alias semántico."""
        return self.usd_cop

    def __str__(self) -> str:
        sign = "+" if self.change_30d_pct >= 0 else ""
        return (
            f"USD/COP={self.usd_cop:.0f} "
            f"({sign}{self.change_30d_pct:.1f}% 30d) "
            f"[{self.risk_level}]"
        )


@dataclass
class FXAdjustedPnL:
    """PnL de un trade expresado en ambas monedas."""

    pnl_usd: float          # Ganancia/pérdida en dólares
    entry_rate: float       # USD/COP al entrar al trade
    exit_rate: float        # USD/COP al cerrar el trade
    pnl_cop: float          # Ganancia/pérdida en pesos colombianos
    fx_impact_cop: float    # Impacto del movimiento FX en COP (independiente del activo)

    @classmethod
    def calculate(
        cls,
        pnl_usd: float,
        entry_usd_cop: float,
        exit_usd_cop: float,
        position_usd: float,
    ) -> "FXAdjustedPnL":
        """
        Calcula el PnL ajustado por FX.

        Método:
          - Valor de entrada en COP  = position_usd × entry_usd_cop
          - Valor de salida en COP   = (position_usd + pnl_usd) × exit_usd_cop
          - PnL en COP               = valor_salida_cop − valor_entrada_cop
          - Impacto FX en COP        = position_usd × (exit_usd_cop − entry_usd_cop)
        """
        entry_value_cop = position_usd * entry_usd_cop
        exit_value_cop = (position_usd + pnl_usd) * exit_usd_cop
        pnl_cop = exit_value_cop - entry_value_cop
        fx_impact_cop = position_usd * (exit_usd_cop - entry_usd_cop)
        return cls(
            pnl_usd=pnl_usd,
            entry_rate=entry_usd_cop,
            exit_rate=exit_usd_cop,
            pnl_cop=pnl_cop,
            fx_impact_cop=fx_impact_cop,
        )


class FXRiskManager:
    """Gestiona el riesgo cambiario COP/USD para un portafolio en USD.

    Parámetros
    ----------
    fx_vol_caution_pct : float
        % de cambio en 30d que activa modo cautela (default 5%).
        En cautela el tamaño de posición se reduce a la mitad.
    fx_vol_block_pct : float
        % de cambio en 30d que bloquea nuevas entradas (default 10%).
    """

    def __init__(
        self,
        fx_vol_caution_pct: float = 5.0,
        fx_vol_block_pct: float = 10.0,
    ) -> None:
        self.fx_vol_caution_pct = fx_vol_caution_pct
        self.fx_vol_block_pct = fx_vol_block_pct
        self._current_rate: Optional[float] = None
        self._entry_rates: dict[str, float] = {}   # symbol → tasa USD/COP al entrar

    # ------------------------------------------------------------------
    # Estado FX
    # ------------------------------------------------------------------

    def evaluate(self, usd_cop: float, change_30d_pct: float) -> FXState:
        """Evalúa el riesgo cambiario actual y retorna un FXState.

        El signo de change_30d_pct sigue la convención USD/COP:
          - Positivo (+) = COP se depreció (más pesos por dólar) → favorable para el inversor
          - Negativo (-) = COP se apreció (menos pesos por dólar) → desfavorable
        Lo que importa para el riesgo es la VOLATILIDAD (valor absoluto).
        """
        self._current_rate = usd_cop
        abs_change = abs(change_30d_pct)

        if abs_change >= self.fx_vol_block_pct:
            risk_level = "high"
        elif abs_change >= self.fx_vol_caution_pct:
            risk_level = "elevated"
        else:
            risk_level = "normal"

        state = FXState(
            usd_cop=usd_cop,
            change_30d_pct=change_30d_pct,
            risk_level=risk_level,
        )
        log.info("FX: %s", state)
        return state

    def allows_new_entry(self, fx_state: FXState) -> bool:
        """True si el riesgo FX permite abrir nuevas posiciones."""
        if fx_state.risk_level == "high":
            log.warning(
                "FX block: cambio 30d=%.1f%% >= %.1f%% — no se abren nuevas posiciones.",
                fx_state.change_30d_pct,
                self.fx_vol_block_pct,
            )
            return False
        return True

    def position_size_multiplier(self, fx_state: FXState) -> float:
        """
        Factor (0.0–1.0) que ajusta el tamaño de posición según riesgo FX.

        - normal   → 1.0  (tamaño completo)
        - elevated → 0.5  (mitad de posición)
        - high     → 0.0  (bloqueado)
        """
        return {"normal": 1.0, "elevated": 0.5, "high": 0.0}.get(
            fx_state.risk_level, 1.0
        )

    # ------------------------------------------------------------------
    # Tracking de tasas por símbolo
    # ------------------------------------------------------------------

    def record_entry(self, symbol: str, usd_cop_at_entry: float) -> None:
        """Registra la tasa USD/COP al momento de comprar un símbolo."""
        self._entry_rates[symbol] = usd_cop_at_entry
        log.debug("FX entrada %s: USD/COP=%.0f", symbol, usd_cop_at_entry)

    def record_exit(self, symbol: str) -> None:
        """Elimina el registro de entrada al cerrar una posición."""
        self._entry_rates.pop(symbol, None)

    def entry_rate(self, symbol: str) -> Optional[float]:
        """Retorna la tasa de entrada registrada para un símbolo, o None."""
        return self._entry_rates.get(symbol)

    # ------------------------------------------------------------------
    # PnL ajustado por FX
    # ------------------------------------------------------------------

    def adjusted_pnl(
        self,
        symbol: str,
        pnl_usd: float,
        position_usd: float,
        current_usd_cop: float,
    ) -> FXAdjustedPnL:
        """Calcula el PnL en COP para un trade cerrado.

        Usa la tasa registrada en record_entry(); si no existe, asume que
        la tasa de entrada es la tasa actual (impacto FX = 0).
        """
        entry_rate = self._entry_rates.get(symbol, current_usd_cop)
        return FXAdjustedPnL.calculate(
            pnl_usd=pnl_usd,
            entry_usd_cop=entry_rate,
            exit_usd_cop=current_usd_cop,
            position_usd=position_usd,
        )

    # ------------------------------------------------------------------
    # Resumen de portafolio en COP
    # ------------------------------------------------------------------

    def portfolio_value_cop(self, portfolio_value_usd: float) -> Optional[float]:
        """Convierte el valor del portafolio a COP usando la tasa más reciente.

        Retorna None si no hay tasa registrada.
        """
        if self._current_rate is None:
            return None
        return portfolio_value_usd * self._current_rate

    def fx_summary(self, portfolio_value_usd: float) -> dict:
        """Resumen FX para incluir en reportes."""
        cop_value = self.portfolio_value_cop(portfolio_value_usd)
        return {
            "usd_value": portfolio_value_usd,
            "usd_cop_rate": self._current_rate,
            "cop_value": cop_value,
            "open_positions_entry_rates": dict(self._entry_rates),
        }
