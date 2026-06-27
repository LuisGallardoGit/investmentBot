"""Análisis macroeconómico usando la API de FRED.

Series usadas:
  T10Y2Y   — Yield curve spread (10Y - 2Y Treasury). Negativo = curva invertida = riesgo alto.
  DEXCOUS  — Tipo de cambio USD/COP (dólares por peso colombiano × 1000).
             FRED lo reporta como COP por USD (ej. 4200 = 1 USD = 4200 COP).

Requiere env: FRED_API_KEY
"""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred_latest(series_id: str, api_key: str, timeout: int = 10) -> float | None:
    """Obtiene el último valor disponible de una serie FRED.

    Retorna None si el valor es '.' (festivo/sin dato) o si hay error de red.
    """
    url = (
        f"{FRED_BASE}?series_id={series_id}"
        f"&api_key={api_key}&file_type=json&sort_order=desc&limit=5"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        # FRED puede devolver '.' en festivos — buscar el primer valor numérico
        for obs in observations:
            val_str = obs.get("value", ".")
            if val_str != ".":
                return float(val_str)
        return None
    except Exception as exc:
        log.error("Error consultando FRED serie %s: %s", series_id, exc)
        return None


class MacroState:
    """Resultado del análisis macro con todos los indicadores."""

    def __init__(
        self,
        risk_level: str = "normal",
        yield_curve: float | None = None,
        usd_cop: float | None = None,
        usd_cop_30d_change_pct: float | None = None,
    ) -> None:
        self.risk_level = risk_level          # "normal" | "high_risk"
        self.yield_curve = yield_curve        # T10Y2Y spread en %
        self.usd_cop = usd_cop                # Tipo de cambio (COP por 1 USD)
        self.usd_cop_30d_change_pct = usd_cop_30d_change_pct  # Variación 30d del USD/COP

    def __repr__(self) -> str:
        return (
            f"MacroState(risk={self.risk_level}, "
            f"T10Y2Y={self.yield_curve}, "
            f"USD/COP={self.usd_cop}, "
            f"FX_30d_chg={self.usd_cop_30d_change_pct}%)"
        )


class MacroAnalyzer:
    """Análisis macroeconómico con contexto Colombia.

    Indicadores:
    - Yield curve (T10Y2Y): proxy de riesgo de recesión global.
    - USD/COP (DEXCOUS): riesgo cambiario para inversores colombianos.
      Una devaluación fuerte del peso amplifica ganancias en USD; una
      revaluación las reduce. Se reporta como factor informativo.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._last_state = MacroState()

    def fetch_macro_state(self) -> str:
        """Actualiza el estado macro y devuelve el risk_level ("normal" | "high_risk").

        Compatible con la interfaz anterior del engine.
        """
        state = self.fetch_full_state()
        self._last_state = state
        return state.risk_level

    def fetch_full_state(self) -> MacroState:
        """Retorna un MacroState completo con todos los indicadores."""
        if not self.api_key:
            log.warning("FRED_API_KEY no configurado — contexto macro neutral.")
            return MacroState()

        risk_level = "normal"

        # 1. Yield curve (T10Y2Y)
        yield_curve = _fred_latest("T10Y2Y", self.api_key)
        if yield_curve is not None:
            log.info("FRED T10Y2Y (Yield Curve Spread): %.2f%%", yield_curve)
            if yield_curve < 0:
                risk_level = "high_risk"
                log.warning("Curva de tipos invertida (%.2f%%). Riesgo macro ALTO.", yield_curve)
        else:
            log.warning("Sin dato de yield curve — se mantiene riesgo anterior.")

        # 2. USD/COP — últimas 30 observaciones para calcular variación mensual
        usd_cop, usd_cop_30d_change = self._fetch_usd_cop()

        state = MacroState(
            risk_level=risk_level,
            yield_curve=yield_curve,
            usd_cop=usd_cop,
            usd_cop_30d_change_pct=usd_cop_30d_change,
        )
        log.info("Estado macro: %s", state)
        return state

    @property
    def last_usd_cop(self) -> float | None:
        """Tipo de cambio USD/COP del último fetch."""
        return self._last_state.usd_cop

    @property
    def last_fx_change_pct(self) -> float | None:
        """Variación 30d del USD/COP del último fetch."""
        return self._last_state.usd_cop_30d_change_pct

    # ------------------------------------------------------------------

    def _fetch_usd_cop(self) -> tuple[float | None, float | None]:
        """Obtiene el tipo de cambio USD/COP y su variación mensual.

        FRED serie DEXCOUS: pesos colombianos por 1 USD.
        Retorna (usd_cop_actual, variacion_30d_pct) o (None, None) si hay error.
        """
        url = (
            f"{FRED_BASE}?series_id=DEXCOUS"
            f"&api_key={self.api_key}&file_type=json&sort_order=desc&limit=30"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            observations = [
                obs for obs in resp.json().get("observations", [])
                if obs.get("value", ".") != "."
            ]
            if not observations:
                log.warning("Sin datos USD/COP de FRED.")
                return None, None

            latest = float(observations[0]["value"])
            log.info("FRED USD/COP (DEXCOUS): %.2f COP/USD", latest)

            change_pct = None
            if len(observations) >= 20:
                oldest = float(observations[-1]["value"])
                if oldest > 0:
                    change_pct = round((latest - oldest) / oldest * 100, 2)
                    direction = "devaluación" if change_pct > 0 else "revaluación"
                    log.info(
                        "USD/COP variación ~30d: %+.2f%% (%s del peso colombiano)",
                        change_pct,
                        direction,
                    )

            return latest, change_pct

        except Exception as exc:
            log.error("Error consultando FRED USD/COP: %s", exc)
            return None, None
