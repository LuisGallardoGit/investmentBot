"""Análisis de régimen de mercado usando VIX y precio del petróleo WTI.

Series FRED usadas:
  VIXCLS       — CBOE Volatility Index (VIX), cierre diario
  DCOILWTICO   — West Texas Intermediate (WTI) crude oil, dólares/barril

El régimen escala el tamaño de posición SIN cambiar la señal del modelo ML.
Esto es correcto porque:
  - El modelo ML está entrenado en patrones de precio histórico
  - VIX y WTI operan a nivel "cuánto arriesgar", no "en qué dirección"

Tabla de efectos:

  VIX < 20  (calm)     → sin restricción      × 1.00
  VIX 20-25 (elevated) → reducir al 75%       × 0.75
  VIX 25-35 (high)     → reducir al 50%       × 0.50
  VIX > 35  (extreme)  → bloquear entradas    × 0.00

  WTI estable (>-10%)  → sin restricción      × 1.00
  WTI -10% a -20%      → reducir al 75%       × 0.75  (predictor de debilidad COP)
  WTI < -20%           → reducir al 50%       × 0.50

El multiplicador final = vix_mult × wti_mult (floored a 0.0).

Contexto Colombia:
  El WTI es el predictor adelantado más útil del COP. Colombia exporta crudo:
  cuando el WTI cae >10% en 30d, el peso colombiano suele debilitarse
  2-3 semanas después. Esto permite anticipar el bloqueo FX antes de que
  el módulo fx_risk lo active.

Requiere env: FRED_API_KEY
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# Resultado del análisis
# ---------------------------------------------------------------------------


@dataclass
class MarketRegime:
    """Estado completo del régimen de mercado."""

    vix: float | None = None
    vix_regime: str = "unknown"        # "calm" | "elevated" | "high" | "extreme" | "unknown"
    wti: float | None = None           # USD por barril
    wti_30d_change_pct: float | None = None
    wti_signal: str = "unknown"        # "normal" | "caution" | "warning" | "unknown"
    posture: str = "normal"            # "normal" | "reduced" | "defensive" | "blocked"
    position_multiplier: float = 1.0   # Factor de escala [0.0 – 1.0]

    def __repr__(self) -> str:
        vix_str = f"{self.vix:.1f}" if self.vix is not None else "N/A"
        wti_str = f"${self.wti:.1f}" if self.wti is not None else "N/A"
        chg_str = f"{self.wti_30d_change_pct:+.1f}%" if self.wti_30d_change_pct is not None else "N/A"
        return (
            f"MarketRegime(posture={self.posture}, mult={self.position_multiplier:.2f}, "
            f"VIX={vix_str} [{self.vix_regime}], "
            f"WTI={wti_str} 30d={chg_str} [{self.wti_signal}])"
        )

    @property
    def blocks_new_entries(self) -> bool:
        return self.position_multiplier == 0.0

    @property
    def is_restricted(self) -> bool:
        return self.position_multiplier < 1.0


# ---------------------------------------------------------------------------
# Analizador
# ---------------------------------------------------------------------------


class MarketRegimeAnalyzer:
    """Determina el régimen de mercado a partir de VIX y WTI (FRED).

    Diseño: sin estado mutable entre ejecuciones — cada llamada a fetch_regime()
    hace sus propias consultas y retorna un MarketRegime fresco.
    """

    # Umbrales VIX
    VIX_CALM: float = 20.0
    VIX_ELEVATED: float = 25.0
    VIX_EXTREME: float = 35.0

    # Umbrales WTI (cambio porcentual en 30 días)
    WTI_CAUTION_PCT: float = -10.0
    WTI_WARNING_PCT: float = -20.0

    def __init__(
        self,
        api_key: str | None = None,
        enabled: bool = True,
        vix_calm: float | None = None,
        vix_elevated: float | None = None,
        vix_extreme: float | None = None,
        wti_caution_pct: float | None = None,
        wti_warning_pct: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.enabled = enabled
        # Permitir override de umbrales desde config
        if vix_calm is not None:
            self.VIX_CALM = vix_calm
        if vix_elevated is not None:
            self.VIX_ELEVATED = vix_elevated
        if vix_extreme is not None:
            self.VIX_EXTREME = vix_extreme
        if wti_caution_pct is not None:
            self.WTI_CAUTION_PCT = wti_caution_pct
        if wti_warning_pct is not None:
            self.WTI_WARNING_PCT = wti_warning_pct

        self._last_regime: MarketRegime = MarketRegime()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fetch_regime(self) -> MarketRegime:
        """Consulta FRED y retorna el régimen de mercado actual.

        Si el módulo está desactivado o sin API key, retorna régimen neutral
        (position_multiplier=1.0) sin bloquear el pipeline.
        """
        if not self.enabled:
            log.debug("MarketRegimeAnalyzer desactivado — régimen neutral.")
            return MarketRegime()

        if not self.api_key:
            log.warning("FRED_API_KEY no configurado — régimen de mercado neutral (sin VIX/WTI).")
            return MarketRegime()

        vix = self._fetch_latest("VIXCLS")
        wti, wti_30d_change = self._fetch_wti()

        vix_regime = self._classify_vix(vix)
        wti_signal = self._classify_wti(wti_30d_change)

        position_multiplier, posture = self._compute_posture(vix_regime, wti_signal)

        if vix is not None:
            log.info("FRED VIX (VIXCLS): %.1f → régimen '%s'", vix, vix_regime)
        if wti is not None:
            log.info(
                "FRED WTI (DCOILWTICO): $%.1f/bbl (30d: %s%%) → señal '%s'",
                wti,
                f"{wti_30d_change:+.1f}" if wti_30d_change is not None else "N/A",
                wti_signal,
            )

        regime = MarketRegime(
            vix=vix,
            vix_regime=vix_regime,
            wti=wti,
            wti_30d_change_pct=wti_30d_change,
            wti_signal=wti_signal,
            posture=posture,
            position_multiplier=position_multiplier,
        )
        log.info("Régimen de mercado: %s", regime)

        if regime.blocks_new_entries:
            log.warning(
                "⚠️  RÉGIMEN EXTREMO (VIX=%.1f) — nuevas entradas BLOQUEADAS.",
                vix or 0,
            )
        elif regime.is_restricted:
            log.warning(
                "⚠️  Régimen restrictivo (%s) — posición escalada a %.0f%%.",
                posture,
                position_multiplier * 100,
            )

        self._last_regime = regime
        return regime

    @property
    def last_regime(self) -> MarketRegime:
        return self._last_regime

    # ------------------------------------------------------------------
    # Clasificación interna
    # ------------------------------------------------------------------

    def _classify_vix(self, vix: float | None) -> str:
        if vix is None:
            return "unknown"
        if vix < self.VIX_CALM:
            return "calm"
        if vix < self.VIX_ELEVATED:
            return "elevated"
        if vix < self.VIX_EXTREME:
            return "high"
        return "extreme"

    def _classify_wti(self, change_pct: float | None) -> str:
        if change_pct is None:
            return "unknown"
        if change_pct > self.WTI_CAUTION_PCT:
            return "normal"
        if change_pct > self.WTI_WARNING_PCT:
            return "caution"
        return "warning"

    def _compute_posture(
        self, vix_regime: str, wti_signal: str
    ) -> tuple[float, str]:
        """Calcula el multiplicador de posición y la etiqueta de postura."""
        # VIX extremo bloquea todo independientemente del WTI
        if vix_regime == "extreme":
            return 0.0, "blocked"

        vix_mult = {"calm": 1.0, "elevated": 0.75, "high": 0.50, "unknown": 1.0}.get(
            vix_regime, 1.0
        )
        wti_mult = {"normal": 1.0, "caution": 0.75, "warning": 0.50, "unknown": 1.0}.get(
            wti_signal, 1.0
        )

        combined = round(vix_mult * wti_mult, 2)

        if combined >= 0.90:
            posture = "normal"
        elif combined >= 0.55:
            posture = "reduced"
        else:
            posture = "defensive"

        return combined, posture

    # ------------------------------------------------------------------
    # FRED helpers
    # ------------------------------------------------------------------

    def _fetch_latest(self, series_id: str, limit: int = 5) -> float | None:
        """Obtiene el último valor numérico disponible de una serie FRED."""
        url = (
            f"{FRED_BASE}?series_id={series_id}"
            f"&api_key={self.api_key}&file_type=json&sort_order=desc&limit={limit}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            for obs in resp.json().get("observations", []):
                val = obs.get("value", ".")
                if val != ".":
                    return float(val)
            return None
        except Exception as exc:
            log.error("Error FRED %s: %s", series_id, exc)
            return None

    def _fetch_wti(self) -> tuple[float | None, float | None]:
        """Retorna (precio_actual, variacion_30d_pct) del WTI."""
        url = (
            f"{FRED_BASE}?series_id=DCOILWTICO"
            f"&api_key={self.api_key}&file_type=json&sort_order=desc&limit=30"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            obs_list = [
                o for o in resp.json().get("observations", [])
                if o.get("value", ".") != "."
            ]
            if not obs_list:
                return None, None
            latest = float(obs_list[0]["value"])
            change_pct = None
            if len(obs_list) >= 20:
                oldest = float(obs_list[-1]["value"])
                if oldest > 0:
                    change_pct = round((latest - oldest) / oldest * 100, 2)
            return latest, change_pct
        except Exception as exc:
            log.error("Error FRED DCOILWTICO: %s", exc)
            return None, None
