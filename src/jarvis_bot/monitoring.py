"""Monitoring y health checks del bot de trading — Fase 5.

Módulos:
  - HealthCheck: verifica conectividad con APIs externas antes de operar
  - PortfolioMonitor: métricas en tiempo real a partir de snapshots del pipeline
  - MetricsSnapshot: estado completo del portafolio en un momento dado

Uso típico:
    # Al inicio del pipeline
    health = HealthCheck()
    report = health.run_all()
    if not report.all_ok:
        alerts.pipeline_error(report.summary())

    # Al final del pipeline
    monitor = PortfolioMonitor(snapshots, initial_capital=10000.0)
    metrics = monitor.compute()
    alerts.daily_summary(**metrics.as_alert_kwargs())
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from .engine import PortfolioSnapshot

log = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT_S = 5


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@dataclass
class ServiceStatus:
    name: str
    ok: bool
    message: str
    latency_ms: float | None = None

    def __str__(self) -> str:
        icon = "✅" if self.ok else "❌"
        latency = f" ({self.latency_ms:.0f}ms)" if self.latency_ms is not None else ""
        return f"{icon} {self.name}{latency}: {self.message}"


@dataclass
class HealthReport:
    services: list[ServiceStatus] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def all_ok(self) -> bool:
        return all(s.ok for s in self.services)

    @property
    def n_ok(self) -> int:
        return sum(1 for s in self.services if s.ok)

    @property
    def n_failed(self) -> int:
        return len(self.services) - self.n_ok

    def summary(self) -> str:
        lines = [
            f"Health check {self.checked_at.strftime('%H:%M:%S UTC')}: "
            f"{self.n_ok}/{len(self.services)} servicios OK"
        ]
        lines += [f"  {s}" for s in self.services]
        return "\n".join(lines)


class HealthCheck:
    """Verifica conectividad con los servicios externos antes de operar.

    Checks incluidos:
      - Alpaca API (paper endpoint)
      - FRED API (si hay API key)
      - Telegram Bot API (si hay token)
    """

    def __init__(self) -> None:
        self.alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca_endpoint = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
        self.fred_key = os.getenv("FRED_API_KEY", "")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")

    def check_alpaca(self) -> ServiceStatus:
        """Verifica que Alpaca responde y las credenciales son válidas."""
        if not (self.alpaca_key and self.alpaca_secret):
            return ServiceStatus("Alpaca", ok=False, message="Credenciales no configuradas (ALPACA_API_KEY / ALPACA_SECRET_KEY)")

        url = f"{self.alpaca_endpoint}/v2/account"
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }
        try:
            t0 = datetime.utcnow()
            resp = requests.get(url, headers=headers, timeout=HEALTH_CHECK_TIMEOUT_S)
            latency = (datetime.utcnow() - t0).total_seconds() * 1000
            if resp.status_code == 200:
                data = resp.json()
                status_str = data.get("status", "?")
                return ServiceStatus(
                    "Alpaca",
                    ok=True,
                    message=f"OK — account status={status_str}",
                    latency_ms=latency,
                )
            return ServiceStatus(
                "Alpaca",
                ok=False,
                message=f"HTTP {resp.status_code}: {resp.text[:80]}",
                latency_ms=latency,
            )
        except requests.RequestException as exc:
            return ServiceStatus("Alpaca", ok=False, message=f"Error de red: {exc}")

    def check_fred(self) -> ServiceStatus:
        """Verifica acceso a FRED API (tasa COP/USD)."""
        if not self.fred_key:
            return ServiceStatus("FRED", ok=False, message="FRED_API_KEY no configurada")

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DEXCOUS",
            "api_key": self.fred_key,
            "file_type": "json",
            "limit": "1",
            "sort_order": "desc",
        }
        try:
            t0 = datetime.utcnow()
            resp = requests.get(url, params=params, timeout=HEALTH_CHECK_TIMEOUT_S)
            latency = (datetime.utcnow() - t0).total_seconds() * 1000
            if resp.status_code == 200:
                obs = resp.json().get("observations", [])
                last_date = obs[0]["date"] if obs else "?"
                return ServiceStatus(
                    "FRED",
                    ok=True,
                    message=f"OK — último dato DEXCOUS: {last_date}",
                    latency_ms=latency,
                )
            return ServiceStatus(
                "FRED",
                ok=False,
                message=f"HTTP {resp.status_code}: {resp.text[:80]}",
                latency_ms=latency,
            )
        except requests.RequestException as exc:
            return ServiceStatus("FRED", ok=False, message=f"Error de red: {exc}")

    def check_telegram(self) -> ServiceStatus:
        """Verifica que el bot de Telegram responde."""
        if not self.telegram_token:
            return ServiceStatus("Telegram", ok=False, message="TELEGRAM_BOT_TOKEN no configurado")

        url = f"https://api.telegram.org/bot{self.telegram_token}/getMe"
        try:
            t0 = datetime.utcnow()
            resp = requests.get(url, timeout=HEALTH_CHECK_TIMEOUT_S)
            latency = (datetime.utcnow() - t0).total_seconds() * 1000
            if resp.status_code == 200:
                bot_name = resp.json().get("result", {}).get("username", "?")
                return ServiceStatus(
                    "Telegram",
                    ok=True,
                    message=f"OK — bot @{bot_name}",
                    latency_ms=latency,
                )
            return ServiceStatus(
                "Telegram",
                ok=False,
                message=f"HTTP {resp.status_code}: {resp.text[:80]}",
                latency_ms=latency,
            )
        except requests.RequestException as exc:
            return ServiceStatus("Telegram", ok=False, message=f"Error de red: {exc}")

    def run_all(self) -> HealthReport:
        """Ejecuta todos los health checks y retorna un HealthReport."""
        checks = [self.check_alpaca(), self.check_fred(), self.check_telegram()]
        report = HealthReport(services=checks)
        if report.all_ok:
            log.info("Health check: todos los servicios OK (%d/%d)", report.n_ok, len(checks))
        else:
            log.warning("Health check: %d servicio(s) con problemas\n%s", report.n_failed, report.summary())
        return report


# ---------------------------------------------------------------------------
# Portfolio Monitor
# ---------------------------------------------------------------------------


@dataclass
class MetricsSnapshot:
    """Estado completo del portafolio en el momento del cálculo."""

    equity: float
    initial_equity: float
    total_return_pct: float
    daily_return_pct: float | None
    max_drawdown_pct: float
    current_drawdown_pct: float
    n_trades: int
    open_positions: dict[str, float]
    equity_peak: float
    cop_equity: float | None = None
    usd_cop: float | None = None
    computed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_in_drawdown(self) -> bool:
        return self.current_drawdown_pct > 0.01  # > 1%

    def as_alert_kwargs(self) -> dict:
        """Kwargs listos para AlertManager.daily_summary()."""
        from math import isnan
        return {
            "equity": self.equity,
            "initial_equity": self.initial_equity,
            "n_trades": self.n_trades,
            "drawdown": self.current_drawdown_pct,
            "open_positions": self.open_positions,
            "usd_cop": self.usd_cop,
            "cop_equity": self.cop_equity,
        }

    def report_lines(self) -> list[str]:
        """Líneas de texto para un reporte markdown."""
        sign = "+" if self.total_return_pct >= 0 else ""
        lines = [
            f"**Equity:** ${self.equity:,.2f} ({sign}{self.total_return_pct:.2f}%)",
            f"**Max drawdown:** {self.max_drawdown_pct * 100:.1f}%",
            f"**Drawdown actual:** {self.current_drawdown_pct * 100:.1f}%",
            f"**Trades totales:** {self.n_trades}",
            f"**Posiciones abiertas:** {len(self.open_positions)}",
        ]
        if self.cop_equity and self.usd_cop:
            lines.append(f"**Valor COP:** ${self.cop_equity:,.0f} (tasa {self.usd_cop:,.0f})")
        return lines


class PortfolioMonitor:
    """Calcula métricas de rendimiento a partir de los snapshots del pipeline.

    Parámetros
    ----------
    snapshots : list[PortfolioSnapshot]
        Snapshots del pipeline (cada uno = un bar procesado).
    initial_equity : float
        Capital inicial para calcular retorno total.
    n_trades : int
        Número de trades ejecutados.
    open_positions : dict[str, float]
        Posiciones abiertas al final del pipeline.
    usd_cop : float | None
        Tasa actual USD/COP para conversión a pesos.
    """

    def __init__(
        self,
        snapshots: list,
        initial_equity: float,
        n_trades: int = 0,
        open_positions: dict[str, float] | None = None,
        usd_cop: float | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.initial_equity = initial_equity
        self.n_trades = n_trades
        self.open_positions = open_positions or {}
        self.usd_cop = usd_cop

    def compute(self) -> MetricsSnapshot:
        """Calcula y retorna un MetricsSnapshot completo."""
        if not self.snapshots:
            return MetricsSnapshot(
                equity=self.initial_equity,
                initial_equity=self.initial_equity,
                total_return_pct=0.0,
                daily_return_pct=None,
                max_drawdown_pct=0.0,
                current_drawdown_pct=0.0,
                n_trades=self.n_trades,
                open_positions=self.open_positions,
                equity_peak=self.initial_equity,
                usd_cop=self.usd_cop,
            )

        equities = [s.equity for s in self.snapshots]
        current_equity = equities[-1]
        equity_peak = max(equities)

        # Retorno total
        total_return_pct = (current_equity / self.initial_equity - 1) * 100

        # Retorno diario (última jornada vs penúltima si hay suficientes datos)
        daily_return_pct: float | None = None
        if len(equities) >= 2:
            daily_return_pct = (equities[-1] / equities[-2] - 1) * 100

        # Max drawdown
        max_dd = self._max_drawdown(equities)

        # Drawdown actual (desde el último pico)
        current_dd = (equity_peak - current_equity) / equity_peak if equity_peak > 0 else 0.0

        # COP equity
        cop_equity = (current_equity * self.usd_cop) if self.usd_cop else None
        # Use snapshot cop_equity if available
        if hasattr(self.snapshots[-1], "cop_equity") and self.snapshots[-1].cop_equity:
            cop_equity = self.snapshots[-1].cop_equity

        return MetricsSnapshot(
            equity=current_equity,
            initial_equity=self.initial_equity,
            total_return_pct=total_return_pct,
            daily_return_pct=daily_return_pct,
            max_drawdown_pct=max_dd,
            current_drawdown_pct=current_dd,
            n_trades=self.n_trades,
            open_positions=self.open_positions,
            equity_peak=equity_peak,
            cop_equity=cop_equity,
            usd_cop=self.usd_cop,
        )

    @staticmethod
    def _max_drawdown(equities: list[float]) -> float:
        """Calcula el max drawdown histórico de la curva de equity."""
        if len(equities) < 2:
            return 0.0
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def daily_pnl_series(self) -> list[tuple[datetime, float]]:
        """Retorna lista (timestamp, pnl_diario_pct) para graficar."""
        if len(self.snapshots) < 2:
            return []
        result = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1].equity
            curr = self.snapshots[i].equity
            pnl_pct = (curr / prev - 1) * 100 if prev > 0 else 0.0
            result.append((self.snapshots[i].timestamp, pnl_pct))
        return result

    def equity_at(self, dt: datetime) -> float | None:
        """Retorna equity en o cerca del datetime dado."""
        for snap in reversed(self.snapshots):
            if snap.timestamp <= dt:
                return snap.equity
        return None
