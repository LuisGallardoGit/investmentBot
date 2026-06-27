"""Sistema de alertas para el bot de trading — Fase 5.

Canales soportados:
  - Telegram (Bot API): notificaciones push en tiempo real
  - Log (siempre activo): fallback si Telegram no está configurado

Tipos de evento:
  - TRADE_EXECUTED: BUY/SELL ejecutado en el simulador o en Alpaca
  - DRAWDOWN_WARNING: drawdown supera umbral (circuit breaker activo)
  - FX_RISK_ELEVATED: volatilidad COP/USD supera fx_vol_caution_pct
  - FX_RISK_HIGH: volatilidad COP/USD supera fx_vol_block_pct (entradas bloqueadas)
  - MARKET_CLOSED: pipeline corrió fuera de horario NYSE
  - PIPELINE_ERROR: excepción no recuperable en el pipeline
  - PIPELINE_START: inicio del pipeline (health check)
  - DAILY_SUMMARY: resumen diario de rendimiento

Configuración (variables de entorno):
  TELEGRAM_BOT_TOKEN — token del bot (BotFather)
  TELEGRAM_CHAT_ID   — chat_id del usuario o grupo destino

Seguridad:
  - Nunca incluye API keys en los mensajes de alerta
  - Mensajes formatados en Markdown (Telegram MarkdownV2)
  - Reintenta 2 veces en errores de red antes de caer al log
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol

import requests

log = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
ALERT_RETRY_COUNT = 2
ALERT_TIMEOUT_S = 5


class AlertLevel(str, Enum):
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    CRITICAL = "🚨"


class AlertEvent(str, Enum):
    TRADE_EXECUTED = "trade_executed"
    DRAWDOWN_WARNING = "drawdown_warning"
    FX_RISK_ELEVATED = "fx_risk_elevated"
    FX_RISK_HIGH = "fx_risk_high"
    MARKET_CLOSED = "market_closed"
    PIPELINE_ERROR = "pipeline_error"
    PIPELINE_START = "pipeline_start"
    DAILY_SUMMARY = "daily_summary"


# Nivel por defecto para cada evento
_EVENT_LEVELS: dict[AlertEvent, AlertLevel] = {
    AlertEvent.TRADE_EXECUTED: AlertLevel.SUCCESS,
    AlertEvent.DRAWDOWN_WARNING: AlertLevel.WARNING,
    AlertEvent.FX_RISK_ELEVATED: AlertLevel.WARNING,
    AlertEvent.FX_RISK_HIGH: AlertLevel.CRITICAL,
    AlertEvent.MARKET_CLOSED: AlertLevel.INFO,
    AlertEvent.PIPELINE_ERROR: AlertLevel.CRITICAL,
    AlertEvent.PIPELINE_START: AlertLevel.INFO,
    AlertEvent.DAILY_SUMMARY: AlertLevel.INFO,
}


@dataclass
class Alert:
    """Una alerta lista para enviar."""

    event: AlertEvent
    title: str
    body: str
    level: AlertLevel = AlertLevel.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def format_text(self) -> str:
        """Formatea el mensaje para Telegram (plain text seguro)."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"{self.level.value} {self.title}\n"
            f"{self.body}\n"
            f"_{ts}_"
        )

    def format_log(self) -> str:
        return f"[{self.event.value.upper()}] {self.title} | {self.body}"


class AlertChannel(Protocol):
    """Interfaz para canales de alerta."""

    def send(self, alert: Alert) -> bool:
        """Envía la alerta. Retorna True si tuvo éxito."""
        ...


class LogChannel:
    """Canal de alerta simple: escribe en el log de Python."""

    def send(self, alert: Alert) -> bool:
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.SUCCESS: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
        }
        log.log(level_map.get(alert.level, logging.INFO), alert.format_log())
        return True


class TelegramChannel:
    """Envía alertas vía Telegram Bot API.

    Requiere:
      TELEGRAM_BOT_TOKEN — token del bot
      TELEGRAM_CHAT_ID   — ID del chat destino (usuario o grupo)
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            log.debug("Telegram no configurado — alerta omitida: %s", alert.title)
            return False

        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": alert.format_text(),
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        for attempt in range(ALERT_RETRY_COUNT + 1):
            try:
                resp = requests.post(url, json=payload, timeout=ALERT_TIMEOUT_S)
                if resp.status_code == 200:
                    log.debug("Alerta Telegram enviada: %s", alert.title)
                    return True
                log.warning(
                    "Telegram respondió %d al enviar alerta '%s': %s",
                    resp.status_code, alert.title, resp.text[:100],
                )
            except requests.RequestException as exc:
                if attempt < ALERT_RETRY_COUNT:
                    log.debug("Reintentando alerta Telegram (%d/%d): %s", attempt + 1, ALERT_RETRY_COUNT, exc)
                else:
                    log.error("Error enviando alerta Telegram tras %d intentos: %s", ALERT_RETRY_COUNT + 1, exc)
        return False


class AlertManager:
    """Gestiona múltiples canales de alerta y despacha eventos.

    Siempre incluye el LogChannel como canal base. Si Telegram está
    configurado, también envía por ahí.

    Uso básico:
        mgr = AlertManager()
        mgr.trade(symbol="AAPL", side="BUY", qty=10, price=155.0, equity=10500.0)
        mgr.drawdown_warning(current_dd=0.16, limit=0.15)
    """

    def __init__(
        self,
        extra_channels: list[AlertChannel] | None = None,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
    ) -> None:
        self._channels: list[AlertChannel] = [LogChannel()]

        # Telegram automático si hay credenciales
        tg = TelegramChannel(bot_token=telegram_token, chat_id=telegram_chat_id)
        if tg.is_configured():
            self._channels.append(tg)
            log.info("AlertManager: Telegram configurado.")
        else:
            log.debug("AlertManager: Telegram no configurado — solo log.")

        if extra_channels:
            self._channels.extend(extra_channels)

    def _dispatch(self, event: AlertEvent, title: str, body: str, level: AlertLevel | None = None) -> None:
        lvl = level or _EVENT_LEVELS.get(event, AlertLevel.INFO)
        alert = Alert(event=event, title=title, body=body, level=lvl)
        for channel in self._channels:
            try:
                channel.send(alert)
            except Exception as exc:
                log.error("Error en canal de alerta %s: %s", type(channel).__name__, exc)

    # ------------------------------------------------------------------
    # Métodos de conveniencia por tipo de evento
    # ------------------------------------------------------------------

    def pipeline_start(self, symbols: list[str], strategy: str, equity: float) -> None:
        self._dispatch(
            AlertEvent.PIPELINE_START,
            title="Pipeline iniciado",
            body=(
                f"Estrategia: *{strategy}*\n"
                f"Símbolos: {', '.join(symbols)}\n"
                f"Equity inicial: ${equity:,.2f}"
            ),
        )

    def trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        equity: float,
        signal: str = "",
        usd_cop: float | None = None,
    ) -> None:
        emoji = "📈" if side.upper() == "BUY" else "📉"
        cop_line = f"\nUSD/COP: {usd_cop:,.0f}" if usd_cop else ""
        self._dispatch(
            AlertEvent.TRADE_EXECUTED,
            title=f"{emoji} {side.upper()} {symbol}",
            body=(
                f"Qty: {qty} @ ${price:.2f}\n"
                f"Equity post-trade: ${equity:,.2f}"
                f"{cop_line}"
                + (f"\nSeñal: {signal}" if signal else "")
            ),
        )

    def drawdown_warning(self, current_dd: float, limit: float, equity: float) -> None:
        self._dispatch(
            AlertEvent.DRAWDOWN_WARNING,
            title="⛔ Circuit breaker activado",
            body=(
                f"Drawdown actual: *{current_dd * 100:.1f}%*\n"
                f"Límite configurado: {limit * 100:.1f}%\n"
                f"Equity: ${equity:,.2f}\n"
                f"No se abrirán nuevas posiciones."
            ),
        )

    def fx_risk_elevated(self, change_30d: float, caution_pct: float, usd_cop: float) -> None:
        self._dispatch(
            AlertEvent.FX_RISK_ELEVATED,
            title="COP/USD — Volatilidad elevada",
            body=(
                f"Cambio 30d: *{change_30d:+.1f}%* (umbral cautela: {caution_pct:.1f}%)\n"
                f"Tasa actual: {usd_cop:,.0f} COP/USD\n"
                f"Tamaño de posición reducido al 50%."
            ),
        )

    def fx_risk_high(self, change_30d: float, block_pct: float, usd_cop: float) -> None:
        self._dispatch(
            AlertEvent.FX_RISK_HIGH,
            title="🚫 COP/USD — Nuevas entradas bloqueadas",
            body=(
                f"Cambio 30d: *{change_30d:+.1f}%* (umbral bloqueo: {block_pct:.1f}%)\n"
                f"Tasa actual: {usd_cop:,.0f} COP/USD\n"
                f"No se abrirán nuevas posiciones hasta que el FX se estabilice."
            ),
        )

    def market_closed(self, session: str, next_open_bogota: str) -> None:
        self._dispatch(
            AlertEvent.MARKET_CLOSED,
            title="Mercado cerrado",
            body=(
                f"Sesión: {session}\n"
                f"Próxima apertura (Bogotá): {next_open_bogota}\n"
                f"El pipeline corrió en modo histórico."
            ),
        )

    def pipeline_error(self, error: str) -> None:
        self._dispatch(
            AlertEvent.PIPELINE_ERROR,
            title="Error en el pipeline",
            body=f"```\n{error[:500]}\n```",
            level=AlertLevel.CRITICAL,
        )

    def daily_summary(
        self,
        equity: float,
        initial_equity: float,
        n_trades: int,
        drawdown: float,
        open_positions: dict[str, float],
        usd_cop: float | None = None,
        cop_equity: float | None = None,
    ) -> None:
        total_return = (equity / initial_equity - 1) * 100
        sign = "+" if total_return >= 0 else ""
        pos_str = (
            ", ".join(f"{s}({q:.0f})" for s, q in open_positions.items())
            if open_positions else "ninguna"
        )
        cop_line = f"\nValor en COP: ${cop_equity:,.0f}" if cop_equity and usd_cop else ""
        self._dispatch(
            AlertEvent.DAILY_SUMMARY,
            title="📊 Resumen diario",
            body=(
                f"Equity: *${equity:,.2f}* ({sign}{total_return:.2f}%)\n"
                f"Trades hoy: {n_trades}\n"
                f"Drawdown actual: {drawdown * 100:.1f}%\n"
                f"Posiciones abiertas: {pos_str}"
                f"{cop_line}"
            ),
        )
