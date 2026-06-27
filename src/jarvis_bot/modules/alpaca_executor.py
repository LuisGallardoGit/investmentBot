"""Executor robusto para Alpaca — Fase 4.

Mejoras respecto al MVP:
  - Fill verification: polling hasta confirmar fill o timeout
  - Bracket orders: entry + stop-loss + take-profit en una sola orden
  - Retry con backoff exponencial en errores transitorios (429, 5xx)
  - Cancel antes de reemplazar: cancela órdenes pendientes del mismo símbolo
  - FillResult: datos reales del fill (precio, qty, comisiones estimadas)
  - is_configured() con mensaje claro de qué variable falta

Seguridad:
  - Nunca hardcodea claves. Lee de env.
  - Por defecto apunta a paper-api.alpaca.markets (no live).
  - Para live se debe setear ALPACA_ENDPOINT explícitamente.

Alpaca API v2 usada:
  POST /v2/orders      — crear orden
  GET  /v2/orders/{id} — consultar estado
  DELETE /v2/orders/{id} — cancelar orden
  GET  /v2/orders?status=open&symbols=X — listar pendientes por símbolo
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

log = logging.getLogger(__name__)

# Constantes
DEFAULT_ENDPOINT = "https://paper-api.alpaca.markets"
DEFAULT_FILL_TIMEOUT_S = 30       # segundos esperando fill
DEFAULT_FILL_POLL_INTERVAL_S = 2  # intervalo de polling
DEFAULT_MAX_RETRIES = 3           # reintentos en errores transitorios
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class OrderStatus(str, Enum):
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PENDING_CANCEL = "pending_cancel"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    REJECTED = "rejected"
    HELD = "held"


@dataclass
class FillResult:
    """Resultado real de una orden ejecutada en Alpaca."""

    order_id: str
    symbol: str
    side: str                          # "buy" | "sell"
    qty_requested: float
    qty_filled: float
    avg_fill_price: float | None       # None si no hubo fill
    status: OrderStatus
    raw: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

    @property
    def fill_value_usd(self) -> float:
        if self.avg_fill_price is None:
            return 0.0
        return self.qty_filled * self.avg_fill_price

    def __str__(self) -> str:
        price_str = f"@${self.avg_fill_price:.4f}" if self.avg_fill_price else "@?"
        return (
            f"[{self.status.value}] {self.side.upper()} {self.qty_filled}/{self.qty_requested} "
            f"{self.symbol} {price_str} (id={self.order_id[:8]}…)"
        )


@dataclass
class BracketConfig:
    """Configuración de una orden bracket (entry + stop + take profit).

    stop_loss_pct y take_profit_pct son relativos al precio de referencia.
    Ejemplo: precio=100, stop_loss_pct=0.04 → stop en 96.0
    """

    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.10

    def stop_price(self, ref_price: float, side: str = "buy") -> float:
        """Precio de stop-loss."""
        if side == "buy":
            return round(ref_price * (1 - self.stop_loss_pct), 4)
        return round(ref_price * (1 + self.stop_loss_pct), 4)  # short

    def take_profit_price(self, ref_price: float, side: str = "buy") -> float:
        """Precio de take-profit."""
        if side == "buy":
            return round(ref_price * (1 + self.take_profit_pct), 4)
        return round(ref_price * (1 - self.take_profit_pct), 4)  # short


class AlpacaAPIError(Exception):
    """Error retornable de la API Alpaca con status code."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class AlpacaExecutor:
    """Executor robusto para Alpaca Paper Trading.

    Parámetros
    ----------
    fill_timeout_s : int
        Segundos máximos esperando confirmación de fill antes de reportar timeout.
    poll_interval_s : int
        Intervalo entre consultas de estado de la orden.
    max_retries : int
        Reintentos en errores transitorios (429, 5xx).
    bracket : BracketConfig | None
        Si se provee, todas las órdenes BUY se envían como bracket.
    """

    def __init__(
        self,
        fill_timeout_s: int = DEFAULT_FILL_TIMEOUT_S,
        poll_interval_s: int = DEFAULT_FILL_POLL_INTERVAL_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        bracket: BracketConfig | None = None,
    ) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.endpoint = os.getenv("ALPACA_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
        self.fill_timeout_s = fill_timeout_s
        self.poll_interval_s = poll_interval_s
        self.max_retries = max_retries
        self.bracket = bracket

        self._headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Configuración y estado
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """True si las credenciales Alpaca están disponibles en env."""
        missing = []
        if not self.api_key:
            missing.append("ALPACA_API_KEY")
        if not self.secret_key:
            missing.append("ALPACA_SECRET_KEY")
        if missing:
            log.debug("Alpaca no configurado. Variables faltantes: %s", missing)
            return False
        return True

    def is_paper(self) -> bool:
        """True si el endpoint apunta a paper trading."""
        return "paper-api" in self.endpoint

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Hace una request HTTP con retry en errores transitorios."""
        url = f"{self.endpoint}{path}"
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= self.max_retries:
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=self._headers,
                    json=payload,
                    params=params,
                    timeout=10,
                )
                if resp.status_code in RETRYABLE_HTTP_CODES:
                    wait = 2 ** attempt
                    log.warning("HTTP %d — reintentando en %ds (intento %d/%d)…",
                                resp.status_code, wait, attempt + 1, self.max_retries)
                    time.sleep(wait)
                    attempt += 1
                    continue
                if resp.status_code not in (200, 201, 204):
                    raise AlpacaAPIError(resp.status_code, resp.text[:300])
                if resp.status_code == 204 or not resp.content:
                    return {}
                return resp.json()
            except AlpacaAPIError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                log.warning("Error de red — reintentando en %ds: %s", wait, exc)
                time.sleep(wait)
                attempt += 1

        raise RuntimeError(
            f"Máximo de reintentos alcanzado para {method} {path}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Órdenes
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        ref_price: float | None = None,
        time_in_force: str = "day",
        use_bracket: bool = True,
    ) -> FillResult | None:
        """Envía una orden a Alpaca y espera confirmación de fill.

        Parámetros
        ----------
        symbol : str
            Ticker (ej. "AAPL").
        side : str
            "buy" o "sell".
        qty : float
            Número de acciones.
        order_type : str
            "market" (default) o "limit".
        ref_price : float | None
            Precio de referencia para bracket orders y logging.
        time_in_force : str
            "day" (default para mercado regular) o "gtc".
        use_bracket : bool
            Si True y self.bracket está configurado, envía bracket order en BUY.

        Retorna
        -------
        FillResult o None si Alpaca no está configurado.
        """
        if not self.is_configured():
            log.info("Alpaca no configurado — omitiendo orden %s %s %s.", side.upper(), qty, symbol)
            return None

        symbol = symbol.upper()
        side = side.lower()

        # Cancelar órdenes abiertas del mismo símbolo antes de enviar una nueva
        self._cancel_open_orders(symbol)

        # Construir payload
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(int(qty)) if qty == int(qty) else str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        # Bracket order (solo para BUY con bracket configurado)
        if use_bracket and side == "buy" and self.bracket and ref_price:
            payload["order_class"] = "bracket"
            payload["stop_loss"] = {
                "stop_price": str(self.bracket.stop_price(ref_price, side)),
            }
            payload["take_profit"] = {
                "limit_price": str(self.bracket.take_profit_price(ref_price, side)),
            }
            log.info(
                "Bracket BUY %s x%s | stop=%.4f | tp=%.4f",
                symbol, qty,
                self.bracket.stop_price(ref_price),
                self.bracket.take_profit_price(ref_price),
            )
        else:
            log.info("Orden %s %s x%s (market)", side.upper(), symbol, qty)

        try:
            data = self._request("POST", "/v2/orders", payload=payload)
            order_id = data.get("id", "")
            log.info("Orden aceptada. ID=%s status=%s", order_id[:8] if order_id else "?", data.get("status"))
            return self._wait_for_fill(order_id, symbol, side, float(qty), data)
        except AlpacaAPIError as exc:
            log.error("Alpaca rechazó la orden %s %s: %s", side.upper(), symbol, exc)
            return FillResult(
                order_id="",
                symbol=symbol,
                side=side,
                qty_requested=float(qty),
                qty_filled=0.0,
                avg_fill_price=None,
                status=OrderStatus.REJECTED,
                raw={"error": str(exc)},
            )
        except Exception as exc:
            log.error("Error enviando orden %s %s: %s", side.upper(), symbol, exc)
            return None

    def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty_requested: float,
        initial_data: dict,
    ) -> FillResult:
        """Espera hasta fill o timeout y retorna FillResult."""
        deadline = time.monotonic() + self.fill_timeout_s
        data = initial_data

        while time.monotonic() < deadline:
            status_str = data.get("status", "")
            filled_qty = float(data.get("filled_qty") or 0)
            fill_price = data.get("filled_avg_price")
            avg_price = float(fill_price) if fill_price else None

            if status_str in ("filled", "partially_filled"):
                result = FillResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    qty_requested=qty_requested,
                    qty_filled=filled_qty,
                    avg_fill_price=avg_price,
                    status=OrderStatus(status_str),
                    raw=data,
                )
                log.info("Fill confirmado: %s", result)
                return result

            if status_str in ("cancelled", "expired", "rejected"):
                log.warning("Orden %s terminó con estado: %s", order_id[:8], status_str)
                return FillResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    qty_requested=qty_requested,
                    qty_filled=filled_qty,
                    avg_fill_price=avg_price,
                    status=OrderStatus(status_str),
                    raw=data,
                )

            # Polling
            time.sleep(self.poll_interval_s)
            try:
                data = self._request("GET", f"/v2/orders/{order_id}")
            except Exception as exc:
                log.warning("Error consultando orden %s: %s", order_id[:8], exc)

        # Timeout
        log.warning(
            "Timeout esperando fill de orden %s después de %ds.",
            order_id[:8] if order_id else "?",
            self.fill_timeout_s,
        )
        return FillResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty_requested=qty_requested,
            qty_filled=float(data.get("filled_qty") or 0),
            avg_fill_price=None,
            status=OrderStatus(data.get("status", "accepted")),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Gestión de órdenes abiertas
    # ------------------------------------------------------------------

    def _cancel_open_orders(self, symbol: str) -> int:
        """Cancela todas las órdenes abiertas de un símbolo. Retorna número canceladas."""
        if not self.is_configured():
            return 0
        try:
            orders = self._request("GET", "/v2/orders", params={"status": "open", "symbols": symbol})
            if not isinstance(orders, list):
                return 0
            cancelled = 0
            for order in orders:
                oid = order.get("id", "")
                if not oid:
                    continue
                try:
                    self._request("DELETE", f"/v2/orders/{oid}")
                    log.info("Cancelada orden abierta %s para %s", oid[:8], symbol)
                    cancelled += 1
                except Exception as exc:
                    log.warning("No se pudo cancelar orden %s: %s", oid[:8], exc)
            return cancelled
        except Exception as exc:
            log.warning("Error listando órdenes abiertas de %s: %s", symbol, exc)
            return 0

    def cancel_all_orders(self) -> int:
        """Cancela TODAS las órdenes abiertas (útil al cerrar el bot)."""
        if not self.is_configured():
            return 0
        try:
            self._request("DELETE", "/v2/orders")
            log.info("Todas las órdenes abiertas canceladas.")
            return 1
        except Exception as exc:
            log.error("Error cancelando todas las órdenes: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Consultas de cuenta
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Retorna las posiciones abiertas en Alpaca."""
        if not self.is_configured():
            return []
        try:
            result = self._request("GET", "/v2/positions")
            return result if isinstance(result, list) else []
        except Exception as exc:
            log.error("Error obteniendo posiciones Alpaca: %s", exc)
            return []

    def get_account(self) -> dict:
        """Retorna datos de la cuenta Alpaca (cash, equity, buying power)."""
        if not self.is_configured():
            return {}
        try:
            return self._request("GET", "/v2/account")
        except Exception as exc:
            log.error("Error obteniendo cuenta Alpaca: %s", exc)
            return {}

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Lista órdenes abiertas, opcionalmente filtradas por símbolo."""
        if not self.is_configured():
            return []
        params = {"status": "open"}
        if symbol:
            params["symbols"] = symbol.upper()
        try:
            result = self._request("GET", "/v2/orders", params=params)
            return result if isinstance(result, list) else []
        except Exception as exc:
            log.error("Error listando órdenes abiertas: %s", exc)
            return []
