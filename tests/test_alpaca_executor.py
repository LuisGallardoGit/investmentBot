"""Tests para AlpacaExecutor — con mocks de la API Alpaca."""

import time
from unittest.mock import MagicMock, patch

import pytest

from jarvis_bot.modules.alpaca_executor import (
    AlpacaAPIError,
    AlpacaExecutor,
    BracketConfig,
    FillResult,
    OrderStatus,
)


# ---------------------------------------------------------------------------
# BracketConfig
# ---------------------------------------------------------------------------


class TestBracketConfig:
    def setup_method(self):
        self.b = BracketConfig(stop_loss_pct=0.04, take_profit_pct=0.10)

    def test_stop_price_buy(self):
        assert self.b.stop_price(100.0, "buy") == pytest.approx(96.0, rel=1e-4)

    def test_stop_price_short(self):
        assert self.b.stop_price(100.0, "sell") == pytest.approx(104.0, rel=1e-4)

    def test_take_profit_buy(self):
        assert self.b.take_profit_price(100.0, "buy") == pytest.approx(110.0, rel=1e-4)

    def test_take_profit_short(self):
        assert self.b.take_profit_price(100.0, "sell") == pytest.approx(90.0, rel=1e-4)

    def test_prices_are_rounded(self):
        # Precio que da muchos decimales → máximo 4
        p = BracketConfig(stop_loss_pct=0.033, take_profit_pct=0.077)
        stop = p.stop_price(123.456)
        assert len(str(stop).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# FillResult
# ---------------------------------------------------------------------------


class TestFillResult:
    def _make(self, status=OrderStatus.FILLED, qty_filled=10.0, price=150.0):
        return FillResult(
            order_id="abc123",
            symbol="AAPL",
            side="buy",
            qty_requested=10.0,
            qty_filled=qty_filled,
            avg_fill_price=price,
            status=status,
        )

    def test_is_filled_when_filled(self):
        assert self._make(OrderStatus.FILLED).is_filled

    def test_is_filled_when_partially_filled(self):
        assert self._make(OrderStatus.PARTIALLY_FILLED, qty_filled=5.0).is_filled

    def test_not_filled_when_cancelled(self):
        assert not self._make(OrderStatus.CANCELLED, qty_filled=0.0, price=None).is_filled

    def test_fill_value_usd(self):
        r = self._make(qty_filled=10.0, price=150.0)
        assert r.fill_value_usd == pytest.approx(1500.0)

    def test_fill_value_usd_zero_when_no_price(self):
        r = FillResult("id", "X", "buy", 5.0, 0.0, None, OrderStatus.CANCELLED)
        assert r.fill_value_usd == 0.0

    def test_str_contains_symbol(self):
        assert "AAPL" in str(self._make())


# ---------------------------------------------------------------------------
# AlpacaExecutor.is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_not_configured_without_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            ex = AlpacaExecutor()
            ex.api_key = ""
            ex.secret_key = ""
            assert not ex.is_configured()

    def test_configured_with_both_keys(self):
        ex = AlpacaExecutor()
        ex.api_key = "test_key"
        ex.secret_key = "test_secret"
        assert ex.is_configured()

    def test_not_configured_missing_secret(self):
        ex = AlpacaExecutor()
        ex.api_key = "test_key"
        ex.secret_key = ""
        assert not ex.is_configured()

    def test_is_paper_endpoint(self):
        ex = AlpacaExecutor()
        ex.endpoint = "https://paper-api.alpaca.markets"
        assert ex.is_paper()

    def test_not_paper_live_endpoint(self):
        ex = AlpacaExecutor()
        ex.endpoint = "https://api.alpaca.markets"
        assert not ex.is_paper()


# ---------------------------------------------------------------------------
# AlpacaExecutor.submit_order — sin credenciales
# ---------------------------------------------------------------------------


class TestSubmitOrderUnconfigured:
    def test_returns_none_when_not_configured(self):
        ex = AlpacaExecutor()
        ex.api_key = ""
        ex.secret_key = ""
        result = ex.submit_order("AAPL", "buy", 10)
        assert result is None

    def test_get_positions_empty_when_not_configured(self):
        ex = AlpacaExecutor()
        ex.api_key = ""
        ex.secret_key = ""
        assert ex.get_positions() == []

    def test_get_account_empty_when_not_configured(self):
        ex = AlpacaExecutor()
        ex.api_key = ""
        ex.secret_key = ""
        assert ex.get_account() == {}

    def test_get_open_orders_empty_when_not_configured(self):
        ex = AlpacaExecutor()
        ex.api_key = ""
        ex.secret_key = ""
        assert ex.get_open_orders() == []


# ---------------------------------------------------------------------------
# AlpacaExecutor._request con retry
# ---------------------------------------------------------------------------


class TestRequestRetry:
    def _executor(self):
        ex = AlpacaExecutor(max_retries=2)
        ex.api_key = "key"
        ex.secret_key = "secret"
        return ex

    @patch("requests.request")
    def test_retries_on_429(self, mock_req):
        """Debe reintentar en 429 y eventualmente tener éxito."""
        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.content = b'{"id": "abc"}'
        resp_ok.json.return_value = {"id": "abc"}

        # Primera llamada 429, segunda OK
        mock_req.side_effect = [resp_429, resp_ok]

        ex = self._executor()
        with patch("time.sleep"):  # no esperamos en tests
            result = ex._request("GET", "/v2/account")
        assert result == {"id": "abc"}
        assert mock_req.call_count == 2

    @patch("requests.request")
    def test_raises_api_error_on_4xx(self, mock_req):
        """Errores 4xx no reintentables deben lanzar AlpacaAPIError."""
        resp = MagicMock()
        resp.status_code = 403
        resp.text = "Forbidden"
        mock_req.return_value = resp

        ex = self._executor()
        with pytest.raises(AlpacaAPIError) as exc_info:
            ex._request("GET", "/v2/account")
        assert exc_info.value.status_code == 403

    @patch("requests.request")
    def test_raises_after_max_retries(self, mock_req):
        """Tras max_retries de 5xx debe lanzar RuntimeError."""
        resp = MagicMock()
        resp.status_code = 503
        mock_req.return_value = resp

        ex = AlpacaExecutor(max_retries=1)
        ex.api_key = "key"
        ex.secret_key = "secret"

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Máximo de reintentos"):
                ex._request("GET", "/v2/account")


# ---------------------------------------------------------------------------
# AlpacaExecutor.submit_order con mock completo
# ---------------------------------------------------------------------------


class TestSubmitOrderMocked:
    def _executor(self, bracket=None):
        ex = AlpacaExecutor(fill_timeout_s=5, poll_interval_s=0, bracket=bracket)
        ex.api_key = "key"
        ex.secret_key = "secret"
        return ex

    def _make_order_response(self, status="filled", filled_qty="10", fill_price="150.5"):
        return {
            "id": "order-uuid-1234",
            "status": status,
            "filled_qty": filled_qty,
            "filled_avg_price": fill_price,
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
        }

    @patch.object(AlpacaExecutor, "_request")
    def test_immediate_fill(self, mock_req):
        """Orden que se llena inmediatamente."""
        order_resp = self._make_order_response(status="filled")
        # POST order → GET open orders (cancelar) → respuesta de orden
        mock_req.side_effect = [
            [],           # GET open orders (cancel check)
            order_resp,   # POST order
        ]

        ex = self._executor()
        result = ex.submit_order("AAPL", "buy", 10, ref_price=150.0)

        assert result is not None
        assert result.is_filled
        assert result.qty_filled == 10.0
        assert result.avg_fill_price == pytest.approx(150.5)
        assert result.order_id == "order-uuid-1234"

    @patch.object(AlpacaExecutor, "_request")
    def test_fill_after_polling(self, mock_req):
        """Orden que se llena en el segundo poll."""
        pending_resp = self._make_order_response(status="accepted", filled_qty="0", fill_price=None)
        filled_resp = self._make_order_response(status="filled")

        mock_req.side_effect = [
            [],            # cancel check
            pending_resp,  # POST order → pending
            filled_resp,   # GET order poll → filled
        ]

        ex = self._executor()
        result = ex.submit_order("AAPL", "buy", 10, ref_price=150.0)
        assert result is not None
        assert result.is_filled

    @patch.object(AlpacaExecutor, "_request")
    def test_rejected_order(self, mock_req):
        """Orden rechazada retorna FillResult con status REJECTED."""
        rejected_resp = self._make_order_response(status="rejected", filled_qty="0", fill_price=None)
        mock_req.side_effect = [[], rejected_resp]

        ex = self._executor()
        result = ex.submit_order("AAPL", "buy", 10)
        assert result is not None
        assert not result.is_filled
        assert result.status == OrderStatus.REJECTED

    @patch.object(AlpacaExecutor, "_request")
    def test_bracket_order_payload(self, mock_req):
        """Orden bracket debe incluir stop_loss y take_profit en payload."""
        filled_resp = self._make_order_response(status="filled")
        mock_req.side_effect = [[], filled_resp]

        bracket = BracketConfig(stop_loss_pct=0.04, take_profit_pct=0.10)
        ex = self._executor(bracket=bracket)
        ex.submit_order("AAPL", "buy", 10, ref_price=100.0, use_bracket=True)

        # El segundo call es el POST a /v2/orders
        post_call = mock_req.call_args_list[1]
        payload = post_call[1].get("payload") or post_call[0][2]
        assert "stop_loss" in payload
        assert "take_profit" in payload
        assert payload["order_class"] == "bracket"

    @patch.object(AlpacaExecutor, "_request")
    def test_sell_no_bracket(self, mock_req):
        """SELL no debe tener bracket aunque esté configurado."""
        filled_resp = {**self._make_order_response(status="filled"), "side": "sell"}
        mock_req.side_effect = [[], filled_resp]

        bracket = BracketConfig()
        ex = self._executor(bracket=bracket)
        ex.submit_order("AAPL", "sell", 10, use_bracket=False)

        post_call = mock_req.call_args_list[1]
        payload = post_call[1].get("payload") or post_call[0][2]
        assert "stop_loss" not in payload
        assert "order_class" not in payload

    @patch.object(AlpacaExecutor, "_request")
    def test_api_error_returns_rejected_fill(self, mock_req):
        """Error de API retorna FillResult REJECTED (no propaga excepción)."""
        mock_req.side_effect = [[], AlpacaAPIError(422, "insufficient funds")]

        ex = self._executor()
        result = ex.submit_order("AAPL", "buy", 10)
        assert result is not None
        assert result.status == OrderStatus.REJECTED
        assert not result.is_filled


# ---------------------------------------------------------------------------
# AlpacaExecutor.cancel_all_orders
# ---------------------------------------------------------------------------


class TestCancelAllOrders:
    @patch.object(AlpacaExecutor, "_request")
    def test_cancel_all_calls_delete(self, mock_req):
        mock_req.return_value = {}
        ex = AlpacaExecutor()
        ex.api_key = "key"
        ex.secret_key = "secret"
        result = ex.cancel_all_orders()
        assert result == 1
        mock_req.assert_called_once_with("DELETE", "/v2/orders")
