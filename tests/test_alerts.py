"""Tests para el sistema de alertas — mocks HTTP para Telegram."""

from unittest.mock import MagicMock, patch

import pytest

from jarvis_bot.modules.alerts import (
    Alert,
    AlertEvent,
    AlertLevel,
    AlertManager,
    LogChannel,
    TelegramChannel,
)


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


class TestAlert:
    def test_format_text_contains_title(self):
        a = Alert(AlertEvent.TRADE_EXECUTED, "BUY AAPL", "qty=10 @$150", AlertLevel.SUCCESS)
        text = a.format_text()
        assert "BUY AAPL" in text
        assert "qty=10" in text

    def test_format_text_contains_level_icon(self):
        a = Alert(AlertEvent.DRAWDOWN_WARNING, "Warning", "dd=20%", AlertLevel.WARNING)
        assert AlertLevel.WARNING.value in a.format_text()

    def test_format_log_contains_event(self):
        a = Alert(AlertEvent.PIPELINE_START, "Inicio", "...", AlertLevel.INFO)
        assert "pipeline_start" in a.format_log().lower()

    def test_format_text_contains_timestamp(self):
        a = Alert(AlertEvent.DAILY_SUMMARY, "Resumen", "...", AlertLevel.INFO)
        assert "UTC" in a.format_text()


# ---------------------------------------------------------------------------
# LogChannel
# ---------------------------------------------------------------------------


class TestLogChannel:
    def test_send_always_returns_true(self):
        ch = LogChannel()
        a = Alert(AlertEvent.PIPELINE_START, "test", "body", AlertLevel.INFO)
        assert ch.send(a) is True

    def test_send_critical_uses_error_log(self):
        ch = LogChannel()
        a = Alert(AlertEvent.PIPELINE_ERROR, "error", "traceback", AlertLevel.CRITICAL)
        with patch("jarvis_bot.modules.alerts.log") as mock_log:
            ch.send(a)
            mock_log.log.assert_called_once()
            args = mock_log.log.call_args[0]
            import logging
            assert args[0] == logging.ERROR


# ---------------------------------------------------------------------------
# TelegramChannel
# ---------------------------------------------------------------------------


class TestTelegramChannel:
    def test_not_configured_without_token(self):
        ch = TelegramChannel(bot_token="", chat_id="")
        assert not ch.is_configured()

    def test_not_configured_without_chat_id(self):
        ch = TelegramChannel(bot_token="tok", chat_id="")
        assert not ch.is_configured()

    def test_configured_with_both(self):
        ch = TelegramChannel(bot_token="tok", chat_id="123")
        assert ch.is_configured()

    def test_send_returns_false_when_not_configured(self):
        ch = TelegramChannel(bot_token="", chat_id="")
        a = Alert(AlertEvent.TRADE_EXECUTED, "T", "B", AlertLevel.INFO)
        assert ch.send(a) is False

    @patch("requests.post")
    def test_send_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        ch = TelegramChannel(bot_token="tok", chat_id="123")
        a = Alert(AlertEvent.TRADE_EXECUTED, "BUY AAPL", "qty=5", AlertLevel.SUCCESS)
        result = ch.send(a)
        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["chat_id"] == "123"
        assert "BUY AAPL" in payload["text"]

    @patch("requests.post")
    def test_send_retries_on_failure(self, mock_post):
        bad_resp = MagicMock(status_code=500)
        mock_post.return_value = bad_resp
        ch = TelegramChannel(bot_token="tok", chat_id="123")
        a = Alert(AlertEvent.TRADE_EXECUTED, "T", "B", AlertLevel.INFO)
        result = ch.send(a)
        # Falla pero reintenta ALERT_RETRY_COUNT+1 veces
        from jarvis_bot.modules.alerts import ALERT_RETRY_COUNT
        assert mock_post.call_count == ALERT_RETRY_COUNT + 1
        assert result is False

    @patch("requests.post", side_effect=__import__("requests").RequestException("network error"))
    def test_send_handles_network_error(self, mock_post):
        ch = TelegramChannel(bot_token="tok", chat_id="123")
        a = Alert(AlertEvent.TRADE_EXECUTED, "T", "B", AlertLevel.INFO)
        result = ch.send(a)
        assert result is False


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def _manager_with_mock_channel(self):
        mock_ch = MagicMock()
        mock_ch.send.return_value = True
        mgr = AlertManager()
        mgr._channels = [mock_ch]
        return mgr, mock_ch

    def test_pipeline_start_dispatches_alert(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.pipeline_start(["AAPL", "VOO"], "combined", 10000.0)
        ch.send.assert_called_once()
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.PIPELINE_START
        assert "combined" in alert.body

    def test_trade_buy_dispatches_alert(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.trade("AAPL", "BUY", 10, 150.0, 10100.0, signal="BUY")
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.TRADE_EXECUTED
        assert "AAPL" in alert.title
        assert "BUY" in alert.title

    def test_trade_sell_dispatches_alert(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.trade("VOO", "SELL", 5, 200.0, 9800.0)
        alert = ch.send.call_args[0][0]
        assert "SELL" in alert.title

    def test_trade_includes_usd_cop(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.trade("AAPL", "BUY", 10, 150.0, 10000.0, usd_cop=4200.0)
        alert = ch.send.call_args[0][0]
        assert "4,200" in alert.body or "4200" in alert.body

    def test_drawdown_warning_dispatches_critical(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.drawdown_warning(0.18, 0.15, 8200.0)
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.DRAWDOWN_WARNING
        assert alert.level == AlertLevel.WARNING
        assert "18.0%" in alert.body

    def test_fx_risk_elevated_dispatches(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.fx_risk_elevated(7.2, 5.0, 4150.0)
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.FX_RISK_ELEVATED
        assert "7.2" in alert.body

    def test_fx_risk_high_dispatches_critical(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.fx_risk_high(12.5, 10.0, 4350.0)
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.FX_RISK_HIGH
        assert alert.level == AlertLevel.CRITICAL

    def test_market_closed_dispatches(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.market_closed("after-hours", "2026-06-30 09:30 COT")
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.MARKET_CLOSED

    def test_pipeline_error_dispatches_critical(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.pipeline_error("ValueError: bad data")
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.PIPELINE_ERROR
        assert alert.level == AlertLevel.CRITICAL

    def test_daily_summary_dispatches(self):
        mgr, ch = self._manager_with_mock_channel()
        mgr.daily_summary(
            equity=10500.0,
            initial_equity=10000.0,
            n_trades=5,
            drawdown=0.02,
            open_positions={"AAPL": 10},
            usd_cop=4200.0,
            cop_equity=44_100_000.0,
        )
        alert = ch.send.call_args[0][0]
        assert alert.event == AlertEvent.DAILY_SUMMARY
        assert "+5.00%" in alert.body
        assert "AAPL" in alert.body

    def test_channel_error_does_not_propagate(self):
        mgr = AlertManager()
        bad_ch = MagicMock()
        bad_ch.send.side_effect = RuntimeError("channel exploded")
        mgr._channels = [bad_ch]
        # No debe lanzar excepción
        mgr.trade("AAPL", "BUY", 10, 150.0, 10000.0)

    def test_default_has_log_channel(self):
        mgr = AlertManager()
        from jarvis_bot.modules.alerts import LogChannel
        assert any(isinstance(ch, LogChannel) for ch in mgr._channels)
