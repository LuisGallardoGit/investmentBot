"""Tests para monitoring.py — HealthCheck y PortfolioMonitor."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from jarvis_bot.monitoring import (
    HealthCheck,
    HealthReport,
    MetricsSnapshot,
    PortfolioMonitor,
    ServiceStatus,
)


# ---------------------------------------------------------------------------
# ServiceStatus
# ---------------------------------------------------------------------------


class TestServiceStatus:
    def test_str_ok(self):
        s = ServiceStatus("Alpaca", ok=True, message="OK", latency_ms=42.0)
        assert "✅" in str(s)
        assert "Alpaca" in str(s)
        assert "42ms" in str(s)

    def test_str_failed(self):
        s = ServiceStatus("FRED", ok=False, message="API key missing")
        assert "❌" in str(s)

    def test_str_no_latency(self):
        s = ServiceStatus("Telegram", ok=True, message="OK")
        assert "ms" not in str(s)


# ---------------------------------------------------------------------------
# HealthReport
# ---------------------------------------------------------------------------


class TestHealthReport:
    def _report(self, statuses):
        return HealthReport(services=statuses)

    def test_all_ok_true(self):
        r = self._report([
            ServiceStatus("A", ok=True, message="ok"),
            ServiceStatus("B", ok=True, message="ok"),
        ])
        assert r.all_ok

    def test_all_ok_false_with_one_failure(self):
        r = self._report([
            ServiceStatus("A", ok=True, message="ok"),
            ServiceStatus("B", ok=False, message="error"),
        ])
        assert not r.all_ok

    def test_n_ok_and_n_failed(self):
        r = self._report([
            ServiceStatus("A", ok=True, message="ok"),
            ServiceStatus("B", ok=False, message="error"),
            ServiceStatus("C", ok=True, message="ok"),
        ])
        assert r.n_ok == 2
        assert r.n_failed == 1

    def test_summary_contains_service_names(self):
        r = self._report([
            ServiceStatus("Alpaca", ok=True, message="ok"),
            ServiceStatus("FRED", ok=False, message="error"),
        ])
        s = r.summary()
        assert "Alpaca" in s
        assert "FRED" in s


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def _hc(self, alpaca_key="", alpaca_secret="", fred_key="", telegram_token=""):
        hc = HealthCheck()
        hc.alpaca_key = alpaca_key
        hc.alpaca_secret = alpaca_secret
        hc.fred_key = fred_key
        hc.telegram_token = telegram_token
        return hc

    def test_alpaca_not_configured(self):
        hc = self._hc()
        s = hc.check_alpaca()
        assert not s.ok
        assert "Credenciales" in s.message

    def test_fred_not_configured(self):
        hc = self._hc()
        s = hc.check_fred()
        assert not s.ok
        assert "FRED_API_KEY" in s.message

    def test_telegram_not_configured(self):
        hc = self._hc()
        s = hc.check_telegram()
        assert not s.ok
        assert "TELEGRAM_BOT_TOKEN" in s.message

    @patch("requests.get")
    def test_alpaca_ok(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ACTIVE"},
        )
        hc = self._hc(alpaca_key="k", alpaca_secret="s")
        s = hc.check_alpaca()
        assert s.ok
        assert "ACTIVE" in s.message

    @patch("requests.get")
    def test_alpaca_401(self, mock_get):
        mock_get.return_value = MagicMock(status_code=401, text="Unauthorized")
        hc = self._hc(alpaca_key="bad", alpaca_secret="bad")
        s = hc.check_alpaca()
        assert not s.ok
        assert "401" in s.message

    @patch("requests.get")
    def test_fred_ok(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"observations": [{"date": "2026-06-20"}]},
        )
        hc = self._hc(fred_key="k")
        s = hc.check_fred()
        assert s.ok
        assert "2026-06-20" in s.message

    @patch("requests.get")
    def test_telegram_ok(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": {"username": "my_bot"}},
        )
        hc = self._hc(telegram_token="tok")
        s = hc.check_telegram()
        assert s.ok
        assert "my_bot" in s.message

    @patch("requests.get", side_effect=__import__("requests").RequestException("network error"))
    def test_alpaca_network_error(self, _):
        hc = self._hc(alpaca_key="k", alpaca_secret="s")
        s = hc.check_alpaca()
        assert not s.ok
        assert "red" in s.message.lower() or "error" in s.message.lower()

    @patch.object(HealthCheck, "check_alpaca")
    @patch.object(HealthCheck, "check_fred")
    @patch.object(HealthCheck, "check_telegram")
    def test_run_all_runs_all_checks(self, mock_tg, mock_fred, mock_alpaca):
        for m in (mock_alpaca, mock_fred, mock_tg):
            m.return_value = ServiceStatus("X", ok=True, message="ok")
        hc = HealthCheck()
        report = hc.run_all()
        assert len(report.services) == 3
        mock_alpaca.assert_called_once()
        mock_fred.assert_called_once()
        mock_tg.assert_called_once()


# ---------------------------------------------------------------------------
# MetricsSnapshot
# ---------------------------------------------------------------------------


class TestMetricsSnapshot:
    def _snap(self, **kwargs):
        defaults = dict(
            equity=10500.0,
            initial_equity=10000.0,
            total_return_pct=5.0,
            daily_return_pct=0.5,
            max_drawdown_pct=0.03,
            current_drawdown_pct=0.0,
            n_trades=3,
            open_positions={"AAPL": 10},
            equity_peak=10500.0,
        )
        defaults.update(kwargs)
        return MetricsSnapshot(**defaults)

    def test_is_in_drawdown_false_when_small(self):
        assert not self._snap(current_drawdown_pct=0.005).is_in_drawdown

    def test_is_in_drawdown_true_when_large(self):
        assert self._snap(current_drawdown_pct=0.05).is_in_drawdown

    def test_as_alert_kwargs_has_required_keys(self):
        s = self._snap()
        kwargs = s.as_alert_kwargs()
        for k in ("equity", "initial_equity", "n_trades", "drawdown", "open_positions"):
            assert k in kwargs

    def test_report_lines_contains_equity(self):
        s = self._snap()
        lines = s.report_lines()
        assert any("10,500" in l or "10500" in l for l in lines)

    def test_report_lines_with_cop(self):
        s = self._snap(cop_equity=44_100_000.0, usd_cop=4200.0)
        lines = s.report_lines()
        assert any("COP" in l for l in lines)


# ---------------------------------------------------------------------------
# PortfolioMonitor
# ---------------------------------------------------------------------------


def _make_snapshot(equity: float, timestamp: datetime | None = None) -> MagicMock:
    s = MagicMock()
    s.equity = equity
    s.timestamp = timestamp or datetime(2026, 1, 2)
    s.cop_equity = None
    return s


class TestPortfolioMonitor:
    def test_empty_snapshots_returns_initial_equity(self):
        m = PortfolioMonitor([], initial_equity=10000.0)
        metrics = m.compute()
        assert metrics.equity == 10000.0
        assert metrics.total_return_pct == 0.0

    def test_positive_return(self):
        snaps = [_make_snapshot(10000.0), _make_snapshot(10500.0)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        metrics = m.compute()
        assert metrics.total_return_pct == pytest.approx(5.0)

    def test_negative_return(self):
        snaps = [_make_snapshot(10000.0), _make_snapshot(9500.0)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        metrics = m.compute()
        assert metrics.total_return_pct == pytest.approx(-5.0)

    def test_max_drawdown_calculated(self):
        # Sube a 12000 luego baja a 9600 → DD = (12000-9600)/12000 = 20%
        snaps = [
            _make_snapshot(10000.0),
            _make_snapshot(12000.0),
            _make_snapshot(9600.0),
        ]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        metrics = m.compute()
        assert metrics.max_drawdown_pct == pytest.approx(0.20, rel=1e-3)

    def test_current_drawdown(self):
        snaps = [_make_snapshot(10000.0), _make_snapshot(11000.0), _make_snapshot(9900.0)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        metrics = m.compute()
        # Peak=11000, current=9900 → DD = 10%
        assert metrics.current_drawdown_pct == pytest.approx(0.10, rel=1e-3)

    def test_cop_equity_computed_from_usd_cop(self):
        snaps = [_make_snapshot(10000.0), _make_snapshot(10500.0)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0, usd_cop=4200.0)
        metrics = m.compute()
        assert metrics.cop_equity == pytest.approx(10500.0 * 4200.0)

    def test_daily_return_with_multiple_snaps(self):
        snaps = [_make_snapshot(10000.0), _make_snapshot(10100.0), _make_snapshot(10200.0)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        metrics = m.compute()
        assert metrics.daily_return_pct == pytest.approx(100.0 / 101.0, rel=1e-3)

    def test_daily_pnl_series_length(self):
        snaps = [_make_snapshot(10000.0 + i * 100) for i in range(5)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        series = m.daily_pnl_series()
        assert len(series) == 4  # n-1 puntos

    def test_equity_at_returns_closest(self):
        t1 = datetime(2026, 6, 1)
        t2 = datetime(2026, 6, 2)
        snaps = [_make_snapshot(10000.0, t1), _make_snapshot(10500.0, t2)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        eq = m.equity_at(datetime(2026, 6, 2, 12, 0))
        assert eq == 10500.0

    def test_equity_at_before_all_snaps_returns_none(self):
        t1 = datetime(2026, 6, 10)
        snaps = [_make_snapshot(10000.0, t1)]
        m = PortfolioMonitor(snaps, initial_equity=10000.0)
        eq = m.equity_at(datetime(2026, 6, 1))
        assert eq is None
