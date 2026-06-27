"""Tests para PositionReconciler — comparación estado local vs Alpaca."""

from unittest.mock import MagicMock, patch

import pytest

from jarvis_bot.modules.alpaca_executor import AlpacaExecutor
from jarvis_bot.modules.position_reconciler import (
    DiscrepancyType,
    PositionDiscrepancy,
    PositionReconciler,
    ReconciliationReport,
)


def _make_executor(configured: bool = True) -> AlpacaExecutor:
    ex = MagicMock(spec=AlpacaExecutor)
    ex.is_configured.return_value = configured
    return ex


def _make_raw_position(symbol: str, qty: float, avg_price: float = 150.0, market_value: float = 0.0) -> dict:
    return {
        "symbol": symbol,
        "qty": str(qty),
        "avg_entry_price": str(avg_price),
        "market_value": str(market_value or qty * avg_price),
    }


# ---------------------------------------------------------------------------
# PositionDiscrepancy
# ---------------------------------------------------------------------------


class TestPositionDiscrepancy:
    def test_str_match(self):
        d = PositionDiscrepancy("AAPL", 10.0, 10.0, DiscrepancyType.MATCH)
        assert "[OK]" in str(d)
        assert "AAPL" in str(d)

    def test_str_local_only(self):
        d = PositionDiscrepancy("AAPL", 10.0, None, DiscrepancyType.LOCAL_ONLY)
        assert "LOCAL_ONLY" in str(d)

    def test_str_remote_only(self):
        d = PositionDiscrepancy("VOO", None, 5.0, DiscrepancyType.REMOTE_ONLY)
        assert "REMOTE_ONLY" in str(d)

    def test_str_mismatch(self):
        d = PositionDiscrepancy("QQQ", 10.0, 7.0, DiscrepancyType.MISMATCH)
        assert "MISMATCH" in str(d)


# ---------------------------------------------------------------------------
# ReconciliationReport
# ---------------------------------------------------------------------------


class TestReconciliationReport:
    def _make_report(self, discrepancies):
        return ReconciliationReport(
            local_positions={"AAPL": 10.0},
            remote_positions={"AAPL": 10.0},
            discrepancies=discrepancies,
        )

    def test_no_discrepancies(self):
        r = self._make_report([
            PositionDiscrepancy("AAPL", 10.0, 10.0, DiscrepancyType.MATCH)
        ])
        assert not r.has_discrepancies
        assert r.n_matches == 1
        assert r.n_issues == 0

    def test_has_discrepancies(self):
        r = self._make_report([
            PositionDiscrepancy("AAPL", 10.0, 10.0, DiscrepancyType.MATCH),
            PositionDiscrepancy("VOO", 5.0, None, DiscrepancyType.LOCAL_ONLY),
        ])
        assert r.has_discrepancies
        assert r.n_issues == 1

    def test_summary_contains_discrepancy_info(self):
        r = self._make_report([
            PositionDiscrepancy("VOO", 5.0, None, DiscrepancyType.LOCAL_ONLY),
        ])
        summary = r.summary()
        assert "VOO" in summary
        assert "LOCAL_ONLY" in summary


# ---------------------------------------------------------------------------
# PositionReconciler.reconcile
# ---------------------------------------------------------------------------


class TestReconcile:
    def setup_method(self):
        self.ex = _make_executor(configured=True)

    def test_match_when_positions_equal(self):
        self.ex.get_positions.return_value = [
            _make_raw_position("AAPL", 10.0),
        ]
        r = PositionReconciler(self.ex)
        report = r.reconcile({"AAPL": 10.0})
        assert not report.has_discrepancies
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.MATCH

    def test_mismatch_when_qty_differs(self):
        self.ex.get_positions.return_value = [
            _make_raw_position("AAPL", 5.0),
        ]
        r = PositionReconciler(self.ex)
        report = r.reconcile({"AAPL": 10.0})
        assert report.has_discrepancies
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.MISMATCH

    def test_local_only_when_not_in_alpaca(self):
        self.ex.get_positions.return_value = []
        r = PositionReconciler(self.ex)
        report = r.reconcile({"AAPL": 10.0})
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.LOCAL_ONLY

    def test_remote_only_when_not_in_local(self):
        self.ex.get_positions.return_value = [
            _make_raw_position("AAPL", 10.0),
        ]
        r = PositionReconciler(self.ex)
        report = r.reconcile({})  # estado local vacío
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.REMOTE_ONLY

    def test_multiple_symbols(self):
        self.ex.get_positions.return_value = [
            _make_raw_position("AAPL", 10.0),
            _make_raw_position("VOO", 5.0),
        ]
        r = PositionReconciler(self.ex)
        report = r.reconcile({"AAPL": 10.0, "VOO": 5.0})
        assert report.n_matches == 2
        assert not report.has_discrepancies

    def test_empty_when_not_configured(self):
        ex = _make_executor(configured=False)
        r = PositionReconciler(ex)
        report = r.reconcile({"AAPL": 10.0})
        # Sin conexión: nada remoto → LOCAL_ONLY
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.LOCAL_ONLY

    def test_qty_tolerance_respected(self):
        """Diferencia dentro de tolerancia se clasifica como MATCH."""
        self.ex.get_positions.return_value = [
            _make_raw_position("AAPL", 10.0005),  # diff < 0.001
        ]
        r = PositionReconciler(self.ex, qty_tolerance=0.001)
        report = r.reconcile({"AAPL": 10.0})
        assert report.discrepancies[0].discrepancy_type == DiscrepancyType.MATCH

    def test_zero_qty_local_ignored(self):
        """Posiciones locales con qty=0 no se comparan (no son posiciones reales)."""
        self.ex.get_positions.return_value = []
        r = PositionReconciler(self.ex)
        report = r.reconcile({"AAPL": 0.0})  # qty 0 = no hay posición
        assert len(report.discrepancies) == 0


# ---------------------------------------------------------------------------
# PositionReconciler.positions_from_alpaca
# ---------------------------------------------------------------------------


class TestPositionsFromAlpaca:
    def test_returns_positions_from_alpaca(self):
        ex = _make_executor(configured=True)
        ex.get_positions.return_value = [
            _make_raw_position("AAPL", 10.0),
            _make_raw_position("VOO", 5.0),
        ]
        r = PositionReconciler(ex)
        positions = r.positions_from_alpaca()
        assert positions == {"AAPL": 10.0, "VOO": 5.0}

    def test_returns_empty_when_no_positions(self):
        ex = _make_executor(configured=True)
        ex.get_positions.return_value = []
        r = PositionReconciler(ex)
        assert r.positions_from_alpaca() == {}

    def test_returns_empty_when_not_configured(self):
        ex = _make_executor(configured=False)
        r = PositionReconciler(ex)
        assert r.positions_from_alpaca() == {}


# ---------------------------------------------------------------------------
# PositionReconciler.account_equity
# ---------------------------------------------------------------------------


class TestAccountEquity:
    def test_returns_equity_from_account(self):
        ex = _make_executor(configured=True)
        ex.get_account.return_value = {"equity": "12345.67"}
        r = PositionReconciler(ex)
        assert r.account_equity() == pytest.approx(12345.67)

    def test_returns_portfolio_value_if_no_equity_key(self):
        ex = _make_executor(configured=True)
        ex.get_account.return_value = {"portfolio_value": "9876.00"}
        r = PositionReconciler(ex)
        assert r.account_equity() == pytest.approx(9876.0)

    def test_returns_none_when_not_configured(self):
        ex = _make_executor(configured=True)
        ex.get_account.return_value = {}
        r = PositionReconciler(ex)
        assert r.account_equity() is None
