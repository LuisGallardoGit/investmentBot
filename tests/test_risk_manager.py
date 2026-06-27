"""Tests para RiskManager."""

import pytest
from jarvis_bot.modules.risk_manager import RiskManager, SizingResult


@pytest.fixture
def rm():
    return RiskManager(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.04,
        max_drawdown_limit=0.15,
        max_position_pct=0.25,
    )


class TestInit:
    def test_rejects_zero_capital(self):
        with pytest.raises(ValueError, match="initial_capital"):
            RiskManager(initial_capital=0, risk_per_trade=0.01, stop_loss_pct=0.05)

    def test_rejects_invalid_risk_per_trade(self):
        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskManager(initial_capital=10_000, risk_per_trade=1.5, stop_loss_pct=0.05)

    def test_rejects_invalid_stop_loss(self):
        with pytest.raises(ValueError, match="stop_loss_pct"):
            RiskManager(initial_capital=10_000, risk_per_trade=0.01, stop_loss_pct=0.0)


class TestPositionSize:
    def test_returns_sizing_result(self, rm):
        result = rm.position_size(equity=10_000.0, price=100.0)
        assert isinstance(result, SizingResult)

    def test_approved_normal_case(self, rm):
        result = rm.position_size(equity=10_000.0, price=100.0)
        assert result.approved is True
        assert result.quantity >= 1

    def test_respects_risk_per_trade(self, rm):
        # risk_amount = equity * risk_per_trade = 10000 * 0.01 = 100
        # qty_by_risk = 100 / (100 * 0.04) = 25
        # qty_by_cap  = (10000 * 0.25) / 100 = 25
        result = rm.position_size(equity=10_000.0, price=100.0)
        assert result.quantity == 25.0

    def test_cap_limits_large_position(self, rm):
        # precio muy bajo: qty_by_risk sería enorme, pero cap lo limita
        result = rm.position_size(equity=10_000.0, price=1.0)
        max_allowed = (10_000.0 * 0.25) / 1.0  # = 2500
        assert result.quantity <= max_allowed

    def test_zero_equity_not_approved(self, rm):
        result = rm.position_size(equity=0.0, price=100.0)
        assert result.approved is False

    def test_zero_price_not_approved(self, rm):
        result = rm.position_size(equity=10_000.0, price=0.0)
        assert result.approved is False

    def test_very_expensive_stock_not_approved(self, rm):
        # precio tan alto que qty = 0
        result = rm.position_size(equity=100.0, price=100_000.0)
        assert result.approved is False


class TestDrawdownBreach:
    def test_no_breach_below_limit(self, rm):
        # equity plana: drawdown = 0
        curve = [10_000.0, 10_100.0, 10_050.0, 10_200.0]
        assert rm.drawdown_breach(curve) is False

    def test_breach_above_limit(self, rm):
        # cae de 10000 a 8000 = 20% drawdown > 15%
        curve = [10_000.0, 9_500.0, 8_000.0]
        assert rm.drawdown_breach(curve) is True

    def test_empty_curve_no_breach(self, rm):
        assert rm.drawdown_breach([]) is False

    def test_exactly_at_limit_is_breach(self, rm):
        # 15% exacto debe activar el circuit breaker
        curve = [10_000.0, 8_500.0]
        assert rm.drawdown_breach(curve) is True
