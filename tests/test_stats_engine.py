"""Tests para stats_engine — funciones puras, fáciles de verificar."""

import pytest
from jarvis_bot import stats_engine


class TestReturnsFromEquity:
    def test_simple_returns(self):
        curve = [100.0, 110.0, 99.0]
        rets = stats_engine.returns_from_equity(curve)
        assert rets[0] == pytest.approx(0.10)
        assert rets[1] == pytest.approx(-0.10, abs=1e-9)

    def test_single_point_returns_empty(self):
        assert len(stats_engine.returns_from_equity([100.0])) == 0

    def test_empty_returns_empty(self):
        assert len(stats_engine.returns_from_equity([])) == 0


class TestMaxDrawdown:
    def test_flat_curve_zero_drawdown(self):
        assert stats_engine.max_drawdown([100.0, 100.0, 100.0]) == pytest.approx(0.0)

    def test_monotone_increase_zero_drawdown(self):
        assert stats_engine.max_drawdown([100.0, 110.0, 120.0]) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # pico=110, valle=88 → dd = (110-88)/110 ≈ 0.2
        curve = [100.0, 110.0, 88.0]
        dd = stats_engine.max_drawdown(curve)
        assert dd == pytest.approx((110.0 - 88.0) / 110.0)

    def test_empty_curve_zero(self):
        assert stats_engine.max_drawdown([]) == pytest.approx(0.0)


class TestSharpeRatio:
    def test_zero_returns_zero_sharpe(self):
        returns = [0.0, 0.0, 0.0]
        assert stats_engine.sharpe_ratio(returns) == pytest.approx(0.0)

    def test_empty_returns_zero_sharpe(self):
        assert stats_engine.sharpe_ratio([]) == pytest.approx(0.0)

    def test_positive_consistent_returns_positive_sharpe(self):
        returns = [0.001] * 252  # retorno diario fijo positivo
        sharpe = stats_engine.sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0.0

    def test_single_return_zero_sharpe(self):
        assert stats_engine.sharpe_ratio([0.01]) == pytest.approx(0.0)


class TestCagr:
    def test_flat_equity_zero_cagr(self):
        assert stats_engine.cagr([100.0, 100.0]) == pytest.approx(0.0)

    def test_known_cagr(self):
        # 252 periodos de retorno diario 0.001 → CAGR ≈ e^(252*0.001) - 1 ≈ 28.4%
        import numpy as np
        curve = list(100.0 * np.exp(np.cumsum([0.001] * 253)))
        c = stats_engine.cagr(curve, periods_per_year=252)
        assert c == pytest.approx(0.284, abs=0.02)

    def test_empty_curve_zero(self):
        assert stats_engine.cagr([]) == pytest.approx(0.0)


class TestSummarize:
    def test_returns_all_keys(self):
        result = stats_engine.summarize([100.0, 110.0, 105.0, 115.0])
        for key in ("sharpe_ratio", "max_drawdown", "cagr", "total_return", "n_periods"):
            assert key in result

    def test_total_return_correct(self):
        result = stats_engine.summarize([100.0, 200.0])
        assert result["total_return"] == pytest.approx(1.0)
