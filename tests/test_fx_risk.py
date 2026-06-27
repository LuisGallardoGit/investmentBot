"""Tests para el módulo de riesgo FX COP/USD."""

import pytest

from jarvis_bot.modules.fx_risk import FXAdjustedPnL, FXRiskManager, FXState


class TestFXState:
    def test_str_positive_change(self):
        s = FXState(usd_cop=4200.0, change_30d_pct=3.5, risk_level="normal")
        assert "4200" in str(s)
        assert "+3.5%" in str(s)

    def test_str_negative_change(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=-2.1, risk_level="elevated")
        assert "-2.1%" in str(s)

    def test_cop_per_usd_alias(self):
        s = FXState(usd_cop=4150.0, change_30d_pct=0.0, risk_level="normal")
        assert s.cop_per_usd == 4150.0


class TestFXAdjustedPnL:
    def test_no_fx_movement_pnl_cop_equals_pnl_usd_times_rate(self):
        # Misma tasa entrada/salida → PnL COP = PnL USD × tasa
        result = FXAdjustedPnL.calculate(
            pnl_usd=100.0,
            entry_usd_cop=4000.0,
            exit_usd_cop=4000.0,
            position_usd=1000.0,
        )
        assert result.pnl_cop == pytest.approx(100.0 * 4000.0, rel=1e-6)
        assert result.fx_impact_cop == pytest.approx(0.0)

    def test_cop_depreciation_amplifies_pnl(self):
        # COP se depreció: más pesos por dólar → PnL en COP > PnL USD × tasa_entrada
        result = FXAdjustedPnL.calculate(
            pnl_usd=100.0,
            entry_usd_cop=4000.0,
            exit_usd_cop=4200.0,  # depreció 5%
            position_usd=1000.0,
        )
        # FX impact = 1000 × (4200 - 4000) = 200_000 COP extra
        assert result.fx_impact_cop == pytest.approx(200_000.0)
        assert result.pnl_cop > result.pnl_usd * 4000.0

    def test_cop_appreciation_reduces_pnl(self):
        # COP se apreció: menos pesos por dólar → PnL en COP < PnL USD × tasa_entrada
        result = FXAdjustedPnL.calculate(
            pnl_usd=100.0,
            entry_usd_cop=4200.0,
            exit_usd_cop=4000.0,  # apreció ~4.8%
            position_usd=1000.0,
        )
        assert result.fx_impact_cop < 0  # pérdida por FX
        assert result.pnl_cop < result.pnl_usd * 4200.0

    def test_zero_pnl_usd_only_fx_impact(self):
        result = FXAdjustedPnL.calculate(
            pnl_usd=0.0,
            entry_usd_cop=4000.0,
            exit_usd_cop=4100.0,
            position_usd=1000.0,
        )
        assert result.pnl_cop == pytest.approx(1000.0 * 100.0)  # solo FX
        assert result.pnl_usd == 0.0


class TestFXRiskManagerEvaluate:
    def setup_method(self):
        self.mgr = FXRiskManager(fx_vol_caution_pct=5.0, fx_vol_block_pct=10.0)

    def test_normal_risk_below_caution(self):
        state = self.mgr.evaluate(4000.0, 3.0)
        assert state.risk_level == "normal"

    def test_elevated_risk_between_caution_and_block(self):
        state = self.mgr.evaluate(4000.0, 7.0)
        assert state.risk_level == "elevated"

    def test_high_risk_above_block(self):
        state = self.mgr.evaluate(4000.0, 11.0)
        assert state.risk_level == "high"

    def test_negative_change_uses_abs_value(self):
        # -8% en términos absolutos → elevated (entre 5 y 10)
        state = self.mgr.evaluate(4000.0, -8.0)
        assert state.risk_level == "elevated"

    def test_exactly_at_caution_threshold(self):
        state = self.mgr.evaluate(4000.0, 5.0)
        assert state.risk_level == "elevated"

    def test_exactly_at_block_threshold(self):
        state = self.mgr.evaluate(4000.0, 10.0)
        assert state.risk_level == "high"


class TestFXRiskManagerPositionSize:
    def setup_method(self):
        self.mgr = FXRiskManager()

    def test_normal_multiplier_is_1(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=2.0, risk_level="normal")
        assert self.mgr.position_size_multiplier(s) == 1.0

    def test_elevated_multiplier_is_half(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=7.0, risk_level="elevated")
        assert self.mgr.position_size_multiplier(s) == 0.5

    def test_high_multiplier_is_zero(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=12.0, risk_level="high")
        assert self.mgr.position_size_multiplier(s) == 0.0

    def test_allows_entry_normal(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=2.0, risk_level="normal")
        assert self.mgr.allows_new_entry(s) is True

    def test_allows_entry_elevated(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=7.0, risk_level="elevated")
        assert self.mgr.allows_new_entry(s) is True  # cautela pero no bloqueo

    def test_blocks_entry_high(self):
        s = FXState(usd_cop=4000.0, change_30d_pct=12.0, risk_level="high")
        assert self.mgr.allows_new_entry(s) is False


class TestFXRiskManagerTracking:
    def setup_method(self):
        self.mgr = FXRiskManager()

    def test_record_and_retrieve_entry_rate(self):
        self.mgr.record_entry("AAPL", 4150.0)
        assert self.mgr.entry_rate("AAPL") == 4150.0

    def test_entry_rate_unknown_symbol_returns_none(self):
        assert self.mgr.entry_rate("UNKNOWN") is None

    def test_record_exit_removes_entry(self):
        self.mgr.record_entry("VOO", 4000.0)
        self.mgr.record_exit("VOO")
        assert self.mgr.entry_rate("VOO") is None

    def test_portfolio_value_cop_without_rate_returns_none(self):
        assert self.mgr.portfolio_value_cop(10000.0) is None

    def test_portfolio_value_cop_with_rate(self):
        self.mgr.evaluate(4200.0, 0.0)  # establece tasa interna
        result = self.mgr.portfolio_value_cop(10000.0)
        assert result == pytest.approx(10000.0 * 4200.0)

    def test_adjusted_pnl_uses_entry_rate(self):
        self.mgr.record_entry("AAPL", 4000.0)
        self.mgr.evaluate(4200.0, 5.0)
        result = self.mgr.adjusted_pnl("AAPL", pnl_usd=50.0, position_usd=1000.0, current_usd_cop=4200.0)
        assert result.entry_rate == 4000.0
        assert result.exit_rate == 4200.0

    def test_adjusted_pnl_no_entry_uses_current_rate(self):
        result = self.mgr.adjusted_pnl("QQQ", pnl_usd=0.0, position_usd=1000.0, current_usd_cop=4100.0)
        # Sin entry registrada, usa tasa actual → impacto FX = 0
        assert result.fx_impact_cop == pytest.approx(0.0)

    def test_fx_summary_includes_keys(self):
        self.mgr.evaluate(4100.0, 3.0)
        summary = self.mgr.fx_summary(10000.0)
        assert "usd_value" in summary
        assert "usd_cop_rate" in summary
        assert "cop_value" in summary
