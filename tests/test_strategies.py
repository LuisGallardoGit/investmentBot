"""Tests para las estrategias: TrendFollow, MeanReversion, Combined."""

import numpy as np
import pandas as pd
import pytest

from jarvis_bot.indicators import enrich
from jarvis_bot.strategies import CombinedStrategy, MeanReversionStrategy, TrendFollowStrategy, get_strategy
from jarvis_bot.strategies.base import Signal


def make_enriched_row(**overrides) -> pd.Series:
    """Fila base enriquecida con valores neutrales."""
    base = {
        "close": 100.0,
        "ema_fast": 100.0,
        "ema_slow": 100.0,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_line": 0.0,
        "macd_signal": 0.0,
        "bb_pct_b": 0.5,
        "bb_upper": 110.0,
        "bb_lower": 90.0,
        "volume_ratio": 1.0,
    }
    base.update(overrides)
    return pd.Series(base)


class TestTrendFollowStrategy:
    def setup_method(self):
        self.s = TrendFollowStrategy()

    def test_buy_on_ema_crossup_neutral(self):
        prev = make_enriched_row(ema_fast=99.0, ema_slow=100.0, macd_hist=0.1)
        curr = make_enriched_row(ema_fast=101.0, ema_slow=100.0, rsi=55.0, macd_hist=0.2)
        assert self.s.signal(prev, curr) == Signal.BUY

    def test_no_buy_overbought(self):
        prev = make_enriched_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=101.0, ema_slow=100.0, rsi=75.0)
        assert self.s.signal(prev, curr) != Signal.BUY

    def test_sell_on_ema_crossdown(self):
        prev = make_enriched_row(ema_fast=101.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=99.0, ema_slow=100.0, rsi=50.0)
        assert self.s.signal(prev, curr) == Signal.SELL

    def test_no_buy_bearish_sentiment(self):
        prev = make_enriched_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0)
        assert self.s.signal(prev, curr, sentiment=0.2) != Signal.BUY

    def test_no_buy_macro_danger(self):
        prev = make_enriched_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0)
        assert self.s.signal(prev, curr, macro_risk="high_risk") != Signal.BUY

    def test_hold_no_crossover(self):
        prev = make_enriched_row(ema_fast=101.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=102.0, ema_slow=100.0, rsi=50.0)
        assert self.s.signal(prev, curr) == Signal.HOLD

    def test_buy_oversold_uptrend(self):
        prev = make_enriched_row(ema_fast=101.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=102.0, ema_slow=100.0, rsi=25.0, macd_hist=0.1)
        assert self.s.signal(prev, curr) == Signal.BUY


class TestMeanReversionStrategy:
    def setup_method(self):
        self.s = MeanReversionStrategy()

    def test_buy_near_lower_band_oversold(self):
        prev = make_enriched_row(bb_pct_b=0.15)
        curr = make_enriched_row(bb_pct_b=0.1, rsi=30.0, volume_ratio=1.2)
        assert self.s.signal(prev, curr) == Signal.BUY

    def test_sell_near_upper_band_overbought(self):
        prev = make_enriched_row(bb_pct_b=0.85)
        curr = make_enriched_row(bb_pct_b=0.9, rsi=68.0)
        assert self.s.signal(prev, curr) == Signal.SELL

    def test_hold_price_in_middle(self):
        prev = make_enriched_row(bb_pct_b=0.5)
        curr = make_enriched_row(bb_pct_b=0.5, rsi=50.0)
        assert self.s.signal(prev, curr) == Signal.HOLD

    def test_no_buy_with_bearish_sentiment(self):
        prev = make_enriched_row(bb_pct_b=0.15)
        curr = make_enriched_row(bb_pct_b=0.1, rsi=30.0)
        assert self.s.signal(prev, curr, sentiment=0.15) != Signal.BUY

    def test_nan_bb_returns_hold(self):
        prev = make_enriched_row()
        curr = make_enriched_row(bb_pct_b=float("nan"))
        assert self.s.signal(prev, curr) == Signal.HOLD


class TestCombinedStrategy:
    def setup_method(self):
        self.s = CombinedStrategy(trend_weight=0.6, reversion_weight=0.4)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            CombinedStrategy(trend_weight=0.6, reversion_weight=0.5)

    def test_both_strategies_agree_buy(self):
        # EMA crossup (trend BUY) + near lower band oversold (reversion BUY)
        prev = make_enriched_row(ema_fast=99.0, ema_slow=100.0, bb_pct_b=0.15)
        curr = make_enriched_row(ema_fast=101.0, ema_slow=100.0, rsi=28.0, bb_pct_b=0.1, macd_hist=0.1, volume_ratio=1.1)
        assert self.s.signal(prev, curr) == Signal.BUY

    def test_both_strategies_agree_sell(self):
        prev = make_enriched_row(ema_fast=101.0, ema_slow=100.0, bb_pct_b=0.85)
        curr = make_enriched_row(ema_fast=99.0, ema_slow=100.0, rsi=72.0, bb_pct_b=0.9)
        assert self.s.signal(prev, curr) == Signal.SELL

    def test_explain_returns_all_keys(self):
        prev = make_enriched_row()
        curr = make_enriched_row()
        result = self.s.explain(prev, curr)
        for key in ("trend_signal", "reversion_signal", "score", "final_signal"):
            assert key in result

    def test_score_is_weighted(self):
        # trend=BUY(+1), reversion=HOLD(0) → score = 0.6*1 + 0.4*0 = 0.6
        prev = make_enriched_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_enriched_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0, macd_hist=0.1, bb_pct_b=0.5)
        result = self.s.explain(prev, curr)
        assert result["score"] == pytest.approx(0.6, abs=0.01)


class TestGetStrategy:
    def test_get_trend_follow(self):
        s = get_strategy("trend_follow")
        assert isinstance(s, TrendFollowStrategy)

    def test_get_mean_reversion(self):
        s = get_strategy("mean_reversion")
        assert isinstance(s, MeanReversionStrategy)

    def test_get_combined(self):
        s = get_strategy("combined")
        assert isinstance(s, CombinedStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="desconocida"):
            get_strategy("magic_ai_strategy")


class TestBacktesterIntegration:
    """Tests de integración del backtester con fixtures sintéticos."""

    def _make_fixture_df(self, n: int = 200) -> pd.DataFrame:
        np.random.seed(42)
        closes = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
        closes = np.maximum(closes, 10.0)
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": closes,
            "high": closes + np.abs(np.random.randn(n)),
            "low": closes - np.abs(np.random.randn(n)),
            "close": closes,
            "volume": np.random.randint(500_000, 2_000_000, size=n).astype(float),
        })

    def test_backtest_runs_without_error(self):
        from jarvis_bot.backtester import walk_forward
        df = enrich(self._make_fixture_df())
        strategies = [TrendFollowStrategy(), MeanReversionStrategy(), CombinedStrategy()]
        report = walk_forward(df, strategies, symbol="TEST", n_splits=3)
        assert report.best_strategy in ("trend_follow", "mean_reversion", "combined")

    def test_report_has_results_for_all_strategies(self):
        from jarvis_bot.backtester import walk_forward
        df = enrich(self._make_fixture_df())
        strategies = [TrendFollowStrategy(), CombinedStrategy()]
        report = walk_forward(df, strategies, symbol="TEST", n_splits=2)
        names = {r.strategy_name for r in report.results}
        assert "trend_follow" in names
        assert "combined" in names

    def test_sim_result_equity_is_positive(self):
        from jarvis_bot.backtester import walk_forward
        df = enrich(self._make_fixture_df())
        report = walk_forward(df, [TrendFollowStrategy()], symbol="TEST", n_splits=2)
        for r in report.results:
            assert r.final_equity > 0
