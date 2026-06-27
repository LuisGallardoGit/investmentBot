"""Tests para ml/features.py — ingeniería de features ML — Fase 6."""

import numpy as np
import pandas as pd
import pytest

from jarvis_bot.ml.features import (
    FEATURE_NAMES,
    REQUIRED_INDICATOR_COLS,
    FeatureConfig,
    FeatureMatrix,
    build_feature_matrix,
    build_features,
    build_targets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """DataFrame sintético con todas las columnas requeridas por enrich()."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)  # no negativos
    df = pd.DataFrame({
        "close": close,
        "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        "rsi": rng.uniform(20, 80, n),
        "macd_hist": rng.normal(0, 0.5, n),
        "macd_line": rng.normal(0, 1.0, n),
        "bb_pct_b": rng.uniform(-0.1, 1.1, n),
        "bb_width": rng.uniform(0.01, 0.1, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n),
        "ema_fast": close * rng.uniform(0.98, 1.02, n),
        "ema_slow": close * rng.uniform(0.95, 1.05, n),
        "atr": close * rng.uniform(0.005, 0.02, n),
        "atr_pct": rng.uniform(0.005, 0.02, n),
    })
    return df


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


class TestFeatureConfig:
    def test_defaults(self):
        cfg = FeatureConfig()
        assert cfg.forward_horizon == 5
        assert cfg.target_threshold_pct == 1.0
        assert cfg.include_optional is True
        assert cfg.drop_na is True

    def test_custom(self):
        cfg = FeatureConfig(forward_horizon=10, target_threshold_pct=2.0)
        assert cfg.forward_horizon == 10
        assert cfg.target_threshold_pct == 2.0


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = _make_df()
        feat = build_features(df)
        assert isinstance(feat, pd.DataFrame)

    def test_same_index_as_input(self):
        df = _make_df()
        feat = build_features(df)
        assert feat.index.equals(df.index)

    def test_core_feature_names_present(self):
        df = _make_df()
        feat = build_features(df)
        for name in FEATURE_NAMES:
            assert name in feat.columns, f"Feature '{name}' faltante"

    def test_atr_pct_included_when_present(self):
        df = _make_df()
        feat = build_features(df, FeatureConfig(include_optional=True))
        assert "atr_pct" in feat.columns

    def test_atr_pct_excluded_when_disabled(self):
        df = _make_df()
        feat = build_features(df, FeatureConfig(include_optional=False))
        assert "atr_pct" not in feat.columns

    def test_atr_pct_excluded_when_not_in_df(self):
        df = _make_df().drop(columns=["atr_pct"])
        feat = build_features(df, FeatureConfig(include_optional=True))
        assert "atr_pct" not in feat.columns

    def test_ema_crossover_computed_correctly(self):
        df = _make_df()
        feat = build_features(df)
        expected = (df["ema_fast"] - df["ema_slow"]) / df["close"]
        pd.testing.assert_series_equal(feat["ema_crossover"], expected, check_names=False)

    def test_return_1d_is_log_return(self):
        df = _make_df(n=20)
        feat = build_features(df)
        expected_r1 = np.log(df["close"] / df["close"].shift(1))
        pd.testing.assert_series_equal(feat["return_1d"], expected_r1, check_names=False)

    def test_missing_required_column_raises(self):
        df = _make_df().drop(columns=["rsi"])
        with pytest.raises(ValueError, match="rsi"):
            build_features(df)

    def test_rsi_lag1_is_shifted(self):
        df = _make_df(n=20)
        feat = build_features(df)
        np.testing.assert_array_almost_equal(
            feat["rsi_lag1"].iloc[1:].values,
            df["rsi"].iloc[:-1].values,
        )

    def test_close_std_5d_is_rolling(self):
        df = _make_df(n=30)
        feat = build_features(df)
        assert feat["close_std_5d"].isna().sum() > 0  # primeras filas NaN


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------


class TestBuildTargets:
    def test_returns_series(self):
        df = _make_df()
        t = build_targets(df)
        assert isinstance(t, pd.Series)

    def test_same_length_as_input(self):
        df = _make_df(n=50)
        t = build_targets(df)
        assert len(t) == len(df)

    def test_last_rows_are_nan(self):
        df = _make_df(n=50)
        t = build_targets(df, forward_horizon=5)
        assert t.iloc[-5:].isna().all()

    def test_values_are_minus_one_zero_or_one(self):
        df = _make_df(n=100)
        t = build_targets(df).dropna()
        assert set(t.unique()).issubset({-1, 0, 1})

    def test_buy_when_price_rises_above_threshold(self):
        n = 20
        close = pd.Series([100.0] * n)
        close.iloc[5] = 103.0  # +3% → horizon 5, row 0 should be BUY
        df = pd.DataFrame({"close": close})
        t = build_targets(df, forward_horizon=5, threshold_pct=1.0)
        # row 0 forward price is row 5 → +3% → BUY
        assert int(t.iloc[0]) == 1

    def test_sell_when_price_falls_below_threshold(self):
        n = 20
        close = pd.Series([100.0] * n)
        close.iloc[5] = 97.0  # -3%
        df = pd.DataFrame({"close": close})
        t = build_targets(df, forward_horizon=5, threshold_pct=1.0)
        assert int(t.iloc[0]) == -1

    def test_hold_when_price_within_threshold(self):
        n = 20
        close = pd.Series([100.0] * n)
        close.iloc[5] = 100.5  # +0.5% < 1% threshold
        df = pd.DataFrame({"close": close})
        t = build_targets(df, forward_horizon=5, threshold_pct=1.0)
        assert int(t.iloc[0]) == 0

    def test_custom_threshold(self):
        n = 20
        close = pd.Series([100.0] * n)
        close.iloc[3] = 101.5  # +1.5% > 0.5% threshold but < 2%
        df = pd.DataFrame({"close": close})
        t = build_targets(df, forward_horizon=3, threshold_pct=0.5)
        assert int(t.iloc[0]) == 1
        t2 = build_targets(df, forward_horizon=3, threshold_pct=2.0)
        assert int(t2.iloc[0]) == 0


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def test_returns_feature_matrix(self):
        df = _make_df(n=100)
        fm = build_feature_matrix(df)
        assert isinstance(fm, FeatureMatrix)

    def test_x_and_y_same_length(self):
        df = _make_df(n=100)
        fm = build_feature_matrix(df)
        assert len(fm.X) == len(fm.y)

    def test_no_nan_when_drop_na_true(self):
        df = _make_df(n=100)
        fm = build_feature_matrix(df, FeatureConfig(drop_na=True))
        assert not fm.X.isna().any().any()
        assert not fm.y.isna().any()

    def test_fewer_rows_than_input_due_to_warmup_and_horizon(self):
        df = _make_df(n=100)
        fm = build_feature_matrix(df)
        assert len(fm.X) < len(df)

    def test_feature_names_match_columns(self):
        df = _make_df(n=100)
        fm = build_feature_matrix(df)
        assert fm.feature_names == list(fm.X.columns)

    def test_counts_sum_to_total(self):
        df = _make_df(n=200)
        fm = build_feature_matrix(df)
        assert fm.n_buy + fm.n_sell + fm.n_hold == len(fm.X)

    def test_class_balance_percentages_sum_to_100(self):
        df = _make_df(n=200)
        fm = build_feature_matrix(df)
        bal = fm.class_balance()
        total_pct = sum(
            float(v.split("(")[1].rstrip("%)")) for v in bal.values()
        )
        assert abs(total_pct - 100.0) < 0.5

    def test_valid_index_aligns_with_x(self):
        df = _make_df(n=100)
        fm = build_feature_matrix(df)
        assert fm.valid_index.equals(fm.X.index)
