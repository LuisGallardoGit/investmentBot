"""Tests para indicators.py — funciones puras."""

import numpy as np
import pandas as pd
import pytest

from jarvis_bot.indicators import (
    atr,
    atr_pct,
    bb_pct_b,
    bollinger_bands,
    ema,
    enrich,
    macd,
    rsi,
    sma,
    volume_ratio,
)


def make_close(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def make_ohlcv(n: int = 50, start: float = 100.0) -> pd.DataFrame:
    np.random.seed(42)
    closes = start + np.cumsum(np.random.randn(n) * 2)
    highs = closes + np.abs(np.random.randn(n))
    lows = closes - np.abs(np.random.randn(n))
    volumes = np.random.randint(100_000, 1_000_000, size=n).astype(float)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestEma:
    def test_output_length_matches_input(self):
        s = make_close([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(ema(s, span=3)) == len(s)

    def test_ema_approaches_value_for_constant_series(self):
        s = make_close([100.0] * 50)
        result = ema(s, span=9)
        assert result.iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_fast_ema_reacts_faster_than_slow(self):
        # Serie que sube de golpe: EMA rápida debe alcanzar el precio nuevo más rápido
        s = make_close([100.0] * 20 + [200.0] * 20)
        fast = ema(s, span=5)
        slow = ema(s, span=20)
        # Al final, ambas deben estar cerca de 200, pero fast debe estar más cerca
        assert abs(fast.iloc[-1] - 200) < abs(slow.iloc[-1] - 200)


class TestRsi:
    def test_rsi_range(self):
        s = make_close(list(range(1, 51)))
        r = rsi(s, period=14)
        assert r.dropna().between(0, 100).all()

    def test_rsi_uptrend_above_50(self):
        # Serie consistentemente creciente → RSI debe ser alto
        s = make_close([float(i) for i in range(1, 51)])
        r = rsi(s, period=14)
        assert r.iloc[-1] > 50

    def test_rsi_downtrend_below_50(self):
        s = make_close([float(50 - i) for i in range(50)])
        r = rsi(s, period=14)
        assert r.iloc[-1] < 50


class TestMacd:
    def test_output_lengths(self):
        s = make_close(list(range(1, 51)))
        ml, sl, mh = macd(s)
        assert len(ml) == len(sl) == len(mh) == len(s)

    def test_histogram_is_macd_minus_signal(self):
        s = make_close(list(range(1, 51)))
        ml, sl, mh = macd(s)
        pd.testing.assert_series_equal(mh, ml - sl)


class TestBollingerBands:
    def test_upper_above_lower(self):
        s = make_close(list(range(1, 51)))
        upper, mid, lower = bollinger_bands(s, window=10)
        valid = upper.dropna()
        valid_lower = lower.dropna()
        assert (valid.values >= valid_lower.values).all()

    def test_pct_b_on_mid_is_0_5(self):
        # Si el precio es exactamente la media, %B debe ser ~0.5
        s = pd.Series([100.0] * 30)
        pct = bb_pct_b(s, window=10)
        # Con serie constante la std es 0 → fallback a 0.5
        assert pct.dropna().iloc[-1] == pytest.approx(0.5, abs=0.01)


class TestAtr:
    def test_atr_positive(self):
        df = make_ohlcv(50)
        result = atr(df, period=14)
        assert result.dropna().gt(0).all()

    def test_atr_pct_between_0_and_1(self):
        df = make_ohlcv(50)
        result = atr_pct(df, period=14)
        assert result.dropna().between(0, 1).all()


class TestVolumeRatio:
    def test_constant_volume_ratio_is_1(self):
        s = pd.Series([1000.0] * 30)
        ratio = volume_ratio(s, window=10)
        assert ratio.dropna().iloc[-1] == pytest.approx(1.0)

    def test_spike_gives_ratio_above_1(self):
        s = pd.Series([100.0] * 25 + [500.0])
        ratio = volume_ratio(s, window=20)
        assert ratio.iloc[-1] > 1.0


class TestEnrich:
    def test_enrich_adds_all_columns(self):
        df = make_ohlcv(60)
        result = enrich(df)
        expected_cols = [
            "ema_fast", "ema_slow", "rsi",
            "macd_line", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower", "bb_pct_b", "bb_width",
            "atr", "atr_pct", "volume_ratio",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Falta columna: {col}"

    def test_enrich_preserves_original_columns(self):
        df = make_ohlcv(60)
        result = enrich(df)
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_enrich_same_row_count(self):
        df = make_ohlcv(60)
        result = enrich(df)
        assert len(result) == len(df)
