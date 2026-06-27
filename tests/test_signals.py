"""Tests para generate_signal del engine."""

import pandas as pd
import pytest

from jarvis_bot.engine import generate_signal


def make_row(ema_fast, ema_slow, rsi=50.0):
    return pd.Series({"ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi})


class TestCrossUp:
    def test_buy_on_cross_up_neutral_sentiment(self):
        prev = make_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0)
        assert generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30) == "BUY"

    def test_no_buy_if_rsi_overbought(self):
        prev = make_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_row(ema_fast=101.0, ema_slow=100.0, rsi=75.0)
        assert generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30) != "BUY"

    def test_no_buy_if_bearish_extreme_sentiment(self):
        prev = make_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0)
        result = generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30, sentiment_score=0.2)
        assert result != "BUY"

    def test_no_buy_if_macro_high_risk(self):
        prev = make_row(ema_fast=99.0, ema_slow=100.0)
        curr = make_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0)
        result = generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30, macro_risk="high_risk")
        assert result != "BUY"


class TestCrossDown:
    def test_sell_on_cross_down(self):
        prev = make_row(ema_fast=101.0, ema_slow=100.0)
        curr = make_row(ema_fast=99.0, ema_slow=100.0, rsi=50.0)
        assert generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30) == "SELL"

    def test_sell_on_overbought_rsi(self):
        prev = make_row(ema_fast=100.0, ema_slow=99.0)
        curr = make_row(ema_fast=101.0, ema_slow=99.0, rsi=80.0)
        assert generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30) == "SELL"

    def test_sell_on_bearish_extreme_with_uptrend(self):
        # Sentimiento muy bajista fuerza salida aunque haya uptrend
        prev = make_row(ema_fast=100.0, ema_slow=99.0)
        curr = make_row(ema_fast=101.0, ema_slow=99.0, rsi=60.0)
        result = generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30, sentiment_score=0.2)
        assert result == "SELL"


class TestHold:
    def test_hold_when_no_cross_neutral(self):
        # EMA fast por encima pero sin cruce — sin señal fuerte
        prev = make_row(ema_fast=101.0, ema_slow=100.0)
        curr = make_row(ema_fast=102.0, ema_slow=100.0, rsi=50.0)
        result = generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30)
        assert result == "HOLD"

    def test_hold_when_nan_indicators(self):
        prev = make_row(ema_fast=float("nan"), ema_slow=100.0)
        curr = make_row(ema_fast=101.0, ema_slow=100.0, rsi=50.0)
        assert generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30) == "HOLD"


class TestOversoldBuy:
    def test_buy_on_oversold_rsi_with_uptrend(self):
        # RSI sobrevendido + fast > slow → BUY aunque no haya cruce
        prev = make_row(ema_fast=101.0, ema_slow=100.0)
        curr = make_row(ema_fast=102.0, ema_slow=100.0, rsi=25.0)
        result = generate_signal(prev, curr, rsi_overbought=70, rsi_oversold=30)
        assert result == "BUY"
