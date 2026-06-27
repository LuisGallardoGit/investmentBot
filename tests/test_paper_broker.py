"""Tests para PaperBroker."""

from datetime import datetime

import pytest

from jarvis_bot.brokers import Order, OrderSide, PaperBroker
from jarvis_bot.brokers.paper import InsufficientCashError, InsufficientPositionError

NOW = datetime(2025, 1, 1, 10, 0, 0)


def make_order(symbol="AAPL", side=OrderSide.BUY, qty=10.0, price=100.0):
    return Order(symbol=symbol, side=side, quantity=qty, timestamp=NOW, reference_price=price)


class TestBuy:
    def test_buy_reduces_cash(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=10.0, price=100.0))
        assert broker.cash() == pytest.approx(9_000.0)

    def test_buy_adds_position(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=5.0, price=100.0))
        assert broker.positions().get("AAPL") == pytest.approx(5.0)

    def test_buy_insufficient_cash_raises(self):
        broker = PaperBroker(starting_cash=500.0)
        with pytest.raises(InsufficientCashError):
            broker.submit(make_order(qty=10.0, price=100.0))

    def test_multiple_buys_accumulate_position(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=3.0, price=100.0))
        broker.submit(make_order(qty=2.0, price=100.0))
        assert broker.positions().get("AAPL") == pytest.approx(5.0)

    def test_zero_qty_raises(self):
        broker = PaperBroker(starting_cash=10_000.0)
        with pytest.raises(ValueError):
            broker.submit(make_order(qty=0.0))


class TestSell:
    def test_sell_restores_cash(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=10.0, price=100.0, side=OrderSide.BUY))
        broker.submit(make_order(qty=10.0, price=110.0, side=OrderSide.SELL))
        assert broker.cash() == pytest.approx(10_100.0)

    def test_sell_removes_position(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=10.0, price=100.0, side=OrderSide.BUY))
        broker.submit(make_order(qty=10.0, price=100.0, side=OrderSide.SELL))
        assert "AAPL" not in broker.positions()

    def test_sell_without_position_raises(self):
        broker = PaperBroker(starting_cash=10_000.0)
        with pytest.raises(InsufficientPositionError):
            broker.submit(make_order(qty=5.0, price=100.0, side=OrderSide.SELL))

    def test_partial_sell_keeps_remainder(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=10.0, price=100.0, side=OrderSide.BUY))
        broker.submit(make_order(qty=4.0, price=100.0, side=OrderSide.SELL))
        assert broker.positions().get("AAPL") == pytest.approx(6.0)


class TestSlippage:
    def test_buy_with_slippage_costs_more(self):
        broker = PaperBroker(starting_cash=10_000.0, slippage_pct=0.01)
        broker.submit(make_order(qty=10.0, price=100.0))
        # precio efectivo = 100 * 1.01 = 101 → costo = 1010
        assert broker.cash() == pytest.approx(10_000.0 - 1_010.0)

    def test_sell_with_slippage_earns_less(self):
        broker = PaperBroker(starting_cash=10_000.0, slippage_pct=0.01)
        broker.submit(make_order(qty=10.0, price=100.0, side=OrderSide.BUY))
        cash_after_buy = broker.cash()
        broker.submit(make_order(qty=10.0, price=100.0, side=OrderSide.SELL))
        # precio efectivo venta = 100 * 0.99 = 99 → ingreso = 990
        assert broker.cash() == pytest.approx(cash_after_buy + 990.0)


class TestFills:
    def test_fills_recorded(self):
        broker = PaperBroker(starting_cash=10_000.0)
        broker.submit(make_order(qty=5.0, price=100.0))
        fills = list(broker.fills())
        assert len(fills) == 1
        assert fills[0].symbol == "AAPL"
        assert fills[0].quantity == pytest.approx(5.0)
