"""Tests para AlpacaDataIngestor y validate_ohlcv."""

import pandas as pd
import pytest

from jarvis_bot.modules.alpaca_data import validate_ohlcv, OHLCV_COLS


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestValidateOhlcv:
    def _base_row(self, date="2024-01-01", close=100.0, high=105.0, low=95.0, volume=1000):
        return {"date": pd.Timestamp(date, tz="UTC"), "open": close, "high": high, "low": low, "close": close, "volume": volume}

    def test_clean_df_unchanged(self):
        df = make_df([self._base_row("2024-01-01"), self._base_row("2024-01-02")])
        result = validate_ohlcv(df, "TEST")
        assert len(result) == 2

    def test_removes_nan_close(self):
        rows = [self._base_row("2024-01-01"), {**self._base_row("2024-01-02"), "close": float("nan")}]
        df = make_df(rows)
        result = validate_ohlcv(df, "TEST")
        assert len(result) == 1

    def test_removes_zero_close(self):
        rows = [self._base_row("2024-01-01"), {**self._base_row("2024-01-02"), "close": 0.0}]
        df = make_df(rows)
        result = validate_ohlcv(df, "TEST")
        assert len(result) == 1

    def test_removes_inverted_high_low(self):
        rows = [self._base_row("2024-01-01"), {**self._base_row("2024-01-02"), "high": 90.0, "low": 110.0}]
        df = make_df(rows)
        result = validate_ohlcv(df, "TEST")
        assert len(result) == 1

    def test_removes_duplicate_dates(self):
        rows = [self._base_row("2024-01-01"), self._base_row("2024-01-01")]
        df = make_df(rows)
        result = validate_ohlcv(df, "TEST")
        assert len(result) == 1

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=OHLCV_COLS)
        result = validate_ohlcv(df, "TEST")
        assert result.empty

    def test_zero_volume_kept_with_warning(self):
        # Volumen cero no se elimina (festivos/halts)
        rows = [self._base_row("2024-01-01", volume=0)]
        df = make_df(rows)
        result = validate_ohlcv(df, "TEST")
        assert len(result) == 1


class TestTimeframeMap:
    def test_timeframe_aliases_resolve(self):
        from jarvis_bot.modules.alpaca_data import TIMEFRAME_MAP
        assert TIMEFRAME_MAP["1h"] == "1Hour"
        assert TIMEFRAME_MAP["1H"] == "1Hour"
        assert TIMEFRAME_MAP["1d"] == "1Day"
        assert TIMEFRAME_MAP["1D"] == "1Day"


class TestAlpacaDataIngestorConfig:
    def test_not_configured_without_env(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        from jarvis_bot.modules.alpaca_data import AlpacaDataIngestor
        ingestor = AlpacaDataIngestor()
        assert not ingestor.is_configured()

    def test_default_feed_is_sip(self, monkeypatch):
        monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
        from jarvis_bot.modules.alpaca_data import AlpacaDataIngestor
        ingestor = AlpacaDataIngestor()
        assert ingestor.feed == "sip"

    def test_feed_override_via_env(self, monkeypatch):
        monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
        from jarvis_bot.modules.alpaca_data import AlpacaDataIngestor
        ingestor = AlpacaDataIngestor()
        assert ingestor.feed == "iex"
