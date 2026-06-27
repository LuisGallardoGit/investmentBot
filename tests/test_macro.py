"""Tests para MacroAnalyzer — sin llamadas reales a FRED."""

from unittest.mock import MagicMock, patch

import pytest

from jarvis_bot.modules.macro import MacroAnalyzer, MacroState


class TestMacroAnalyzerNoKey:
    def test_no_api_key_returns_normal(self):
        analyzer = MacroAnalyzer(api_key=None)
        result = analyzer.fetch_macro_state()
        assert result == "normal"

    def test_no_api_key_full_state_has_nones(self):
        analyzer = MacroAnalyzer(api_key=None)
        state = analyzer.fetch_full_state()
        assert isinstance(state, MacroState)
        assert state.yield_curve is None
        assert state.usd_cop is None


class TestMacroAnalyzerWithMockedFred:
    def _mock_fred_response(self, value: str):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"observations": [{"value": value}]}
        return mock_resp

    def test_inverted_curve_returns_high_risk(self):
        analyzer = MacroAnalyzer(api_key="fake_key")
        with patch("jarvis_bot.modules.macro.requests.get") as mock_get:
            # Primera llamada: T10Y2Y negativo → high_risk
            # Segunda llamada: USD/COP
            mock_get.side_effect = [
                self._mock_fred_response("-0.50"),   # T10Y2Y invertida
                self._mock_fred_response("4200.0"),  # USD/COP actual
            ]
            result = analyzer.fetch_macro_state()
        assert result == "high_risk"

    def test_positive_curve_returns_normal(self):
        analyzer = MacroAnalyzer(api_key="fake_key")
        with patch("jarvis_bot.modules.macro.requests.get") as mock_get:
            mock_get.side_effect = [
                self._mock_fred_response("1.20"),    # T10Y2Y positiva
                self._mock_fred_response("4100.0"),  # USD/COP
            ]
            result = analyzer.fetch_macro_state()
        assert result == "normal"

    def test_holiday_dot_value_keeps_previous(self):
        analyzer = MacroAnalyzer(api_key="fake_key")
        with patch("jarvis_bot.modules.macro.requests.get") as mock_get:
            # Primera observación es '.', la segunda tiene valor numérico
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "observations": [{"value": "."}, {"value": "0.80"}]
            }
            mock_get.return_value = mock_resp
            state = analyzer.fetch_full_state()
        # Debe haber encontrado 0.80 saltando el '.'
        assert state.yield_curve == pytest.approx(0.80)

    def test_usd_cop_populated_in_full_state(self):
        analyzer = MacroAnalyzer(api_key="fake_key")
        with patch("jarvis_bot.modules.macro.requests.get") as mock_get:
            mock_get.side_effect = [
                self._mock_fred_response("0.50"),    # T10Y2Y
                self._mock_fred_response("4350.0"),  # USD/COP
            ]
            state = analyzer.fetch_full_state()
        assert state.usd_cop == pytest.approx(4350.0)

    def test_network_error_returns_normal(self):
        analyzer = MacroAnalyzer(api_key="fake_key")
        with patch("jarvis_bot.modules.macro.requests.get", side_effect=ConnectionError("timeout")):
            result = analyzer.fetch_macro_state()
        assert result == "normal"
