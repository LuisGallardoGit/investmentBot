"""Tests para el módulo de horarios de mercado NYSE (contexto Colombia)."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from jarvis_bot.modules.market_hours import (
    TZ_BOGOTA,
    TZ_NEW_YORK,
    assert_market_open,
    is_after_hours,
    is_holiday,
    is_market_open,
    is_premarket,
    is_weekend,
    market_status,
    minutes_to_close,
    next_market_open,
)

# Helpers para crear datetimes en ET


def et(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ_NEW_YORK)


def bogota(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ_BOGOTA)


# -----------------------------------------------------------------------
# Festivos y fines de semana
# -----------------------------------------------------------------------


class TestHolidaysAndWeekends:
    def test_christmas_2025_is_holiday(self):
        assert is_holiday(date(2025, 12, 25))

    def test_mlk_2026_is_holiday(self):
        assert is_holiday(date(2026, 1, 19))

    def test_regular_tuesday_is_not_holiday(self):
        assert not is_holiday(date(2026, 3, 10))

    def test_saturday_is_weekend(self):
        assert is_weekend(date(2026, 6, 27))  # sábado

    def test_sunday_is_weekend(self):
        assert is_weekend(date(2026, 6, 28))  # domingo

    def test_monday_is_not_weekend(self):
        assert not is_weekend(date(2026, 6, 29))  # lunes


# -----------------------------------------------------------------------
# is_market_open
# -----------------------------------------------------------------------


class TestIsMarketOpen:
    def test_open_at_930(self):
        assert is_market_open(et(2026, 6, 29, 9, 30))

    def test_open_at_1500(self):
        assert is_market_open(et(2026, 6, 29, 15, 0))

    def test_closed_at_1600(self):
        # 16:00 exactas ya es cierre
        assert not is_market_open(et(2026, 6, 29, 16, 0))

    def test_closed_before_open(self):
        assert not is_market_open(et(2026, 6, 29, 9, 29))

    def test_closed_on_saturday(self):
        assert not is_market_open(et(2026, 6, 27, 10, 0))

    def test_closed_on_sunday(self):
        assert not is_market_open(et(2026, 6, 28, 12, 0))

    def test_closed_on_holiday(self):
        # 4 Jul 2025 = Independence Day
        assert not is_market_open(et(2025, 7, 4, 10, 0))

    def test_closed_late_night(self):
        assert not is_market_open(et(2026, 6, 29, 22, 0))


# -----------------------------------------------------------------------
# is_premarket / is_after_hours
# -----------------------------------------------------------------------


class TestPremarketAndAfterHours:
    def test_premarket_at_700(self):
        assert is_premarket(et(2026, 6, 29, 7, 0))

    def test_not_premarket_at_930(self):
        assert not is_premarket(et(2026, 6, 29, 9, 30))

    def test_after_hours_at_1700(self):
        assert is_after_hours(et(2026, 6, 29, 17, 0))

    def test_not_after_hours_during_market(self):
        assert not is_after_hours(et(2026, 6, 29, 13, 0))

    def test_not_after_hours_on_weekend(self):
        assert not is_after_hours(et(2026, 6, 27, 17, 0))


# -----------------------------------------------------------------------
# minutes_to_close
# -----------------------------------------------------------------------


class TestMinutesToClose:
    def test_returns_none_when_closed(self):
        assert minutes_to_close(et(2026, 6, 29, 8, 0)) is None

    def test_returns_positive_during_market(self):
        m = minutes_to_close(et(2026, 6, 29, 15, 0))
        assert m is not None
        assert m == pytest.approx(60.0, abs=1.0)

    def test_returns_near_zero_just_before_close(self):
        m = minutes_to_close(et(2026, 6, 29, 15, 59))
        assert m is not None
        assert 0 < m <= 1.1


# -----------------------------------------------------------------------
# next_market_open
# -----------------------------------------------------------------------


class TestNextMarketOpen:
    def test_before_open_same_day(self):
        at = et(2026, 6, 29, 8, 0)  # lunes 8am ET
        nxt = next_market_open(at)
        assert nxt.date() == date(2026, 6, 29)
        assert nxt.hour == 9
        assert nxt.minute == 30

    def test_during_market_returns_next_day(self):
        at = et(2026, 6, 29, 11, 0)  # lunes 11am
        nxt = next_market_open(at)
        assert nxt.date() == date(2026, 6, 30)  # martes

    def test_friday_after_close_returns_monday(self):
        at = et(2026, 6, 26, 17, 0)  # viernes 5pm
        nxt = next_market_open(at)
        assert nxt.date() == date(2026, 6, 29)  # lunes

    def test_skips_holiday(self):
        # Día anterior a Independence Day 2026 (3 jul = viernes observado)
        at = et(2026, 7, 2, 17, 0)  # jueves 5pm
        nxt = next_market_open(at)
        # 3 Jul es festivo observado, siguiente día hábil = 6 Jul (lunes)
        assert nxt.date() == date(2026, 7, 6)

    def test_result_in_et_timezone(self):
        at = et(2026, 6, 29, 8, 0)
        nxt = next_market_open(at)
        assert nxt.tzinfo is not None


# -----------------------------------------------------------------------
# market_status
# -----------------------------------------------------------------------


class TestMarketStatus:
    def test_open_session_shows_regular(self):
        status = market_status(et(2026, 6, 29, 10, 0))
        assert status["is_open"] is True
        assert status["session"] == "regular"
        assert status["minutes_to_close"] is not None

    def test_closed_session_shows_closed(self):
        status = market_status(et(2026, 6, 29, 20, 30))
        assert status["is_open"] is False
        assert status["session"] == "closed"

    def test_status_includes_bogota_time(self):
        status = market_status(et(2026, 6, 29, 10, 0))
        assert "bogota_time" in status
        assert "COT" in status["bogota_time"] or "BOT" in status["bogota_time"] or "-05" in status["bogota_time"]

    def test_status_includes_next_open(self):
        status = market_status(et(2026, 6, 29, 20, 0))
        assert "next_open_et" in status
        assert "next_open_bogota" in status


# -----------------------------------------------------------------------
# assert_market_open
# -----------------------------------------------------------------------


class TestAssertMarketOpen:
    def test_raises_when_closed(self):
        with pytest.raises(RuntimeError, match="Mercado cerrado"):
            assert_market_open(et(2026, 6, 27, 10, 0))  # sábado

    def test_no_raise_when_open(self):
        assert_market_open(et(2026, 6, 29, 10, 0))  # lunes en horario


# -----------------------------------------------------------------------
# Colombia timezone (horario de Bogotá)
# -----------------------------------------------------------------------


class TestColombiaTimezone:
    def test_bogota_is_utc_minus5(self):
        """Colombia no tiene DST — siempre UTC-5."""
        dt_bog = bogota(2026, 6, 29, 8, 30)   # 8:30 AM Bogotá
        dt_et = dt_bog.astimezone(TZ_NEW_YORK)
        # En verano NY = UTC-4, así que 8:30 Bogotá (UTC-5) = 9:30 ET (UTC-4)
        assert dt_et.hour == 9
        assert dt_et.minute == 30

    def test_market_open_time_in_bogota(self):
        """9:30 ET en horario de verano = 8:30 Bogotá."""
        open_et = et(2026, 6, 29, 9, 30)  # junio = EDT (UTC-4)
        open_bog = open_et.astimezone(TZ_BOGOTA)
        assert open_bog.hour == 8
        assert open_bog.minute == 30

    def test_market_close_time_in_bogota_summer(self):
        """16:00 ET (EDT) = 15:00 Bogotá."""
        close_et = et(2026, 6, 29, 16, 0)
        close_bog = close_et.astimezone(TZ_BOGOTA)
        assert close_bog.hour == 15
        assert close_bog.minute == 0

    def test_market_open_time_bogota_winter(self):
        """9:30 ET en horario estándar (EST = UTC-5) = 9:30 Bogotá también."""
        open_et = et(2026, 1, 5, 9, 30)  # enero = EST (UTC-5)
        open_bog = open_et.astimezone(TZ_BOGOTA)
        assert open_bog.hour == 9
        assert open_bog.minute == 30
