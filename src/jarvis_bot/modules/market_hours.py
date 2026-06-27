"""Gestión de horarios de mercado NYSE/NASDAQ desde Colombia.

Contexto Colombia:
  - Zona horaria: America/Bogota (UTC-5, SIN cambio de horario — Colombia no usa DST)
  - NYSE/NASDAQ abren 9:30 AM ET y cierran 4:00 PM ET
  - ET en horario estándar (EST) = UTC-5 = igual a Bogotá → 9:30–16:00 hora Colombia
  - ET en horario de verano (EDT) = UTC-4 → 8:30–15:00 hora Colombia
  - Pre-market US: 4:00 AM – 9:30 AM ET
  - After-hours US: 4:00 PM – 8:00 PM ET

  El bot opera SOLO en horario regular (9:30–16:00 ET) para evitar
  baja liquidez en pre/after-market.

Días festivos:
  El NYSE tiene ~9 días festivos al año. En la Fase 3 se implementa
  la lista fija del año vigente; en Fase 4 se puede consultar la API
  de Alpaca para festivos dinámicos.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

# Zonas horarias
TZ_BOGOTA = ZoneInfo("America/Bogota")
TZ_NEW_YORK = ZoneInfo("America/New_York")

# Horario regular NYSE/NASDAQ (hora Nueva York)
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)

# Pre-market y after-hours (informativos, no se opera)
PREMARKET_OPEN_ET = time(4, 0)
AFTER_HOURS_CLOSE_ET = time(20, 0)

# Festivos NYSE 2025 y 2026 (fechas confirmadas)
NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 1, 1),   # Año Nuevo
    date(2025, 1, 20),  # Martin Luther King Jr.
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Navidad
    # 2026
    date(2026, 1, 1),   # Año Nuevo
    date(2026, 1, 19),  # Martin Luther King Jr.
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observado)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Navidad
}


def now_et() -> datetime:
    """Hora actual en zona horaria Nueva York."""
    return datetime.now(tz=TZ_NEW_YORK)


def now_bogota() -> datetime:
    """Hora actual en zona horaria Bogotá."""
    return datetime.now(tz=TZ_BOGOTA)


def is_holiday(d: date | None = None) -> bool:
    """True si la fecha es festivo NYSE."""
    check = d or datetime.now(tz=TZ_NEW_YORK).date()
    return check in NYSE_HOLIDAYS


def is_weekend(d: date | None = None) -> bool:
    """True si la fecha cae en fin de semana (sábado=5, domingo=6)."""
    check = d or datetime.now(tz=TZ_NEW_YORK).date()
    return check.weekday() >= 5


def is_market_open(at: datetime | None = None) -> bool:
    """True si el NYSE está abierto en el momento dado (o ahora).

    Condiciones:
      1. Día de semana (lun-vie)
      2. No es festivo NYSE
      3. Hora entre 9:30 y 16:00 ET
    """
    now = (at or datetime.now(tz=TZ_NEW_YORK)).astimezone(TZ_NEW_YORK)
    d = now.date()
    if is_weekend(d) or is_holiday(d):
        return False
    t = now.time().replace(second=0, microsecond=0)
    return MARKET_OPEN_ET <= t < MARKET_CLOSE_ET


def is_premarket(at: datetime | None = None) -> bool:
    """True si estamos en pre-market (4:00–9:30 AM ET)."""
    now = (at or datetime.now(tz=TZ_NEW_YORK)).astimezone(TZ_NEW_YORK)
    d = now.date()
    if is_weekend(d) or is_holiday(d):
        return False
    t = now.time()
    return PREMARKET_OPEN_ET <= t < MARKET_OPEN_ET


def is_after_hours(at: datetime | None = None) -> bool:
    """True si estamos en after-hours (16:00–20:00 ET)."""
    now = (at or datetime.now(tz=TZ_NEW_YORK)).astimezone(TZ_NEW_YORK)
    d = now.date()
    if is_weekend(d) or is_holiday(d):
        return False
    t = now.time()
    return MARKET_CLOSE_ET <= t < AFTER_HOURS_CLOSE_ET


def minutes_to_close(at: datetime | None = None) -> float | None:
    """Minutos hasta el cierre del mercado. None si el mercado no está abierto."""
    if not is_market_open(at):
        return None
    now = (at or datetime.now(tz=TZ_NEW_YORK)).astimezone(TZ_NEW_YORK)
    close_today = now.replace(
        hour=MARKET_CLOSE_ET.hour, minute=MARKET_CLOSE_ET.minute, second=0, microsecond=0
    )
    delta = close_today - now
    return delta.total_seconds() / 60


def next_market_open(at: datetime | None = None) -> datetime:
    """Retorna el próximo momento en que abre el NYSE (en hora ET).

    Si el mercado está actualmente abierto, retorna el próximo día de mercado.
    Si estamos antes de las 9:30 en un día hábil, retorna las 9:30 de hoy.
    """
    now = (at or datetime.now(tz=TZ_NEW_YORK)).astimezone(TZ_NEW_YORK)
    candidate = now.date()

    # Si estamos en un día hábil antes de la apertura → hoy a las 9:30
    if (
        not is_weekend(candidate)
        and not is_holiday(candidate)
        and now.time() < MARKET_OPEN_ET
    ):
        return now.replace(
            hour=MARKET_OPEN_ET.hour, minute=MARKET_OPEN_ET.minute, second=0, microsecond=0
        )

    # Avanzamos al siguiente día hasta encontrar un día hábil
    candidate += timedelta(days=1)
    while is_weekend(candidate) or is_holiday(candidate):
        candidate += timedelta(days=1)

    return datetime(
        year=candidate.year,
        month=candidate.month,
        day=candidate.day,
        hour=MARKET_OPEN_ET.hour,
        minute=MARKET_OPEN_ET.minute,
        tzinfo=TZ_NEW_YORK,
    )


def market_status(at: datetime | None = None) -> dict:
    """Resumen del estado del mercado para logging/UI.

    Returns
    -------
    dict con: is_open, session, et_time, bogota_time, next_open, minutes_to_close
    """
    now_ny = (at or datetime.now(tz=TZ_NEW_YORK)).astimezone(TZ_NEW_YORK)
    now_bog = now_ny.astimezone(TZ_BOGOTA)

    if is_market_open(now_ny):
        session = "regular"
    elif is_premarket(now_ny):
        session = "pre-market"
    elif is_after_hours(now_ny):
        session = "after-hours"
    else:
        session = "closed"

    next_open = next_market_open(now_ny)
    minutes_close = minutes_to_close(now_ny)

    return {
        "is_open": session == "regular",
        "session": session,
        "et_time": now_ny.strftime("%Y-%m-%d %H:%M %Z"),
        "bogota_time": now_bog.strftime("%Y-%m-%d %H:%M %Z"),
        "is_holiday": is_holiday(now_ny.date()),
        "is_weekend": is_weekend(now_ny.date()),
        "next_open_et": next_open.strftime("%Y-%m-%d %H:%M %Z"),
        "next_open_bogota": next_open.astimezone(TZ_BOGOTA).strftime("%Y-%m-%d %H:%M %Z"),
        "minutes_to_close": minutes_close,
    }


def assert_market_open(at: datetime | None = None) -> None:
    """Lanza RuntimeError si el mercado no está abierto.

    Usar como guard en engine.py antes de enviar órdenes reales.
    """
    if not is_market_open(at):
        status = market_status(at)
        next_open = status["next_open_et"]
        raise RuntimeError(
            f"Mercado cerrado (sesión: '{status['session']}'). "
            f"Próxima apertura: {next_open} / {status['next_open_bogota']} (Bogotá)."
        )
