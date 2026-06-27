"""Ingesta de datos de mercado desde Alpaca Data API v2.

Feed: SIP (Securities Information Processor) — feed consolidado con 100%
del volumen de mercado US. Preferible a IEX (~33% del volumen).

Si la cuenta Alpaca es free-tier, SIP no está disponible; en ese caso
configurar ALPACA_DATA_FEED=iex en el .env.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

ALPACA_DATA_ENDPOINT = "https://data.alpaca.markets/v2"

OHLCV_COLS = ["date", "open", "high", "low", "close", "volume"]

# Mapeo de timeframe legible → formato Alpaca API
TIMEFRAME_MAP = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "1h": "1Hour",
    "1H": "1Hour",
    "1Hour": "1Hour",
    "1d": "1Day",
    "1D": "1Day",
    "1Day": "1Day",
}

# Lookback por defecto según timeframe (en días)
DEFAULT_LOOKBACK = {
    "1Min": 5,
    "5Min": 10,
    "15Min": 20,
    "1Hour": 365,
    "1Day": 730,
}


class AlpacaDataIngestor:
    """Descarga barras OHLCV desde Alpaca Data API v2.

    Parámetros de entorno:
        ALPACA_API_KEY       — clave pública Alpaca
        ALPACA_SECRET_KEY    — clave secreta Alpaca
        ALPACA_DATA_FEED     — "sip" (default) o "iex" para cuentas free-tier
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        # SIP por defecto; permite override via env para cuentas free-tier
        self.feed = os.getenv("ALPACA_DATA_FEED", "sip").lower()
        self.endpoint = ALPACA_DATA_ENDPOINT
        self._headers = {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
        }

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def fetch_historical_data(
        self,
        symbols: Iterable[str],
        timeframe: str = "1Hour",
        lookback_days: int | None = None,
        limit: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        """Descarga barras OHLCV para los símbolos dados.

        Args:
            symbols:      Lista de tickers (ej. ["AAPL", "VOO"]).
            timeframe:    Resolución temporal. Acepta "1h","1H","1Hour","1d","1D","1Day", etc.
            lookback_days: Días de historia a pedir. Si None, usa el default por timeframe.
            limit:        Máximo de barras por símbolo por request (paginación automática).

        Returns:
            Dict {SYMBOL: DataFrame con columnas date,open,high,low,close,volume}.
        """
        if not self.is_configured():
            raise ValueError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY no configurados. "
                "Copia .env.example a .env y rellena las claves."
            )

        tf_api = TIMEFRAME_MAP.get(timeframe, timeframe)
        days = lookback_days or DEFAULT_LOOKBACK.get(tf_api, 365)

        start_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        syms = [s.upper() for s in symbols]
        syms_str = ",".join(syms)

        log.info(
            "Descargando %s bars desde Alpaca (%s feed, lookback=%dd): %s",
            tf_api,
            self.feed,
            days,
            syms_str,
        )

        all_bars: dict[str, list] = {s: [] for s in syms}
        next_token: str | None = None

        # Paginación automática — Alpaca devuelve next_page_token cuando hay más datos
        while True:
            params: dict = {
                "symbols": syms_str,
                "timeframe": tf_api,
                "start": start_str,
                "limit": limit,
                "adjustment": "all",
                "feed": self.feed,
            }
            if next_token:
                params["page_token"] = next_token

            resp = requests.get(
                f"{self.endpoint}/stocks/bars",
                headers=self._headers,
                params=params,
                timeout=20,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Alpaca Data API error {resp.status_code}: {resp.text[:300]}"
                )

            body = resp.json()
            bars_page = body.get("bars", {})
            for sym in syms:
                all_bars[sym].extend(bars_page.get(sym, []))

            next_token = body.get("next_page_token")
            if not next_token:
                break

        return {sym: self._to_dataframe(sym, all_bars[sym]) for sym in syms}

    def fetch_multi_timeframe(
        self,
        symbols: Iterable[str],
        timeframes: list[str] | None = None,
        lookback_days: int | None = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Descarga múltiples resoluciones temporales de una vez.

        Returns:
            Dict anidado {timeframe: {SYMBOL: DataFrame}}.
            Ej: result["1Hour"]["AAPL"] → DataFrame horario de AAPL.
        """
        if timeframes is None:
            timeframes = ["1Hour", "1Day"]

        syms = list(symbols)
        result: dict[str, dict[str, pd.DataFrame]] = {}
        for tf in timeframes:
            log.info("Descargando timeframe: %s", tf)
            result[tf] = self.fetch_historical_data(syms, timeframe=tf, lookback_days=lookback_days)
        return result

    # ------------------------------------------------------------------

    def _to_dataframe(self, symbol: str, bars: list[dict]) -> pd.DataFrame:
        """Convierte lista de barras Alpaca a DataFrame normalizado."""
        if not bars:
            log.warning("Sin datos para %s — DataFrame vacío.", symbol)
            return pd.DataFrame(columns=OHLCV_COLS)

        df = pd.DataFrame(bars)
        df = df.rename(
            columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        # Conservar solo columnas estándar
        existing = [c for c in OHLCV_COLS if c in df.columns]
        df = df[existing]

        df = validate_ohlcv(df, symbol)
        return df


# ---------------------------------------------------------------------------
# Validación de calidad de datos
# ---------------------------------------------------------------------------


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Valida y limpia un DataFrame OHLCV.

    Checks:
    - NaN en columnas críticas → elimina filas y loggea warning.
    - Volumen cero → loggea warning (no elimina, algunos ETFs tienen V=0 en festivos).
    - High < Low → loggea error y elimina fila (dato corrupto).
    - Close <= 0 → elimina fila.
    - Filas duplicadas por fecha → conserva la última.

    Returns el DataFrame limpio.
    """
    tag = f"[{symbol}]" if symbol else ""
    original_len = len(df)

    if df.empty:
        return df

    # Duplicados por fecha
    dupes = df.duplicated(subset=["date"], keep="last")
    if dupes.any():
        log.warning("%s %d filas duplicadas por fecha eliminadas.", tag, dupes.sum())
        df = df[~dupes].reset_index(drop=True)

    # NaN en precio de cierre
    nan_close = df["close"].isna()
    if nan_close.any():
        log.warning("%s %d filas con close=NaN eliminadas.", tag, nan_close.sum())
        df = df[~nan_close].reset_index(drop=True)

    # Close <= 0
    bad_price = df["close"] <= 0
    if bad_price.any():
        log.warning("%s %d filas con close<=0 eliminadas.", tag, bad_price.sum())
        df = df[~bad_price].reset_index(drop=True)

    # High < Low (dato corrupto)
    if "high" in df.columns and "low" in df.columns:
        inverted = df["high"] < df["low"]
        if inverted.any():
            log.error("%s %d filas con high<low eliminadas (datos corruptos).", tag, inverted.sum())
            df = df[~inverted].reset_index(drop=True)

    # Volumen cero (warning informativo)
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            log.warning("%s %d barras con volumen=0 (festivos/halt).", tag, zero_vol)

    cleaned = len(df)
    if cleaned < original_len:
        log.info("%s Validación: %d → %d filas (%d eliminadas).", tag, original_len, cleaned, original_len - cleaned)

    return df
