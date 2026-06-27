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
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

ALPACA_DATA_ENDPOINT = "https://data.alpaca.markets/v2"

OHLCV_COLS = ["date", "open", "high", "low", "close", "volume"]

# Caché local — carpeta relativa al directorio de trabajo
CACHE_DIR = Path("data/cache")

# Ventana de refresco: si el último dato del caché es más reciente que esto,
# se considera fresco y NO se descarga nada (ahorra llamadas a la API).
_FRESH_WINDOW: dict[str, timedelta] = {
    "1Min":  timedelta(minutes=2),
    "5Min":  timedelta(minutes=6),
    "15Min": timedelta(minutes=16),
    "1Hour": timedelta(hours=2),
    "1Day":  timedelta(hours=20),
}

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

    # ------------------------------------------------------------------
    # Caché local
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, tf_api: str) -> Path:
        return CACHE_DIR / f"{symbol}_{tf_api}.csv"

    def _load_cache(self, symbol: str, tf_api: str) -> pd.DataFrame | None:
        """Carga el caché local si existe. Retorna None si no hay caché."""
        path = self._cache_path(symbol, tf_api)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            log.warning("No se pudo leer caché %s: %s", path, exc)
            return None

    def _save_cache(self, symbol: str, tf_api: str, df: pd.DataFrame) -> None:
        """Guarda el DataFrame en caché CSV."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(symbol, tf_api)
        df.to_csv(path, index=False)

    def _is_fresh(self, df: pd.DataFrame, tf_api: str) -> bool:
        """True si el último dato del caché es suficientemente reciente."""
        if df.empty:
            return False
        last_date = df["date"].iloc[-1]
        if not hasattr(last_date, "tzinfo") or last_date.tzinfo is None:
            last_date = pd.Timestamp(last_date, tz="UTC")
        now = datetime.now(tz=timezone.utc)
        window = _FRESH_WINDOW.get(tf_api, timedelta(hours=2))
        return (now - last_date.to_pydatetime()) <= window

    # ------------------------------------------------------------------
    # Descarga con caché incremental
    # ------------------------------------------------------------------

    def fetch_historical_data(
        self,
        symbols: Iterable[str],
        timeframe: str = "1Hour",
        lookback_days: int | None = None,
        limit: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        """Descarga barras OHLCV para los símbolos dados, con caché incremental.

        - Si el caché está fresco → lo devuelve sin tocar Alpaca.
        - Si el caché existe pero está desactualizado → descarga solo las barras nuevas.
        - Si no hay caché → descarga el historial completo y lo guarda.

        Args:
            symbols:       Lista de tickers (ej. ["AAPL", "VOO"]).
            timeframe:     Resolución temporal.
            lookback_days: Días de historia para la descarga inicial.
            limit:         Máximo de barras por request (paginación automática).

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
        syms = [s.upper() for s in symbols]
        result: dict[str, pd.DataFrame] = {}

        for sym in syms:
            cached = self._load_cache(sym, tf_api)

            if cached is not None and self._is_fresh(cached, tf_api):
                log.info("[%s/%s] Caché fresco — sin descarga necesaria.", sym, tf_api)
                result[sym] = cached
                continue

            if cached is not None and not cached.empty:
                # Incremental: solo descarga desde el último dato
                last_ts = cached["date"].iloc[-1]
                if hasattr(last_ts, "to_pydatetime"):
                    last_ts = last_ts.to_pydatetime()
                start_dt = last_ts + timedelta(seconds=1)
                log.info("[%s/%s] Caché parcial — descargando desde %s.", sym, tf_api, start_dt.date())
            else:
                # Descarga completa
                start_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
                log.info("[%s/%s] Sin caché — descarga completa (%dd).", sym, tf_api, days)

            new_df = self._fetch_symbol(sym, tf_api, start_dt, limit)

            if cached is not None and not cached.empty and not new_df.empty:
                combined = pd.concat([cached, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined = combined.sort_values("date").reset_index(drop=True)
            elif not new_df.empty:
                combined = new_df
            else:
                combined = cached if cached is not None else new_df

            if not combined.empty:
                self._save_cache(sym, tf_api, combined)

            result[sym] = combined

        return result

    def _fetch_symbol(
        self,
        symbol: str,
        tf_api: str,
        start_dt: datetime,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Descarga barras para UN símbolo desde start_dt hasta ahora."""
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        all_bars: list = []
        next_token: str | None = None

        while True:
            params: dict = {
                "symbols": symbol,
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
            all_bars.extend(body.get("bars", {}).get(symbol, []))
            next_token = body.get("next_page_token")
            if not next_token:
                break

        return self._to_dataframe(symbol, all_bars)

    def fetch_multi_timeframe(
        self,
        symbols: Iterable[str],
        timeframes: list[str] | None = None,
        lookback_days: int | None = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Descarga múltiples resoluciones temporales con caché incremental por timeframe.

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
