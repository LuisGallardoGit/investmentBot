import requests
import pandas as pd
import logging
import os
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

class AlpacaDataIngestor:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        # Para datos, el endpoint siempre es data.alpaca.markets
        self.endpoint = "https://data.alpaca.markets/v2"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }

    def fetch_historical_data(self, symbols, timeframe="1Hour", limit=250):
        if not self.api_key:
            raise ValueError("ALPACA_API_KEY no configurado para ingesta de datos.")
            
        data_dict = {}
        # Alpaca requiere símbolos en mayúscula y separados por coma
        syms_str = ",".join([s.upper() for s in symbols])
        
        # Calcular un start date para traer la data suficiente (ej. 30 dias atrás para H1)
        start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
        url = f"{self.endpoint}/stocks/bars?symbols={syms_str}&timeframe={timeframe}&start={start_date}&limit={limit}&adjustment=all&feed=iex"
        
        try:
            log.info(f"Descargando market data de Alpaca ({timeframe}): {syms_str}")
            res = requests.get(url, headers=self.headers, timeout=15)
            
            if res.status_code != 200:
                raise RuntimeError(f"Error Alpaca Data API: {res.status_code} - {res.text}")
                
            json_data = res.json()
            bars = json_data.get("bars", {})
            
            for symbol in symbols:
                sym_upper = symbol.upper()
                if sym_upper not in bars or not bars[sym_upper]:
                    log.warning(f"No hay datos para el símbolo {sym_upper}")
                    # Retornamos DataFrame vacío con columnas esperadas para evitar crashes
                    data_dict[sym_upper] = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
                    continue
                    
                df = pd.DataFrame(bars[sym_upper])
                # Mapear columnas de Alpaca (t, o, h, l, c, v) al formato del pipeline
                df = df.rename(columns={
                    "t": "date",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume"
                })
                # Asegurar formato datetime y UTC
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df = df.sort_values("date").reset_index(drop=True)
                
                # Descartar columnas extra de Alpaca (n, vw)
                cols = ["date", "open", "high", "low", "close", "volume"]
                data_dict[sym_upper] = df[cols]
                
        except Exception as e:
            log.error(f"Error fetching data from Alpaca: {e}")
            raise
            
        return data_dict
