import requests
import logging
import os

log = logging.getLogger(__name__)

class AlpacaExecutor:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.endpoint = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
    def is_configured(self):
        return bool(self.api_key and self.secret_key)
        
    def submit_order(self, symbol, side, qty):
        """Envía una orden al mercado simulado de Alpaca."""
        if not self.is_configured():
            log.warning("Alpaca no está configurado. Omitiendo ejecución live.")
            return False
            
        url = f"{self.endpoint}/v2/orders"
        
        payload = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.lower(),  # 'buy' o 'sell'
            "type": "market",
            "time_in_force": "gtc"
        }
        
        try:
            log.info(f"Enviando orden a Alpaca: {side.upper()} {qty} {symbol.upper()}")
            res = requests.post(url, headers=self.headers, json=payload, timeout=10)
            
            if res.status_code in [200, 201]:
                data = res.json()
                log.info(f"Orden aceptada por Alpaca. ID: {data.get('id')}, Status: {data.get('status')}")
                return True
            else:
                log.error(f"Alpaca rechazó la orden. Status: {res.status_code}, Msg: {res.text}")
                return False
                
        except Exception as e:
            log.error(f"Error comunicando con Alpaca: {e}")
            return False
