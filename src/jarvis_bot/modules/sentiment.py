import requests
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analiza el sentimiento del mercado mediante procesamiento de noticias financieras.
    Proporciona un multiplicador de confianza para las decisiones del Decision Engine.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.sentiment_score = 0.5  # Neutral por defecto (0.0 a 1.0)

    def fetch_news_sentiment(self, symbols):
        """
        Consulta Alpha Vantage (News Sentiment) o devuelve neutral si no hay API key.
        Convierte el score de Alpha Vantage (generalmente de -0.35 a 0.35) a un rango (0.0 a 1.0).
        """
        if not self.api_key:
            log.warning("No se proporcionó ALPHA_VANTAGE_API_KEY. Usando sentimiento estático neutral (0.5).")
            return 0.5
            
        tickers = ",".join(symbols)
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={self.api_key}"
        
        try:
            # Consultar solo el primer ticker (el mas liquido) si son varios para evitar que Alpha Vantage filtre en exceso al buscar la intersección
            query_ticker = symbols[0] if isinstance(symbols, (list, tuple)) and len(symbols) > 0 else "AAPL"
            log.info(f"Obteniendo sentimiento real desde Alpha Vantage para: {query_ticker} (proxy macro)...")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={query_ticker}&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "feed" in data and len(data["feed"]) > 0:
                # Calculamos el promedio de los primeros artículos devueltos (los más relevantes/recientes)
                scores = [float(item.get("overall_sentiment_score", 0)) for item in data["feed"][:10]]
                raw_score = sum(scores) / len(scores) if scores else 0.0
                
                # Alpha Vantage score: <= -0.35 es Bearish, >= 0.35 es Bullish
                # Normalizamos al rango 0.0 a 1.0
                normalized = (raw_score + 0.5)
                self.sentiment_score = max(0.0, min(1.0, normalized))
                log.info(f"Sentimiento agregado obtenido: {self.sentiment_score:.2f} (Raw Avg: {raw_score:.3f})")
            else:
                log.warning(f"Respuesta de API sin datos de 'feed' esperados. Keys: {list(data.keys())}. Usando fallback neutral.")
                self.sentiment_score = 0.5
                
        except Exception as e:
            log.error(f"Error fetching sentiment: {e}. Fallback to neutral.")
            self.sentiment_score = 0.5

        return self.sentiment_score

    def get_bias(self):
        """
        Retorna el sesgo actual: 'bullish', 'bearish' o 'neutral'.
        """
        if self.sentiment_score >= 0.65: return "bullish"
        if self.sentiment_score <= 0.35: return "bearish"
        return "neutral"
