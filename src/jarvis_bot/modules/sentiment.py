"""Análisis de sentimiento de noticias financieras.

Fuente primaria: Finnhub (gratis, sin tarjeta, 60 req/min).
  Endpoint: GET https://finnhub.io/api/v1/news-sentiment?symbol=X&token=KEY
  Score: buzz.weeklyAverage + sentiment.bullishPercent → normalizado a [0, 1]

Fallback: Alpha Vantage (si ALPHA_VANTAGE_API_KEY está configurada).
Fallback final: 0.5 neutral.

Requiere env: FINNHUB_API_KEY (o ALPHA_VANTAGE_API_KEY como alternativa)
"""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1/news-sentiment"
AV_BASE = "https://www.alphavantage.co/query"
NEUTRAL = 0.5


class SentimentAnalyzer:
    """Analiza el sentimiento del mercado via Finnhub o Alpha Vantage.

    Retorna un score [0.0, 1.0]:
      < 0.35 = bajista
      0.35-0.65 = neutral
      > 0.65 = alcista
    """

    def __init__(self, api_key: str | None = None) -> None:
        # api_key es Alpha Vantage (compatibilidad hacia atrás)
        self.av_key = api_key
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.sentiment_score: float = NEUTRAL

    def fetch_news_sentiment(self, symbols: list[str]) -> float:
        """Obtiene sentimiento para los símbolos dados.

        Intenta Finnhub primero, luego Alpha Vantage, luego neutral.
        """
        if self.finnhub_key:
            score = self._fetch_finnhub(symbols)
            if score is not None:
                self.sentiment_score = score
                return score

        if self.av_key:
            score = self._fetch_alpha_vantage(symbols)
            if score is not None:
                self.sentiment_score = score
                return score

        log.warning(
            "Sin FINNHUB_API_KEY ni ALPHA_VANTAGE_API_KEY — sentimiento neutral (0.5). "
            "Registrarse en https://finnhub.io (gratis) para activar el módulo."
        )
        return NEUTRAL

    def _fetch_finnhub(self, symbols: list[str]) -> float | None:
        """Consulta Finnhub por cada símbolo y promedia los scores."""
        scores: list[float] = []
        for sym in symbols:
            try:
                resp = requests.get(
                    FINNHUB_BASE,
                    params={"symbol": sym, "token": self.finnhub_key},
                    timeout=8,
                )
                resp.raise_for_status()
                data = resp.json()

                # Finnhub devuelve: buzz.weeklyAverage + sentiment.bullishPercent
                bullish_pct = data.get("sentiment", {}).get("bullishPercent")
                bearish_pct = data.get("sentiment", {}).get("bearishPercent")
                if bullish_pct is not None and bearish_pct is not None:
                    # bullishPercent ya está en [0, 1]
                    scores.append(float(bullish_pct))
                    log.debug(
                        "Finnhub %s: bullish=%.2f, bearish=%.2f",
                        sym, bullish_pct, bearish_pct,
                    )
            except Exception as exc:
                log.debug("Finnhub error para %s: %s", sym, exc)

        if not scores:
            return None

        avg = sum(scores) / len(scores)
        log.info(
            "Sentimiento Finnhub (%d símbolo(s)): %.2f [%s]",
            len(scores), avg, self._bias_label(avg),
        )
        return round(avg, 4)

    def _fetch_alpha_vantage(self, symbols: list[str]) -> float | None:
        """Fallback a Alpha Vantage NEWS_SENTIMENT."""
        query_ticker = symbols[0] if symbols else "AAPL"
        try:
            log.info("Sentimiento Alpha Vantage para: %s (proxy macro)...", query_ticker)
            resp = requests.get(
                AV_BASE,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": query_ticker,
                    "apikey": self.av_key,
                },
                timeout=10,
            )
            data = resp.json()
            feed = data.get("feed", [])
            if not feed:
                log.warning("Alpha Vantage: sin datos de feed. Keys: %s", list(data.keys()))
                return None
            raw_scores = [float(item.get("overall_sentiment_score", 0)) for item in feed[:10]]
            raw_avg = sum(raw_scores) / len(raw_scores)
            # Alpha Vantage range ~[-0.5, 0.5] → normalizar a [0, 1]
            normalized = max(0.0, min(1.0, raw_avg + 0.5))
            log.info("Sentimiento Alpha Vantage: %.2f (raw %.3f)", normalized, raw_avg)
            return normalized
        except Exception as exc:
            log.error("Error Alpha Vantage sentiment: %s", exc)
            return None

    def get_bias(self) -> str:
        if self.sentiment_score >= 0.65:
            return "bullish"
        if self.sentiment_score <= 0.35:
            return "bearish"
        return "neutral"

    @staticmethod
    def _bias_label(score: float) -> str:
        if score >= 0.65:
            return "bullish"
        if score <= 0.35:
            return "bearish"
        return "neutral"
