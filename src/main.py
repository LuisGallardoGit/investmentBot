import sys
import os
# Asegurar que el directorio src esté en el path para las importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
from engine import DecisionEngine
from modules.risk_manager import RiskManager
from modules.sentiment import SentimentAnalyzer
import json
import os

class JarvisBot:
    def __init__(self):
        self.symbols = ["VOO", "QQQ", "AAPL"]
        self.initial_capital = 10000.0
        self.risk_manager = RiskManager(self.initial_capital)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.engine = DecisionEngine(self.symbols)

    def run_cycle(self):
        print("Iniciando ciclo de análisis...")
        # 1. Ingesta de datos
        data = {}
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data[symbol] = ticker.history(period="5d")

        # 2. Análisis de sentimiento
        sentiment = self.sentiment_analyzer.fetch_news_sentiment(self.symbols)

        # 3. Decisiones
        decisions = self.engine.evaluate(data, sentiment)

        # 4. Gestión de Riesgo y ejecución ficticia
        for symbol, action in decisions.items():
            if action != "HOLD":
                size = self.risk_manager.suggest_position_size(0.02)
                print(f"Sugerencia para {symbol}: {action} | Tamaño: {size}")

        return decisions

if __name__ == "__main__":
    bot = JarvisBot()
    bot.run_cycle()
