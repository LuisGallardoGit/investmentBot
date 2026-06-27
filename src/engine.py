import pandas as pd
import numpy as np

class DecisionEngine:
    """
    El 'cerebro' del bot. Evalúa indicadores técnicos y sentimiento
    para decidir acciones de inversión.
    """
    def __init__(self, symbols):
        self.symbols = symbols

    def evaluate(self, market_data, sentiment_score):
        decisions = {}
        for symbol in self.symbols:
            df = market_data[symbol]
            if df.empty:
                decisions[symbol] = "HOLD"
                continue

            # Cálculo de indicadores básicos (SMA)
            sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]

            # Lógica de decisión simplificada para el entrenamiento
            if current_price < sma_20 and sentiment_score > 0.5:
                decisions[symbol] = "BUY"
            elif current_price > sma_20 and sentiment_score < 0.5:
                decisions[symbol] = "SELL"
            else:
                decisions[symbol] = "HOLD"
        
        return decisions
