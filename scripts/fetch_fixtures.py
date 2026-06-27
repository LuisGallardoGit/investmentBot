import yfinance as yf
import os

symbols = ["VOO", "QQQ", "AAPL"]
output_dir = "jarvis-trading-bot/data/fixtures"
os.makedirs(output_dir, exist_ok=True)

for symbol in symbols:
    print(f"Descargando datos para {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y")
    df.to_csv(os.path.join(output_dir, f"{symbol}.csv"))
    print(f"Guardado en {output_dir}/{symbol}.csv")
