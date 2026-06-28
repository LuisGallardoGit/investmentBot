# Jarvis Trading Bot

Bot de paper trading para mercados internacionales (NYSE) desde Colombia. Opera en modo periódico con estrategia ML, filtros macroeconómicos y gestión de riesgo FX COP/USD.

**Estado actual:** Paper trading activo — 1 año de datos, +14.57% retorno, win rate 61.6%, max drawdown 3.46%.

---

## Arquitectura

```
Alpaca Data API (1H bars)
        ↓
   enrich() — indicadores técnicos (EMA, RSI, MACD, BB, ATR)
        ↓
   MLStrategy — RandomForest, walk-forward training, 17 features
        ↓
   Filtros de entrada:
     • FXRiskManager        — volatilidad COP/USD (open.er-api.com)
     • MarketRegimeAnalyzer — VIX + WTI desde FRED
     • RiskManager          — Kelly-lite, max drawdown circuit breaker
        ↓
   PaperBroker → AlpacaExecutor (bracket orders)
        ↓
   Reporting: history.csv / portfolio_state.csv / logs/report.md
```

**4 ciclos diarios (lunes–viernes, hora Bogotá UTC-5):**

| Hora Bogotá | UTC   | Motivo                 |
|-------------|-------|------------------------|
| 8:35 AM     | 13:35 | Pre-apertura NYSE       |
| 9:00 AM     | 14:00 | Apertura (gap capture)  |
| 12:00 PM    | 17:00 | Mediodía               |
| 3:30 PM     | 20:30 | Pre-cierre NYSE         |

---

## Módulos principales

| Módulo | Descripción |
|--------|-------------|
| `engine.py` | Pipeline principal — orquesta datos, señales, riesgo y ejecución |
| `modules/macro.py` | Yield curve T10Y2Y (FRED) + USD/COP (open.er-api.com) |
| `modules/market_regime.py` | VIX + WTI desde FRED → multiplica tamaño de posición |
| `modules/fx_risk.py` | Riesgo COP/USD — caution 5%, block 10% de variación 30d |
| `modules/sentiment.py` | Sentimiento Finnhub (por símbolo) con fallback Alpha Vantage |
| `modules/risk_manager.py` | Kelly-lite: qty = min(qty_por_riesgo, qty_por_capital) |
| `modules/alpaca_executor.py` | Bracket orders stop-loss + take-profit en Alpaca |
| `modules/position_reconciler.py` | Sincroniza estado local con posiciones reales Alpaca |
| `modules/market_hours.py` | Guard NYSE — no opera fuera de horario |
| `modules/alerts.py` | Alertas de drawdown, FX, trades, resumen diario |
| `monitoring.py` | Health checks de APIs al inicio del pipeline |
| `ml/features.py` | 17 features: RSI, MACD, BB, retornos, lags |
| `ml/model.py` | SignalClassifier (RandomForest) con walk-forward training |
| `strategies/` | TrendFollow, MeanReversion, Combined, ML |

---

## Estrategia ML

- **Modelo:** RandomForest con walk-forward validation (3 splits, 70% train)
- **Target:** Retorno forward a 5 barras (1H), threshold ±1% → BUY/SELL/HOLD
- **Features top:** return_5d (36.3%), return_1d (26.2%), return_10d (14.3%)
- **Confianza mínima:** 0.45 para ejecutar señal
- **Símbolos:** AAPL, QQQ, VOO

### Lógica de sizing

```
qty = floor(sizing.quantity × fx_multiplier × regime_multiplier)

fx_multiplier (COP/USD variación 30d):
  < 5%   → 1.00  normal
  5-10%  → 0.50  cautela
  > 10%  → 0.00  bloqueado

regime_multiplier (VIX + WTI):
  VIX < 20, WTI normal      → 1.00
  VIX 20-25 o WTI caída 10% → 0.75
  VIX 25-35 o WTI caída 20% → 0.50
  VIX > 35                   → 0.00  bloqueado
```

---

## Rendimiento paper trading (1 año)

| Métrica | Valor | KPI objetivo |
|---------|-------|--------------|
| Retorno total | +14.57% | — |
| Win rate | 61.6% | ≥ 50% |
| Profit factor | 1.89 | ≥ 1.2 |
| Max drawdown | 3.46% | < 10% |
| Sharpe ratio | 0.95 | ≥ 0.5 |
| Hold promedio | 59h (~2.5 días) | — |

---

## Instalación

**Requisitos:** Python 3.11+, macOS (launchd para automatización)

```bash
git clone https://github.com/LuisGallardoGit/investmentBot.git
cd jarvis-trading-bot-feat-mvp-paper-trading-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Variables de entorno (`.env`)

```env
# Alpaca Paper Trading
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
ALPACA_DATA_FEED=sip

# FRED API — yield curve T10Y2Y, VIX, WTI
FRED_API_KEY=...

# Finnhub — sentimiento de noticias por símbolo (gratis: finnhub.io)
FINNHUB_API_KEY=...
```

### Ejecución manual

```bash
source .venv/bin/activate
python -m jarvis_bot.main --log-file logs/jarvis.log
```

### Automatización con launchd (macOS)

```bash
# 4 ciclos diarios del bot
cp ../com.luisj.investmentbot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.luisj.investmentbot.plist

# Monitor semanal USD/COP (lunes 8:00 AM Bogotá)
cp ../com.luisj.usdcop-monitor.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.luisj.usdcop-monitor.plist
```

---

## Archivos de salida

| Archivo | Descripción |
|---------|-------------|
| `data/history.csv` | Historial de trades con PnL USD y COP por operación |
| `data/portfolio_state.csv` | Snapshots de equity en cada barra procesada |
| `data/usdcop_history.csv` | Historial semanal USD/COP |
| `data/cache/{SYM}_{tf}.csv` | Cache incremental de datos OHLCV |
| `logs/jarvis.log` | Log principal del pipeline |
| `logs/cron.log` | Log de ejecuciones automáticas via launchd |
| `logs/report.md` | Reporte markdown del último ciclo |
| `models/signal_classifier.pkl` | Modelo ML entrenado |

---

## Reportes automáticos (Cowork)

- **8:25 AM Bogotá (L-V):** Briefing matutino — equity, posiciones, USD/COP, ciclos del día
- **4:00 PM Bogotá (L-V):** Resumen fin de día — trades ejecutados, PnL, errores del cron

---

## Plan de inversión

Ver [`PLAN_INVERSION.md`](../PLAN_INVERSION.md):

- **Fase 1:** Paper trading 8 semanas — KPIs cumplidos ✅
- **Fase 2:** Capital real $500–$2,000 USD
- **Fase 3:** Escalar 2× por mes si KPIs se mantienen

---

## Seguridad

- `trading_mode` bloqueado en `paper` por guardrail en `load_config()` — no puede activarse trading real por config
- Cero secretos hardcodeados — todo via variables de entorno
- `.env` en `.gitignore`
