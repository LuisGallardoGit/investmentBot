import pandas as pd
from datetime import datetime
from pathlib import Path
import os

MASTER_RAW_REPORTS_PATH = Path("/Volumes/Jarvis 1.0/reports_content.csv")

history = [
    {
        "timestamp": "2026-05-18T10:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 10:00 AM\n\n- **Equity:** $9,919.96\n- **Estado:** 100% Liquidez\n- **Análisis:** Mercado en estancamiento. EMA 9 plana contra EMA 21. RSI: 50.3. Filtro de sentimiento autoriza compra pero técnico en HOLD."
    },
    {
        "timestamp": "2026-05-18T11:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 11:00 AM\n\n- **Equity:** $9,919.96\n- **Estado:** 100% Liquidez\n- **Análisis:** Enfriamiento del sentimiento (0.59). Ausencia de cruce Golden Cross. El sistema evita el whipsaw en zona de indecisión."
    },
    {
        "timestamp": "2026-05-18T13:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 01:00 PM\n\n- **Equity:** $9,919.96\n- **Estado:** 100% Liquidez\n- **Análisis:** Sentimiento estable en 0.67. El Shadow Ingestor reporta crecimiento exponencial en Quotes. Próximo escaneo a las 2:00 PM."
    },
    {
        "timestamp": "2026-05-18T14:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 02:00 PM\n\n- **Equity:** $9,919.96\n- **Estado:** 100% Liquidez\n- **Análisis:** Consolidación lateral estrecha. EMAs en paralelo. RSI en 52. Preparando cierre de mercado y recolección de Fase 4."
    },
    {
        "timestamp": "2026-05-18T15:04:00",
        "bot_id": "Fase 3",
        "content": "### Reporte de Cierre Fase 3 - 03:00 PM\n\n- **Equity Final:** $9,919.96\n- **PnL Diario:** $0.00\n- **Análisis Final:** Día de preservación de capital. El sistema filtró el ruido lateral exitosamente."
    },
    {
        "timestamp": "2026-05-19T10:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 10:00 AM\n\n- **Equity:** $9,919.96\n- **Macro FRED:** 0.54% (Estable)\n- **Sentimiento:** 0.65\n- **Observación:** El Shadow Ingestor detecta muros de venta en QQQ. Bot en HOLD."
    }
]

def backfill():
    df_new = pd.DataFrame(history)
    if MASTER_RAW_REPORTS_PATH.exists():
        df_old = pd.read_csv(MASTER_RAW_REPORTS_PATH)
        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['timestamp', 'bot_id']).sort_values('timestamp', ascending=False)
    else:
        df_final = df_new.sort_values('timestamp', ascending=False)
        
    df_final.to_csv(MASTER_RAW_REPORTS_PATH, index=False)
    print(f"Historial completo restaurado en {MASTER_RAW_REPORTS_PATH}")

if __name__ == "__main__":
    backfill()
