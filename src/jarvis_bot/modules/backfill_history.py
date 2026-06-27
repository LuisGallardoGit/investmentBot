import pandas as pd
from datetime import datetime
from pathlib import Path
import os

MASTER_RAW_REPORTS_PATH = Path("/Volumes/Jarvis 1.0/reports_content.csv")

history = [
    {
        "timestamp": "2026-05-18T08:30:00",
        "bot_id": "Fase 4",
        "content": "### Reporte de Inicio - Shadow Ingestor\n\n- **Estado:** Activo\n- **Acción:** Iniciando recolección de microestructura (Trades/Quotes) para AAPL, VOO, QQQ.\n- **Destino:** /Volumes/Jarvis 1.0/jarvis_hft_data_lake"
    },
    {
        "timestamp": "2026-05-18T09:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 09:00 AM\n\n- **Equity:** $9,919.96\n- **Estado:** 100% Liquidez\n- **Análisis:** Apertura con volatilidad controlada. EMAs 9/21 sin cruce. Sentiment: 0.69 (Neutral-Alcista). Macro: 0.50%."
    },
    {
        "timestamp": "2026-05-18T12:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 12:00 PM\n\n- **Equity:** $9,919.96\n- **Estado:** 100% Liquidez\n- **Análisis:** Mercado lateral en 'lunch hour'. El sentimiento se mantiene estable en 0.66. Sin señales de entrada."
    },
    {
        "timestamp": "2026-05-18T15:05:00",
        "bot_id": "Fase 4",
        "content": "### Reporte de Cierre - Shadow Ingestor\n\n- **Estado:** Detenido\n- **Datos Recolectados:** 1.1 GB (.jsonl)\n- **Análisis:** Recolección exitosa de 4,626 paquetes de datos del Order Book."
    },
    {
        "timestamp": "2026-05-19T08:30:00",
        "bot_id": "Fase 4",
        "content": "### Reporte de Inicio - Shadow Ingestor\n\n- **Estado:** Activo\n- **PID:** 80954\n- **Acción:** Sincronizado con la apertura de Wall Street para recolección HFT."
    },
    {
        "timestamp": "2026-05-19T09:00:00",
        "bot_id": "Fase 3",
        "content": "### Reporte Intradiario H1 - 09:00 AM\n\n- **Equity:** $9,919.96\n- **Macro FRED:** 0.54% (Mejora en la curva de rendimientos)\n- **Sentimiento:** 0.64\n- **Orden:** HOLD"
    }
]

def backfill():
    df_new = pd.DataFrame(history)
    if MASTER_RAW_REPORTS_PATH.exists():
        df_old = pd.read_csv(MASTER_RAW_REPORTS_PATH)
        # Combinar y eliminar duplicados basados en bot_id y timestamp aproximado
        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['timestamp', 'bot_id']).sort_values('timestamp', ascending=False)
    else:
        df_final = df_new.sort_values('timestamp', ascending=False)
        
    df_final.to_csv(MASTER_RAW_REPORTS_PATH, index=False)
    print(f"Historial reconstruido en {MASTER_RAW_REPORTS_PATH}")

if __name__ == "__main__":
    backfill()
