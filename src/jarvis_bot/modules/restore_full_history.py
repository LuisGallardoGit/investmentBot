import pandas as pd
from pathlib import Path

MASTER_RAW_REPORTS_PATH = Path("/Volumes/Jarvis 1.0/reports_content.csv")

history_updates = {
    "2026-05-19T09": """📊 Reporte de Ejecución Intradiaria (H1) - 9:00 AM

Estado de la Cartera:
- Equity Actual: $9,919.96 (Paper Trading).
- Estado de Posiciones: 100% Liquidez (Cash).

Órdenes de la Última Hora:
- No se emitieron órdenes nuevas.
- Diagnóstico Técnico: Tras la apertura, el mercado muestra una leve presión lateral. La EMA de 9 períodos continúa por debajo de la EMA de 21.

🧠 Evaluación de Tendencia (Gaia)
- Sentimiento: 0.64 (Neutral-Alcista).
- Macro FRED: 0.54% (Fortaleza económica).
- Veredicto: El bot opera bajo una neutralidad constructiva.""",

    "2026-05-19T11": """📊 Reporte de Ejecución Intradiaria (H1) - 11:00 AM

Estado de la Cartera:
- Equity Actual: $9,919.96
- Estado de Posiciones: 100% Liquidez (Cash).

Órdenes de la Última Hora:
- No se emitieron órdenes.
- Análisis técnico: El precio sigue consolidando. EMAs convergiendo sin Golden Cross. El sistema evita el whipsaw.

🧠 Evaluación de Tendencia (Gaia)
- Sentimiento: 0.59 (Enfriamiento).
- Macro: 0.50%.
- Veredicto: Probabilidad de éxito baja para entradas en este momento."""
}

def restore():
    if not MASTER_RAW_REPORTS_PATH.exists(): return
    df = pd.read_csv(MASTER_RAW_REPORTS_PATH)
    
    for ts_match, content in history_updates.items():
        mask = (df['bot_id'] == 'Fase 3') & (df['timestamp'].str.contains(ts_match))
        if mask.any():
            df.loc[mask, 'content'] = content
            
    df.to_csv(MASTER_RAW_REPORTS_PATH, index=False)
    print("Historial de reportes enriquecido exitosamente.")

if __name__ == "__main__":
    restore()
