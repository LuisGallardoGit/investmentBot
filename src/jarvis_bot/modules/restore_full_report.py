import pandas as pd
from pathlib import Path

MASTER_RAW_REPORTS_PATH = Path("/Volumes/Jarvis 1.0/reports_content.csv")

full_content = """Aquí tienes el balance de la situación técnica y fundamental:
📊 Reporte de Ejecución Intradiaria (H1) - 10:00 AM

Estado de la Cartera:
- Equity Actual: $9,919.96
- Estado de Posiciones: 100% Liquidez (Cash).
- Vela Procesada: 9:00 AM - 10:00 AM Bogotá (14:00 - 15:00 UTC).

Órdenes de la Última Hora:
- No se emitieron órdenes.
- Razonamiento: El mercado ha entrado en una zona de estancamiento tras la primera hora de apertura. La EMA 9 sigue comprimiéndose contra la EMA 21, pero el cruce alcista no se ha materializado. El RSI oscila cerca de 50.3, confirmando una falta total de momentum en el corto plazo. El bot está filtrando correctamente este \"periodo muerto\".

🧠 Evaluación de Tendencia y Entorno (Gaia)

Sentimiento (Alpha Vantage):
- Score: 0.65 (Neutral-Alcista).
- Observación: Ligera mejora desde el 0.64 de la hora anterior. El flujo de noticias sobre Apple y los semiconductores sigue siendo moderadamente positivo, pero no lo suficiente como para arrastrar al precio fuera del rango lateral actual.

Contexto Macro (FRED):
- T10Y2Y: 0.54% (Estable). El mercado de renta fija no está enviando señales de estrés que afecten al Equity por ahora.

Análisis de la Fase 4:
- El Shadow Ingestor está capturando una actividad inusual en el Ask size de QQQ, lo que sugiere que hay muros de venta institucionales impidiendo el avance del Nasdaq en este momento. Esto valida por qué la Fase 3 se mantiene en HOLD.

Próximo paso: El siguiente escaneo se ejecutará a las 11:00 AM. Los archivos de control en /data ya tienen el estado actualizado para su uso en R."""

def restore():
    if not MASTER_RAW_REPORTS_PATH.exists():
        return
    
    df = pd.read_csv(MASTER_RAW_REPORTS_PATH)
    # Actualizar la entrada de las 10:00 AM de hoy
    mask = (df['bot_id'] == 'Fase 3') & (df['timestamp'].str.contains('2026-05-19T10'))
    if mask.any():
        df.loc[mask, 'content'] = full_content
        df.to_csv(MASTER_RAW_REPORTS_PATH, index=False)
        print("Reporte de las 10:00 AM actualizado con el texto completo.")
    else:
        print("No se encontró el reporte para actualizar.")

if __name__ == "__main__":
    restore()
