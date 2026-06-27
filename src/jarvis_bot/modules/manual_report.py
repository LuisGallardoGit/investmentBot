import pandas as pd
from datetime import datetime
from pathlib import Path
import os

MASTER_RAW_REPORTS_PATH = Path("/Volumes/Jarvis 1.0/reports_content.csv")

def save_gaia_report(bot_id: str, content: str):
    """Guarda un reporte de texto arbitrario (como el de los Crons) en el CSV maestro."""
    timestamp = datetime.now().isoformat()
    
    df_content = pd.DataFrame([{
        "timestamp": timestamp,
        "bot_id": bot_id,
        "content": content
    }])
    
    header = not MASTER_RAW_REPORTS_PATH.exists()
    df_content.to_csv(MASTER_RAW_REPORTS_PATH, mode='a', index=False, header=header)
    print(f"Reporte de {bot_id} persistido exitosamente.")
    
    # Notificar al Dashboard vía API local
    try:
        import requests
        requests.post("http://localhost:8000/api/notify_report", timeout=2)
    except:
        pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        save_gaia_report(sys.argv[1], sys.argv[2])
