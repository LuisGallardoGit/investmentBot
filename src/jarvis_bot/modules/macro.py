import requests
import logging

log = logging.getLogger(__name__)

class MacroAnalyzer:
    """
    Analiza el entorno macroeconómico usando la API de FRED (Reserva Federal de St. Louis).
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.macro_risk = "normal"  # normal, high_risk

    def fetch_macro_state(self):
        if not self.api_key:
            log.warning("No FRED_API_KEY. Contexto macro neutral.")
            return self.macro_risk
        
        try:
            # Consultamos la curva de rendimiento (10-Year vs 2-Year Treasury)
            # Una curva invertida (valor < 0) suele predecir recesiones y mercados bajistas.
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key={self.api_key}&file_type=json&sort_order=desc&limit=1"
            
            log.info("Obteniendo contexto macroeconómico desde FRED (T10Y2Y Yield Curve)...")
            res = requests.get(url, timeout=10)
            data = res.json()
            
            if "observations" in data and len(data["observations"]) > 0:
                # Filtrar valores '.' que manda FRED en feriados
                val_str = data["observations"][0]["value"]
                if val_str != '.':
                    val = float(val_str)
                    log.info(f"FRED T10Y2Y (Yield Curve Spread): {val:.2f}%")
                    
                    if val < 0:
                        self.macro_risk = "high_risk"
                        log.warning("Curva de tipos invertida detectada. Riesgo Macro ALTO.")
                    else:
                        self.macro_risk = "normal"
                else:
                    log.info("FRED devolvió '.' (Feriado). Se mantiene riesgo anterior.")
            else:
                log.warning("Respuesta vacía de FRED.")
                
        except Exception as e:
            log.error(f"Error fetching FRED macro data: {e}")

        return self.macro_risk
