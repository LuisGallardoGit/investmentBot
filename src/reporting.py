import pandas as pd
import json
from datetime import datetime

class Reporter:
    """
    Genera reportes técnicos en lenguaje profesional con glosario.
    """
    def __init__(self, output_path="data/portfolio_state.csv"):
        self.output_path = output_path

    def generate_morning_brief(self, plan_data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"--- Morning Brief [{timestamp}] ---\n"
        report += "Plan de ejecución para hoy:\n"
        for item in plan_data:
            report += f"- {item}\n"
        return report

    def generate_evening_recap(self, pnl, trades, projections):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"--- Evening Recap [{timestamp}] ---\n"
        report += f"PnL Diario: {pnl}\n"
        report += f"Movimientos: {len(trades)}\n"
        report += f"Proyecciones: {projections}\n"
        return report

    def update_glossary(self, term, definition):
        glossary_path = "data/glossary.json"
        with open(glossary_path, 'r') as f:
            glossary = json.load(f)
        
        if not any(d['term'] == term for d in glossary):
            glossary.append({"term": term, "definition": definition})
            with open(glossary_path, 'w') as f:
                json.dump(glossary, f, indent=2)
