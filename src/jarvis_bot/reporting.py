"""Persistencia y reporte del pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import stats_engine
from .config import OutputConfig
from .engine import PipelineResult

log = logging.getLogger(__name__)

MASTER_VOLUME = Path("/Volumes/Jarvis 1.0")
MASTER_REPORTS_PATH = MASTER_VOLUME / "reports_history.csv"
MASTER_GLOSSARY_PATH = MASTER_VOLUME / "glossary.csv"
MASTER_RAW_REPORTS_PATH = MASTER_VOLUME / "reports_content.csv"


def _volume_available() -> bool:
    """Devuelve True solo si el volumen externo está montado y accesible."""
    return MASTER_VOLUME.exists()


def append_to_master_reports(result: PipelineResult, bot_id: str = "Fase 3") -> None:
    """Guarda resumen y contenido extenso para visualización en el Front.

    Si el volumen externo no está montado, registra un warning y continúa.
    """
    if not _volume_available():
        log.warning("Volumen '%s' no disponible — omitiendo sync al master.", MASTER_VOLUME)
        return

    equity_curve = [s.equity for s in result.snapshots]
    stats = stats_engine.summarize(equity_curve)
    timestamp = datetime.now().isoformat()

    # 1. Resumen técnico (CSV ligero)
    report_row = {
        "timestamp": timestamp,
        "bot_id": bot_id,
        "symbols": ";".join(result.symbols),
        "snapshots": len(result.snapshots),
        "trades_count": len(result.trades),
        "equity_final": equity_curve[-1] if equity_curve else 0.0,
        "total_return": stats["total_return"],
        "sharpe_ratio": stats["sharpe_ratio"],
        "max_drawdown": stats["max_drawdown"],
    }
    df_meta = pd.DataFrame([report_row])
    write_header = not MASTER_REPORTS_PATH.exists()
    df_meta.to_csv(MASTER_REPORTS_PATH, mode="a", index=False, header=write_header)

    # 2. Contenido extenso (Markdown raw)
    content_path = Path("logs/last_run_tmp.md")
    export_report(result, content_path)
    content = content_path.read_text(encoding="utf-8")

    df_content = pd.DataFrame([{"timestamp": timestamp, "bot_id": bot_id, "content": content}])
    write_header_content = not MASTER_RAW_REPORTS_PATH.exists()
    df_content.to_csv(MASTER_RAW_REPORTS_PATH, mode="a", index=False, header=write_header_content)
    log.info("Reporte contenido anexado a %s", MASTER_RAW_REPORTS_PATH)


def sync_glossary_to_csv(json_path: Path) -> None:
    """Convierte el glosario JSON a CSV en el volumen externo.

    No-op si el volumen no está montado o el archivo no existe.
    """
    if not _volume_available():
        log.warning("Volumen '%s' no disponible — omitiendo sync de glosario.", MASTER_VOLUME)
        return
    if not json_path.exists():
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.to_csv(MASTER_GLOSSARY_PATH, index=False)
    log.info("Glosario sincronizado en %s", MASTER_GLOSSARY_PATH)


def persist_portfolio_state(result: PipelineResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for snap in result.snapshots:
        rows.append(
            {
                "timestamp": snap.timestamp,
                "cash": snap.cash,
                "positions_value": snap.positions_value,
                "equity": snap.equity,
                "positions": ";".join(f"{k}={v}" for k, v in sorted(snap.positions.items())),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info("portfolio_state -> %s (%d filas)", path, len(df))
    return path


def persist_history(result: PipelineResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result.trades)
    df.to_csv(path, index=False)
    log.info("history -> %s (%d trades)", path, len(df))
    return path


def export_report(result: PipelineResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    equity_curve = [s.equity for s in result.snapshots]
    stats = stats_engine.summarize(equity_curve)
    start_eq = equity_curve[0] if equity_curve else 0.0
    end_eq = equity_curve[-1] if equity_curve else 0.0

    lines = [
        "# Jarvis Trading Bot - Reporte de Simulacion",
        "",
        f"- Modo: paper",
        f"- Simbolos: {', '.join(result.symbols)}",
        f"- Snapshots: {len(result.snapshots)}",
        f"- Trades ejecutados: {len(result.trades)}",
        f"- Equity inicial: {start_eq:.2f}",
        f"- Equity final: {end_eq:.2f}",
        "",
        "## Metricas",
        f"- Total return: {stats['total_return']:.4f}",
        f"- Sharpe ratio: {stats['sharpe_ratio']:.4f}",
        f"- Max drawdown: {stats['max_drawdown']:.4f}",
        f"- CAGR: {stats['cagr']:.4f}",
        "",
        "## Trades",
    ]
    if result.trades:
        df = pd.DataFrame(result.trades)
        lines.append(df.to_markdown(index=False))
    else:
        lines.append("(sin trades)")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("report -> %s", path)
    return path


def persist_all(result: PipelineResult, output: OutputConfig) -> dict[str, Path]:
    # Sync para el Dashboard
    append_to_master_reports(result)
    sync_glossary_to_csv(Path("data/glossary.json"))
    
    return {
        "portfolio_state": persist_portfolio_state(result, output.state_path),
        "history": persist_history(result, output.history_path),
        "report": export_report(result, output.report_path),
    }
