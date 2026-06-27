"""Entrypoint del pipeline de paper trading."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from .config import load_config
from .engine import run_pipeline
from .logging_setup import setup_logging
from .reporting import persist_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="jarvis-bot", description="Jarvis Trading Bot - paper MVP")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta a un YAML de config (default: config/default.yaml).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/jarvis.log"),
        help="Archivo de log adicional. Usa '-' para deshabilitar.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    log_file = None if str(args.log_file) == "-" else args.log_file
    logger = setup_logging(cfg.log_level, log_file=log_file)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("Iniciando pipeline paper trading (modo=%s, seed=%d)", cfg.trading_mode, cfg.seed)
    result = run_pipeline(cfg)
    outputs = persist_all(result, cfg.output)
    logger.info("Salidas: %s", {k: str(v) for k, v in outputs.items()})

    logger.info(
        "Resumen: equity_final=%.2f trades=%d snapshots=%d",
        result.final_equity,
        len(result.trades),
        len(result.snapshots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
