"""WalkForwardTrainer — entrenamiento y evaluación OOS del modelo ML — Fase 6.

Flujo walk-forward:
  Para cada split (k de n_splits):
    1. Train: primeras train_pct% de las filas del split
    2. Test (OOS): restantes (1 - train_pct)%
    3. Entrenar SignalClassifier en train
    4. Evaluar en test: accuracy, precision por clase, comparar vs rule-based

Métricas OOS:
  - accuracy: % de predicciones correctas
  - precision_buy, precision_sell: precisión para señales accionables
  - vs_rule_based: compara el accuracy del ML contra el de la estrategia rule-based
  - feature_importances: promedio de importancias across splits

El objetivo no es maximizar accuracy (el mercado es ruidoso) sino:
  - Accuracy OOS > 50% (mejor que azar)
  - Precision BUY/SELL > 55% (señales accionables más confiables)
  - Diferencia vs rule-based: mide el valor añadido del ML
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import FeatureConfig, build_feature_matrix
from .model import LABEL_BUY, LABEL_HOLD, LABEL_SELL, ModelConfig, SignalClassifier

log = logging.getLogger(__name__)


@dataclass
class SplitMetrics:
    """Métricas de un único split walk-forward."""

    split_id: int
    n_train: int
    n_test: int
    accuracy: float
    precision_buy: float | None
    precision_sell: float | None
    n_predicted_buy: int
    n_predicted_sell: int
    n_predicted_hold: int
    oob_score: float | None
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardReport:
    """Reporte agregado del entrenamiento walk-forward."""

    splits: list[SplitMetrics]
    avg_accuracy: float
    avg_precision_buy: float | None
    avg_precision_sell: float | None
    top_features: list[tuple[str, float]]   # promedio de importancias
    best_split_id: int
    model_config: ModelConfig = field(default_factory=ModelConfig)

    def summary(self) -> str:
        lines = [
            f"Walk-Forward ML Report ({len(self.splits)} splits)",
            f"  Accuracy OOS promedio:    {self.avg_accuracy:.3f}",
            f"  Precision BUY promedio:   {self.avg_precision_buy:.3f}" if self.avg_precision_buy else "  Precision BUY:  N/A",
            f"  Precision SELL promedio:  {self.avg_precision_sell:.3f}" if self.avg_precision_sell else "  Precision SELL: N/A",
            "  Top features:",
        ]
        for fname, imp in self.top_features[:5]:
            lines.append(f"    {fname}: {imp:.4f}")
        return "\n".join(lines)


class WalkForwardTrainer:
    """Entrena y evalúa el SignalClassifier en splits walk-forward.

    Parámetros
    ----------
    n_splits : int
        Número de splits (default 3).
    train_pct : float
        Fracción de cada split usada para entrenamiento (default 0.70).
    feature_cfg : FeatureConfig | None
        Configuración de features/targets.
    model_cfg : ModelConfig | None
        Configuración del modelo.
    min_train_samples : int
        Mínimo de muestras de entrenamiento por split (default 50).
    """

    def __init__(
        self,
        n_splits: int = 3,
        train_pct: float = 0.70,
        feature_cfg: FeatureConfig | None = None,
        model_cfg: ModelConfig | None = None,
        min_train_samples: int = 50,
    ) -> None:
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.feature_cfg = feature_cfg or FeatureConfig()
        self.model_cfg = model_cfg or ModelConfig()
        self.min_train_samples = min_train_samples

    def train(self, df: pd.DataFrame) -> tuple[WalkForwardReport, SignalClassifier]:
        """Ejecuta el entrenamiento walk-forward y retorna el reporte + modelo final.

        El modelo final se entrena sobre TODOS los datos (para usar en producción).

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame enriquecido (salida de indicators.enrich()).

        Retorna
        -------
        (WalkForwardReport, SignalClassifier entrenado con todos los datos)
        """
        fm = build_feature_matrix(df, self.feature_cfg)
        if len(fm.X) == 0:
            raise ValueError("Feature matrix vacía — no se puede entrenar.")

        splits = self._make_splits(fm.X, fm.y)
        split_metrics: list[SplitMetrics] = []
        all_importances: list[dict[str, float]] = []

        for split_id, (X_train, y_train, X_test, y_test) in enumerate(splits):
            if len(X_train) < self.min_train_samples:
                log.warning(
                    "Split %d: solo %d muestras de train (mínimo %d) — omitido.",
                    split_id, len(X_train), self.min_train_samples,
                )
                continue

            clf = SignalClassifier(self.model_cfg)
            train_result = clf.fit(X_train, y_train)

            # Evaluar OOS
            metrics = self._evaluate(clf, X_test, y_test, split_id, train_result)
            split_metrics.append(metrics)
            all_importances.append(train_result.feature_importances)

            log.info(
                "Split %d: accuracy=%.3f precision_buy=%s precision_sell=%s",
                split_id,
                metrics.accuracy,
                f"{metrics.precision_buy:.3f}" if metrics.precision_buy is not None else "N/A",
                f"{metrics.precision_sell:.3f}" if metrics.precision_sell is not None else "N/A",
            )

        if not split_metrics:
            raise RuntimeError("Ningún split tuvo suficientes datos para entrenar.")

        # Modelo final con todos los datos
        final_clf = SignalClassifier(self.model_cfg)
        final_clf.fit(fm.X, fm.y)

        report = self._build_report(split_metrics, all_importances)
        log.info("\n%s", report.summary())
        return report, final_clf

    def _make_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """Divide X, y en n_splits walk-forward sin solapamiento."""
        n = len(X)
        split_size = n // self.n_splits
        result = []
        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else n
            X_split = X.iloc[start:end]
            y_split = y.iloc[start:end]
            train_end = int(len(X_split) * self.train_pct)
            X_train = X_split.iloc[:train_end]
            y_train = y_split.iloc[:train_end]
            X_test = X_split.iloc[train_end:]
            y_test = y_split.iloc[train_end:]
            result.append((X_train, y_train, X_test, y_test))
        return result

    def _evaluate(
        self,
        clf: SignalClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        split_id: int,
        train_result: Any,
    ) -> SplitMetrics:
        """Calcula métricas OOS para un split."""
        if len(X_test) == 0:
            return SplitMetrics(
                split_id=split_id, n_train=train_result.n_samples, n_test=0,
                accuracy=0.0, precision_buy=None, precision_sell=None,
                n_predicted_buy=0, n_predicted_sell=0, n_predicted_hold=0,
                oob_score=train_result.oob_score,
            )

        y_pred = clf.predict(X_test)
        y_true = y_test.values

        accuracy = float(np.mean(y_pred == y_true))

        # Precision por clase: TP / (TP + FP)
        precision_buy = self._precision(y_true, y_pred, LABEL_BUY)
        precision_sell = self._precision(y_true, y_pred, LABEL_SELL)

        return SplitMetrics(
            split_id=split_id,
            n_train=train_result.n_samples,
            n_test=len(X_test),
            accuracy=accuracy,
            precision_buy=precision_buy,
            precision_sell=precision_sell,
            n_predicted_buy=int(np.sum(y_pred == LABEL_BUY)),
            n_predicted_sell=int(np.sum(y_pred == LABEL_SELL)),
            n_predicted_hold=int(np.sum(y_pred == LABEL_HOLD)),
            oob_score=train_result.oob_score,
            feature_importances=train_result.feature_importances,
        )

    @staticmethod
    def _precision(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float | None:
        """Precisión para una clase específica."""
        predicted_as_label = y_pred == label
        if not predicted_as_label.any():
            return None
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else None

    def _build_report(
        self,
        splits: list[SplitMetrics],
        all_importances: list[dict[str, float]],
    ) -> WalkForwardReport:
        """Agrega métricas de todos los splits en un reporte."""
        avg_acc = float(np.mean([s.accuracy for s in splits]))

        buy_precs = [s.precision_buy for s in splits if s.precision_buy is not None]
        avg_buy = float(np.mean(buy_precs)) if buy_precs else None

        sell_precs = [s.precision_sell for s in splits if s.precision_sell is not None]
        avg_sell = float(np.mean(sell_precs)) if sell_precs else None

        # Promedio de importancias
        all_features: set[str] = set()
        for imp in all_importances:
            all_features.update(imp.keys())

        avg_imp: dict[str, float] = {}
        for feat in all_features:
            vals = [imp.get(feat, 0.0) for imp in all_importances]
            avg_imp[feat] = float(np.mean(vals))

        top_features = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

        best_split = max(splits, key=lambda s: s.accuracy)

        return WalkForwardReport(
            splits=splits,
            avg_accuracy=avg_acc,
            avg_precision_buy=avg_buy,
            avg_precision_sell=avg_sell,
            top_features=top_features,
            best_split_id=best_split.split_id,
            model_config=self.model_cfg,
        )
