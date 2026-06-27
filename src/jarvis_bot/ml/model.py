"""SignalClassifier — wrapper de RandomForest para señales de trading — Fase 6.

Diseño:
  - Modelo base: RandomForestClassifier de sklearn (robusto, no requiere scaling)
  - Clases: -1 (SELL), 0 (HOLD), +1 (BUY)
  - Umbral de confianza: solo emite BUY/SELL si la probabilidad supera el umbral
    (evita señales ruidosas cuando el modelo está inseguro)
  - Persistencia: guarda/carga con joblib (formato binario compacto)
  - Feature importance: exporta ranking de features como dict

Ciclo de vida:
  1. Construir feature matrix con ml.features.build_feature_matrix()
  2. Entrenar: SignalClassifier.fit(X_train, y_train)
  3. Predecir: SignalClassifier.predict_with_confidence(X_row)
     → retorna (signal_int, confidence_float)
  4. Persistir: .save(path) / SignalClassifier.load(path)

Dependencia requerida:
  scikit-learn >= 1.3  (en [project.optional-dependencies] ml)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Clases del clasificador
LABEL_BUY = 1
LABEL_HOLD = 0
LABEL_SELL = -1
LABELS = [LABEL_SELL, LABEL_HOLD, LABEL_BUY]
LABEL_NAMES = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}


@dataclass
class ModelConfig:
    """Hiperparámetros del SignalClassifier."""

    n_estimators: int = 200          # árboles en el Random Forest
    max_depth: int | None = 6        # profundidad máxima (None = sin límite)
    min_samples_leaf: int = 20       # muestras mínimas por hoja (evita overfit)
    class_weight: str = "balanced"   # compensa desbalance BUY/SELL/HOLD
    random_state: int = 42
    confidence_threshold: float = 0.45  # probabilidad mínima para BUY/SELL
    n_jobs: int = -1                 # usar todos los cores disponibles


@dataclass
class TrainResult:
    """Resultado de un entrenamiento."""

    n_samples: int
    n_features: int
    feature_names: list[str]
    class_distribution: dict[str, int]
    oob_score: float | None = None   # Out-Of-Bag score (si oob_score=True)
    feature_importances: dict[str, float] = field(default_factory=dict)


class SignalClassifier:
    """Clasificador de señales de trading basado en Random Forest.

    Parámetros
    ----------
    cfg : ModelConfig | None
        Configuración del modelo. Se usan valores por defecto si es None.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self._model: Any = None
        self._feature_names: list[str] = []
        self._is_trained = False

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TrainResult:
        """Entrena el clasificador con los datos provistos.

        Parámetros
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples × n_features).
        y : pd.Series
            Targets: -1, 0, +1. Debe tener el mismo índice que X.

        Retorna
        -------
        TrainResult con métricas de entrenamiento.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise ImportError(
                "scikit-learn es requerido para ML. "
                "Instala con: pip install scikit-learn"
            ) from exc

        if len(X) == 0:
            raise ValueError("No hay datos de entrenamiento (X está vacío).")
        if len(X) != len(y):
            raise ValueError(f"X y y tienen diferente longitud: {len(X)} vs {len(y)}")

        self._feature_names = list(X.columns)
        X_arr = X.values
        y_arr = y.values

        log.info(
            "Entrenando SignalClassifier: %d muestras × %d features",
            len(X_arr), len(self._feature_names),
        )

        self._model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            oob_score=True,
        )
        self._model.fit(X_arr, y_arr)
        self._is_trained = True

        # Feature importances
        importances = dict(
            sorted(
                zip(self._feature_names, self._model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Distribución de clases
        unique, counts = np.unique(y_arr, return_counts=True)
        class_dist = {LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)}

        result = TrainResult(
            n_samples=len(X_arr),
            n_features=len(self._feature_names),
            feature_names=self._feature_names,
            class_distribution=class_dist,
            oob_score=float(self._model.oob_score_),
            feature_importances=importances,
        )

        log.info(
            "Entrenamiento completo — OOB score: %.3f | %s",
            result.oob_score or 0,
            result.class_distribution,
        )
        return result

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice la clase (BUY=1, HOLD=0, SELL=-1) para cada fila de X."""
        self._assert_trained()
        X_aligned = self._align_features(X)
        return self._model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predice probabilidades para cada clase.

        Retorna array (n_samples × 3) en orden [SELL, HOLD, BUY]
        (orden de clases en self._model.classes_).
        """
        self._assert_trained()
        X_aligned = self._align_features(X)
        return self._model.predict_proba(X_aligned)

    def predict_with_confidence(
        self, row: pd.Series | pd.DataFrame
    ) -> tuple[int, float]:
        """Predice la señal con su nivel de confianza para una sola fila.

        Retorna
        -------
        (signal, confidence) donde:
          signal = BUY(1) / HOLD(0) / SELL(-1)
          confidence = probabilidad de la clase predicha (0.0–1.0)

        Si la confianza máxima es menor que cfg.confidence_threshold,
        retorna (HOLD, max_proba) — no tomamos posición sin convicción.
        """
        if isinstance(row, pd.Series):
            row = row.to_frame().T

        proba = self.predict_proba(row)[0]
        classes = list(self._model.classes_)

        # Probabilidad para cada clase
        class_proba = {int(cls): float(p) for cls, p in zip(classes, proba)}
        buy_p = class_proba.get(LABEL_BUY, 0.0)
        sell_p = class_proba.get(LABEL_SELL, 0.0)
        hold_p = class_proba.get(LABEL_HOLD, 0.0)

        max_p = max(buy_p, sell_p, hold_p)

        if max_p < self.cfg.confidence_threshold:
            return LABEL_HOLD, max_p

        if buy_p == max_p:
            return LABEL_BUY, buy_p
        if sell_p == max_p:
            return LABEL_SELL, sell_p
        return LABEL_HOLD, hold_p

    def feature_importance_report(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Retorna las top-N features más importantes."""
        self._assert_trained()
        pairs = list(zip(self._feature_names, self._model.feature_importances_))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Guarda el modelo entrenado con pickle (vía joblib si disponible)."""
        self._assert_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            joblib.dump({"model": self._model, "features": self._feature_names, "cfg": self.cfg}, path)
            log.info("Modelo guardado con joblib en %s", path)
        except ImportError:
            with open(path, "wb") as f:
                pickle.dump({"model": self._model, "features": self._feature_names, "cfg": self.cfg}, f)
            log.info("Modelo guardado con pickle en %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SignalClassifier":
        """Carga un modelo previamente guardado."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        try:
            import joblib
            data = joblib.load(path)
        except ImportError:
            with open(path, "rb") as f:
                data = pickle.load(f)

        instance = cls(cfg=data.get("cfg"))
        instance._model = data["model"]
        instance._feature_names = data["features"]
        instance._is_trained = True
        log.info("Modelo cargado desde %s (%d features)", path, len(instance._feature_names))
        return instance

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _assert_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "El modelo no está entrenado. Llama a fit() primero."
            )

    def _align_features(self, X: pd.DataFrame) -> np.ndarray:
        """Alinea las columnas de X con las features del modelo."""
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            raise ValueError(
                f"Features faltantes en la predicción: {missing}. "
                f"Asegúrate de usar el mismo build_features() del entrenamiento."
            )
        return X[self._feature_names].values
