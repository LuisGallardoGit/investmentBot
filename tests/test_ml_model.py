"""Tests para ml/model.py — SignalClassifier — Fase 6."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jarvis_bot.ml.model import (
    LABEL_BUY,
    LABEL_HOLD,
    LABEL_SELL,
    LABEL_NAMES,
    LABELS,
    ModelConfig,
    SignalClassifier,
    TrainResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_xy(n: int = 150, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.choice(LABELS, size=n), name="target")
    return X, y


def _trained_clf(n: int = 150) -> SignalClassifier:
    X, y = _make_xy(n)
    clf = SignalClassifier()
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_estimators == 200
        assert cfg.max_depth == 6
        assert cfg.min_samples_leaf == 20
        assert cfg.class_weight == "balanced"
        assert cfg.confidence_threshold == 0.45
        assert cfg.n_jobs == -1

    def test_custom(self):
        cfg = ModelConfig(n_estimators=50, max_depth=3)
        assert cfg.n_estimators == 50
        assert cfg.max_depth == 3


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_label_values(self):
        assert LABEL_BUY == 1
        assert LABEL_HOLD == 0
        assert LABEL_SELL == -1

    def test_label_names(self):
        assert LABEL_NAMES[LABEL_BUY] == "BUY"
        assert LABEL_NAMES[LABEL_HOLD] == "HOLD"
        assert LABEL_NAMES[LABEL_SELL] == "SELL"

    def test_labels_list(self):
        assert set(LABELS) == {-1, 0, 1}


# ---------------------------------------------------------------------------
# SignalClassifier — entrenamiento
# ---------------------------------------------------------------------------


class TestSignalClassifierFit:
    def test_fit_returns_train_result(self):
        X, y = _make_xy()
        clf = SignalClassifier()
        result = clf.fit(X, y)
        assert isinstance(result, TrainResult)

    def test_is_trained_after_fit(self):
        clf = _trained_clf()
        assert clf.is_trained

    def test_not_trained_before_fit(self):
        clf = SignalClassifier()
        assert not clf.is_trained

    def test_train_result_n_samples(self):
        n = 120
        X, y = _make_xy(n)
        clf = SignalClassifier()
        result = clf.fit(X, y)
        assert result.n_samples == n

    def test_train_result_n_features(self):
        X, y = _make_xy(n_features=7)
        clf = SignalClassifier()
        result = clf.fit(X, y)
        assert result.n_features == 7

    def test_train_result_feature_names(self):
        X, y = _make_xy()
        clf = SignalClassifier()
        result = clf.fit(X, y)
        assert result.feature_names == list(X.columns)

    def test_oob_score_present(self):
        X, y = _make_xy()
        clf = SignalClassifier()
        result = clf.fit(X, y)
        assert result.oob_score is not None
        assert 0.0 <= result.oob_score <= 1.0

    def test_class_distribution_keys(self):
        X, y = _make_xy()
        clf = SignalClassifier()
        result = clf.fit(X, y)
        for k in result.class_distribution:
            assert k in ("BUY", "HOLD", "SELL")

    def test_feature_importances_sum_to_one(self):
        X, y = _make_xy()
        clf = SignalClassifier()
        clf.fit(X, y)
        imps = list(clf.feature_importance_report(top_n=100))
        total = sum(v for _, v in imps)
        assert abs(total - 1.0) < 1e-9

    def test_fit_empty_x_raises(self):
        X = pd.DataFrame(columns=["f0"])
        y = pd.Series([], dtype=int)
        clf = SignalClassifier()
        with pytest.raises(ValueError, match="vacío|empty"):
            clf.fit(X, y)

    def test_fit_mismatched_lengths_raises(self):
        X, y = _make_xy(n=10)
        clf = SignalClassifier()
        with pytest.raises(ValueError, match="longitud"):
            clf.fit(X, y.iloc[:5])

    def test_feature_names_stored(self):
        X, y = _make_xy()
        clf = SignalClassifier()
        clf.fit(X, y)
        assert clf.feature_names == list(X.columns)


# ---------------------------------------------------------------------------
# SignalClassifier — predicción
# ---------------------------------------------------------------------------


class TestSignalClassifierPredict:
    def test_predict_returns_array(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=10)
        preds = clf.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 10

    def test_predict_values_in_labels(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=20)
        preds = clf.predict(X)
        assert set(preds).issubset({-1, 0, 1})

    def test_predict_proba_shape(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=10)
        proba = clf.predict_proba(X)
        # 3 clases
        assert proba.shape == (10, 3)

    def test_predict_proba_rows_sum_to_one(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=10)
        proba = clf.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=1e-9)

    def test_predict_before_fit_raises(self):
        clf = SignalClassifier()
        X, _ = _make_xy(n=5)
        with pytest.raises(RuntimeError, match="no está entrenado"):
            clf.predict(X)

    def test_predict_missing_feature_raises(self):
        clf = _trained_clf()
        X = pd.DataFrame({"unknown": [1.0, 2.0]})
        with pytest.raises(ValueError, match="faltantes"):
            clf.predict(X)

    def test_predict_with_confidence_returns_tuple(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=1)
        row = X.iloc[0]
        label, conf = clf.predict_with_confidence(row)
        assert label in LABELS
        assert 0.0 <= conf <= 1.0

    def test_predict_with_confidence_accepts_dataframe_row(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=1)
        label, conf = clf.predict_with_confidence(X)
        assert label in LABELS

    def test_low_confidence_returns_hold(self):
        """Con threshold muy alto (0.99) casi siempre retorna HOLD."""
        cfg = ModelConfig(
            n_estimators=10,
            min_samples_leaf=1,
            confidence_threshold=0.99,
        )
        X, y = _make_xy(n=100)
        clf = SignalClassifier(cfg)
        clf.fit(X, y)
        X_test, _ = _make_xy(n=50, seed=99)
        results = [clf.predict_with_confidence(X_test.iloc[i]) for i in range(50)]
        holds = sum(1 for label, _ in results if label == LABEL_HOLD)
        # Con threshold 0.99 esperamos mayoría HOLD
        assert holds >= 20


# ---------------------------------------------------------------------------
# feature_importance_report
# ---------------------------------------------------------------------------


class TestFeatureImportanceReport:
    def test_returns_list_of_tuples(self):
        clf = _trained_clf()
        report = clf.feature_importance_report()
        assert isinstance(report, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in report)

    def test_sorted_descending(self):
        clf = _trained_clf()
        report = clf.feature_importance_report()
        importances = [imp for _, imp in report]
        assert importances == sorted(importances, reverse=True)

    def test_top_n_respected(self):
        clf = _trained_clf()
        report = clf.feature_importance_report(top_n=3)
        assert len(report) <= 3

    def test_before_fit_raises(self):
        clf = SignalClassifier()
        with pytest.raises(RuntimeError):
            clf.feature_importance_report()


# ---------------------------------------------------------------------------
# Persistencia save / load
# ---------------------------------------------------------------------------


class TestSignalClassifierPersistence:
    def test_save_and_load_roundtrip(self):
        clf = _trained_clf()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf2 = SignalClassifier.load(path)
            assert clf2.is_trained
            assert clf2.feature_names == clf.feature_names

    def test_loaded_model_predicts_same(self):
        clf = _trained_clf()
        X, _ = _make_xy(n=10)
        preds_orig = clf.predict(X)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf2 = SignalClassifier.load(path)
            preds_loaded = clf2.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            SignalClassifier.load("/nonexistent/path.pkl")

    def test_save_creates_parent_dirs(self):
        clf = _trained_clf()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "subdir" / "model.pkl"
            clf.save(path)
            assert path.exists()

    def test_loaded_config_preserved(self):
        cfg = ModelConfig(confidence_threshold=0.60, n_estimators=10)
        X, y = _make_xy(n=100)
        clf = SignalClassifier(cfg)
        clf.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            clf.save(path)
            clf2 = SignalClassifier.load(path)
            assert clf2.cfg.confidence_threshold == 0.60
