"""Tests for strategy.meta_model — LightGBM wrapper."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy.meta_model import MetaModel, LABEL_TO_CLASS, CLASS_TO_LABEL, WIN_CLASS_IDX


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dummy_dataset(n: int = 200, n_features: int = 10, seed: int = 42):
    """Synthetic X, y for fast unit testing (no real OHLCV needed)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Balanced classes
    y = pd.Series(np.tile([-1, 0, 1], n // 3 + 1)[:n], dtype=int)
    return X, y


@pytest.fixture
def tmp_model_path(tmp_path):
    return str(tmp_path / "test_meta_model_lgbm.pkl")


@pytest.fixture
def trained_model(tmp_model_path):
    """A MetaModel trained on tiny synthetic data."""
    X, y = _make_dummy_dataset(n=300)
    model = MetaModel(model_path=tmp_model_path, config={})
    model.train(X, y)
    return model


# ── Construction & state tests ────────────────────────────────────────────────

def test_model_not_ready_before_training(tmp_model_path):
    model = MetaModel(model_path=tmp_model_path)
    assert not model.is_ready


def test_model_ready_after_training(tmp_model_path):
    X, y = _make_dummy_dataset()
    model = MetaModel(model_path=tmp_model_path)
    model.train(X, y)
    assert model.is_ready


# ── Training tests ────────────────────────────────────────────────────────────

def test_train_returns_stats(tmp_model_path):
    X, y = _make_dummy_dataset()
    model = MetaModel(model_path=tmp_model_path)
    stats = model.train(X, y)
    assert "n_samples" in stats
    assert "n_features" in stats
    assert "train_accuracy" in stats
    assert 0.0 <= stats["train_accuracy"] <= 1.0


def test_train_stores_feature_names(tmp_model_path):
    X, y = _make_dummy_dataset(n_features=15)
    model = MetaModel(model_path=tmp_model_path)
    model.train(X, y)
    assert model._feature_names == [f"feat_{i}" for i in range(15)]


def test_train_with_class_weights(tmp_model_path):
    X, y = _make_dummy_dataset()
    model = MetaModel(model_path=tmp_model_path)
    weights = {-1: 2.0, 0: 1.0, 1: 1.5}
    stats = model.train(X, y, class_weights=weights)
    assert stats is not None


# ── predict_proba tests ───────────────────────────────────────────────────────

def test_predict_proba_returns_float(trained_model):
    X, _ = _make_dummy_dataset(n=1)
    row = X.iloc[0].to_dict()
    result = trained_model.predict_proba(row)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_predict_proba_accepts_dict(trained_model):
    X, _ = _make_dummy_dataset(n=1)
    row = X.iloc[0].to_dict()
    result = trained_model.predict_proba(row)
    assert 0.0 <= result <= 1.0


def test_predict_proba_accepts_feature_vector(trained_model):
    """predict_proba() should work with a FeatureVector via to_list()."""
    from strategy.scorer import FeatureVector

    # Build a mock FeatureVector with the right keys
    fv = FeatureVector()
    fv._data = {name: 0.0 for name in trained_model._feature_names}
    result = trained_model.predict_proba(fv)
    assert 0.0 <= result <= 1.0


def test_predict_proba_returns_zero_when_not_ready(tmp_model_path):
    model = MetaModel(model_path=tmp_model_path)
    result = model.predict_proba({"feat_0": 1.0})
    assert result == 0.0


def test_predict_proba_handles_none_values(trained_model):
    """None feature values should map to NaN — not crash."""
    row = {name: None for name in trained_model._feature_names}
    result = trained_model.predict_proba(row)
    assert isinstance(result, float)


# ── predict_batch tests ───────────────────────────────────────────────────────

def test_predict_batch_shape(trained_model):
    X, _ = _make_dummy_dataset(n=20)
    results = trained_model.predict_batch(X.values)
    assert results.shape == (20,)
    assert np.all((results >= 0.0) & (results <= 1.0))


def test_predict_batch_returns_zeros_when_not_ready(tmp_model_path):
    model = MetaModel(model_path=tmp_model_path)
    X = np.zeros((10, 5))
    results = model.predict_batch(X)
    assert np.all(results == 0.0)


# ── Save / load round-trip ────────────────────────────────────────────────────

def test_save_creates_pkl_and_json(trained_model, tmp_model_path):
    trained_model.save()
    assert Path(tmp_model_path).exists()
    assert Path(tmp_model_path).with_suffix(".json").exists()


def test_load_restores_model(tmp_model_path):
    X, y = _make_dummy_dataset(n=200)
    model_a = MetaModel(model_path=tmp_model_path)
    stats_a = model_a.train(X, y)
    model_a.save()

    model_b = MetaModel(model_path=tmp_model_path)
    assert model_b.is_ready
    assert model_b._feature_names == model_a._feature_names

    # Predictions should be identical after reload
    row = X.iloc[5].to_dict()
    assert model_a.predict_proba(row) == pytest.approx(model_b.predict_proba(row), abs=1e-6)


def test_load_missing_file_does_not_crash(tmp_model_path):
    model = MetaModel(model_path=tmp_model_path + "_nonexistent.pkl")
    assert not model.is_ready


# ── Walk-forward validation ───────────────────────────────────────────────────

def test_walk_forward_returns_fold_results(tmp_model_path):
    X, y = _make_dummy_dataset(n=400)
    model = MetaModel(model_path=tmp_model_path)
    folds = model.walk_forward_validate(X, y, n_splits=3)
    assert len(folds) == 3
    for fold in folds:
        assert "fold" in fold
        assert "train_accuracy" in fold
        assert "val_samples" in fold


# ── Feature importance ────────────────────────────────────────────────────────

def test_feature_importance_returns_dataframe(trained_model):
    df = trained_model.feature_importance(top_n=5)
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "importance_gain" in df.columns
    assert len(df) <= 5


def test_feature_importance_empty_when_not_trained(tmp_model_path):
    model = MetaModel(model_path=tmp_model_path)
    df = model.feature_importance()
    assert df.empty


# ── Label encoding consistency ─────────────────────────────────────────────────

def test_label_class_maps_are_inverse():
    for label, cls in LABEL_TO_CLASS.items():
        assert CLASS_TO_LABEL[cls] == label


def test_win_class_idx_maps_to_plus_one():
    assert CLASS_TO_LABEL[WIN_CLASS_IDX] == 1
