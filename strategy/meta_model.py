"""
meta_model.py — LightGBM Scorecard Wrapper (v2)

Wraps the LightGBM multi-class classifier (labels: -1, 0, +1) trained on
triple-barrier labeled OHLCV + feature vectors.

Prediction contract:
  predict_proba(fv) → float P(win) in [0.0, 1.0]
    ≥ 0.65  → full position size (multiplier 1.0)
    0.52–0.65 → half size (multiplier 0.5)
    < 0.52  → skip trade (multiplier 0.0)

Scorecard versioning:
  Models are saved as scorecard_v{N}.pkl and tracked in scorecard_meta.json.
  Pass model_path="models/scorecard_meta.json" to load the active version.
  On save(): writes scorecard_v{N+1}.pkl and updates scorecard_meta.json.
  Auto-rollback: if new WF accuracy < previous version, keeps the old version.
  Retains last 3 versions; older files are deleted automatically.
"""

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maps triple-barrier int labels → LightGBM class index
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}
WIN_CLASS_IDX = 2  # class index for label +1 (profit hit)

SCORECARD_META_FILENAME = "scorecard_meta.json"
KEEP_LAST_N_VERSIONS = 3


def _load_scorecard_meta(meta_path: Path) -> dict:
    """Read scorecard_meta.json; return empty skeleton if file doesn't exist."""
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {
        "active_version": 0,
        "versions": {},
        "rollback_available": False,
        "keep_last_n_versions": KEEP_LAST_N_VERSIONS,
    }


def _save_scorecard_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _resolve_active_pkl(meta_path: Path) -> Optional[Path]:
    """Given scorecard_meta.json path, return the active .pkl Path or None."""
    meta = _load_scorecard_meta(meta_path)
    active_v = meta.get("active_version", 0)
    if active_v == 0:
        return None
    vkey = f"v{active_v}"
    file_name = meta.get("versions", {}).get(vkey, {}).get("file")
    if not file_name:
        return None
    return meta_path.parent / file_name


class MetaModel:
    """LightGBM multi-class wrapper for the scorecard P(win) prediction.

    Parameters
    ----------
    model_path : str or Path
        Either a direct .pkl path (legacy) or a scorecard_meta.json path (versioned).
        When given scorecard_meta.json the active version's .pkl is resolved at load time.
    config : dict
        Full config.json dict (model section is read internally).
    """

    DEFAULT_PARAMS = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,  # was 20; increased to reduce overfitting
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        model_path: str = "models/scorecard_meta.json",
        config: Optional[Dict] = None,
    ):
        given = Path(model_path)
        self._cfg = (config or {}).get("model", config or {})
        self._model = None
        self._feature_names: List[str] = []
        self._trained_at: Optional[str] = None

        # Versioned mode: path points to scorecard_meta.json
        if given.suffix == ".json" and given.name == SCORECARD_META_FILENAME:
            self._meta_path: Optional[Path] = given
            pkl = _resolve_active_pkl(given)
            self._path = pkl if pkl else given.parent / "scorecard_v1.pkl"
        else:
            # Legacy mode: direct .pkl path
            self._meta_path = given.parent / SCORECARD_META_FILENAME
            self._path = given
        self._train_stats: Dict = {}

        # Auto-load: trigger when the pkl exists OR when a meta registry exists
        should_load = self._path.exists() or (
            self._meta_path is not None and self._meta_path.exists()
        )
        if should_load:
            self.load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None and len(self._feature_names) > 0

    def predict_proba(self, feature_vector) -> float:
        """Return P(win) = P(label=+1) for a single feature vector.

        Accepts a FeatureVector instance or a plain dict.
        Returns 0.0 if model not loaded.
        """
        if not self.is_ready:
            return 0.0

        try:
            if hasattr(feature_vector, "to_list"):
                values = feature_vector.to_list(self._feature_names)
            elif isinstance(feature_vector, dict):
                values = [feature_vector.get(k) for k in self._feature_names]
            else:
                values = list(feature_vector)

            # Replace None with NaN — LightGBM handles NaN natively
            values = [float(v) if v is not None else float("nan") for v in values]
            x = np.array(values, dtype=np.float32).reshape(1, -1)

            proba = self._model.predict_proba(x)[0]  # shape (3,)
            return float(proba[WIN_CLASS_IDX])

        except Exception:
            logger.exception("MetaModel.predict_proba failed — returning 0.0")
            return 0.0

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Return P(win) array for a 2D feature matrix."""
        if not self.is_ready:
            return np.zeros(len(X))
        proba = self._model.predict_proba(X)  # shape (N, 3)
        return proba[:, WIN_CLASS_IDX]

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_weights: Optional[Dict] = None,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict:
        """Train the LightGBM model.

        Parameters
        ----------
        X : DataFrame where columns are feature names (matches scorer.py output)
        y : Series of triple-barrier labels (-1, 0, +1)
        class_weights : dict {-1: w, 0: w, 1: w} — handles class imbalance
        eval_set : (X_val, y_val) for early stopping

        Returns training stats dict (stored in model artifact).
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed — run: pip install lightgbm==4.5.0")

        params = {**self.DEFAULT_PARAMS}
        params.update(self._cfg.get("lgbm_params", {}))

        # Convert labels -1/0/+1 → class indices 0/1/2
        y_enc = y.map(LABEL_TO_CLASS)

        # Build sample weights from class_weights dict
        sample_weights = None
        if class_weights:
            sample_weights = y.map({k: class_weights.get(k, 1.0) for k in y.unique()})

        eval_set_enc = None
        if eval_set:
            X_val, y_val = eval_set
            y_val_enc = y_val.map(LABEL_TO_CLASS)
            eval_set_enc = [(X_val, y_val_enc)]

        # early_stopping requires valid_sets — only add when we have a validation set
        callbacks = [lgb.log_evaluation(period=50)]
        if eval_set_enc:
            callbacks.append(lgb.early_stopping(stopping_rounds=30))

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X, y_enc,
            sample_weight=sample_weights,
            eval_set=eval_set_enc,
            callbacks=callbacks,
        )

        self._model = model
        self._feature_names = list(X.columns)
        self._trained_at = datetime.utcnow().isoformat()

        # Compute in-sample stats
        y_pred_enc = model.predict(X)
        y_pred = pd.Series(y_pred_enc).map(CLASS_TO_LABEL)
        accuracy = (y_pred.values == y.values).mean()

        self._train_stats = {
            "n_samples": len(X),
            "n_features": len(self._feature_names),
            "label_distribution": y.value_counts().to_dict(),
            "train_accuracy": round(accuracy, 4),
            "trained_at": self._trained_at,
            "best_iteration": getattr(model, "best_iteration_", params["n_estimators"]),
        }

        logger.info(
            "MetaModel trained: %d samples, %d features, accuracy=%.3f",
            len(X), len(self._feature_names), accuracy,
        )
        return self._train_stats

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        class_weights: Optional[Dict] = None,
    ) -> List[Dict]:
        """Walk-forward time-series cross-validation.

        Splits are temporal (no shuffling). Each fold trains on all past data
        and tests on the next future block. Returns per-fold metrics.
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            stats = self.train(
                X_tr, y_tr,
                class_weights=class_weights,
                eval_set=(X_val, y_val),
            )

            p_win = self.predict_batch(X_val.values)
            threshold_high = self._cfg.get("meta_model_confidence_high", 0.65)
            threshold_low = self._cfg.get("meta_model_confidence_low", 0.52)

            high_conf_mask = p_win >= threshold_high
            high_conf_accuracy = None
            if high_conf_mask.sum() > 0:
                y_val_mapped = y_val.map(LABEL_TO_CLASS).values
                # P(win) high-confidence → did label end up +1?
                preds_binary = (p_win[high_conf_mask] >= threshold_high).astype(int)
                actual_binary = (y_val_mapped[high_conf_mask] == WIN_CLASS_IDX).astype(int)
                high_conf_accuracy = round(
                    (preds_binary == actual_binary).mean(), 4
                )

            result = {
                "fold": fold + 1,
                "train_samples": len(X_tr),
                "val_samples": len(X_val),
                "train_accuracy": stats["train_accuracy"],
                "high_conf_trades": int(high_conf_mask.sum()),
                "high_conf_accuracy": high_conf_accuracy,
            }
            fold_results.append(result)
            logger.info("Fold %d/%d: %s", fold + 1, n_splits, result)

        return fold_results

    def save(self, symbols: Optional[List[str]] = None, wf_avg_accuracy: Optional[float] = None) -> None:
        """Serialize model + metadata to disk with scorecard versioning.

        Determines the next version number from scorecard_meta.json, writes
        scorecard_v{N}.pkl, then updates the registry. Keeps only the last
        KEEP_LAST_N_VERSIONS versions and deletes older .pkl files.

        Auto-rollback: if wf_avg_accuracy is provided and lower than the previous
        version's accuracy, the new version is NOT activated (rollback logged).
        """
        models_dir = self._path.parent
        models_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialise the registry
        meta_path = self._meta_path or (models_dir / SCORECARD_META_FILENAME)
        meta = _load_scorecard_meta(meta_path)

        prev_version = meta.get("active_version", 0)
        new_version = prev_version + 1
        new_file = f"scorecard_v{new_version}.pkl"
        new_pkl_path = models_dir / new_file

        # Rollback guard: skip activation if walk-forward accuracy regressed
        if wf_avg_accuracy is not None and prev_version > 0:
            prev_key = f"v{prev_version}"
            prev_wf = meta["versions"].get(prev_key, {}).get("wf_avg_accuracy")
            if prev_wf is not None and wf_avg_accuracy < prev_wf:
                logger.warning(
                    "ROLLBACK: new WF accuracy %.4f < previous %.4f — keeping v%d active",
                    wf_avg_accuracy, prev_wf, prev_version,
                )
                return

        # Write .pkl artifact
        artifact = {
            "model": self._model,
            "feature_names": self._feature_names,
            "trained_at": self._trained_at,
            "train_stats": self._train_stats,
        }
        with open(new_pkl_path, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Mark previous version as retired
        if prev_version > 0:
            prev_key = f"v{prev_version}"
            if prev_key in meta["versions"]:
                meta["versions"][prev_key]["retired_at"] = datetime.utcnow().date().isoformat()

        # Register the new version
        meta["versions"][f"v{new_version}"] = {
            "file": new_file,
            "trained_at": self._trained_at,
            "n_samples": self._train_stats.get("n_samples", 0),
            "n_features": len(self._feature_names),
            "symbols": symbols or [],
            "train_accuracy": self._train_stats.get("train_accuracy"),
            "wf_avg_accuracy": wf_avg_accuracy,
            "retired_at": None,
        }
        meta["active_version"] = new_version
        meta["rollback_available"] = prev_version > 0
        meta["keep_last_n_versions"] = KEEP_LAST_N_VERSIONS

        # Prune old versions beyond the keep-limit
        all_versions = sorted(
            (int(k[1:]) for k in meta["versions"]),
            reverse=True,
        )
        for old_v in all_versions[KEEP_LAST_N_VERSIONS:]:
            old_key = f"v{old_v}"
            old_file = meta["versions"].get(old_key, {}).get("file")
            if old_file:
                old_path = models_dir / old_file
                if old_path.exists():
                    old_path.unlink()
                    logger.info("Deleted old scorecard version: %s", old_file)
            meta["versions"].pop(old_key, None)

        _save_scorecard_meta(meta_path, meta)
        self._path = new_pkl_path
        logger.info("Scorecard saved → %s (active v%d)", new_pkl_path, new_version)

    def load(self) -> None:
        """Load model artifact from disk.

        If self._meta_path points to scorecard_meta.json, resolve the active
        version's .pkl first before loading. Falls back gracefully if not found.
        """
        # Re-resolve active pkl in case scorecard_meta.json changed since __init__
        if self._meta_path and self._meta_path.exists():
            resolved = _resolve_active_pkl(self._meta_path)
            if resolved:
                self._path = resolved

        if not self._path.exists():
            logger.warning("Scorecard not found at %s — model not loaded", self._path)
            self._model = None
            return

        try:
            with open(self._path, "rb") as f:
                artifact = pickle.load(f)
            self._model = artifact["model"]
            self._feature_names = artifact["feature_names"]
            self._trained_at = artifact.get("trained_at")
            self._train_stats = artifact.get("train_stats", {})
            logger.info(
                "Scorecard loaded from %s — %d features, trained %s",
                self._path, len(self._feature_names), self._trained_at,
            )
        except Exception:
            logger.exception("Failed to load scorecard from %s", self._path)
            self._model = None

    def get_active_version_info(self) -> dict:
        """Return the active version's metadata from scorecard_meta.json."""
        if not self._meta_path:
            return {}
        meta = _load_scorecard_meta(self._meta_path)
        active_v = meta.get("active_version", 0)
        if active_v == 0:
            return {}
        return meta["versions"].get(f"v{active_v}", {})

    def feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Return top-N most important features (gain-based)."""
        if not self.is_ready:
            return pd.DataFrame()
        importances = self._model.booster_.feature_importance(importance_type="gain")
        df = pd.DataFrame({
            "feature": self._feature_names,
            "importance_gain": importances,
        }).sort_values("importance_gain", ascending=False).head(top_n)
        df["importance_pct"] = round(df["importance_gain"] / df["importance_gain"].sum() * 100, 2)
        return df.reset_index(drop=True)
