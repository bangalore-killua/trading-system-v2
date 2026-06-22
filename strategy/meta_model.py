"""
meta_model.py — LightGBM Meta-Model Wrapper (Phase 3+)

Wraps the LightGBM multi-class classifier (labels: -1, 0, +1) trained on
triple-barrier labeled OHLCV + feature vectors.

Prediction contract:
  predict_proba(fv) → float P(win) in [0.0, 1.0]
    ≥ 0.65  → full position size (multiplier 1.0)
    0.52–0.65 → half size (multiplier 0.5)
    < 0.52  → skip trade (multiplier 0.0)

The Scorer class in scorer.py holds a MetaModel instance and calls it only
when meta_model_enabled=true in config.json. In Phase 1/2 it never executes.

Training flow (see training_pipeline.py):
  1. Retroactively label 5yr OHLCV via TripleBarrierLabeler
  2. Compute feature vectors for each labeled bar
  3. Walk-forward split (TimeSeriesSplit, n_splits=5)
  4. Train LightGBM on each fold
  5. Serialize final model to models/meta_model_lgbm.pkl
"""

import json
import logging
import pickle
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


class MetaModel:
    """LightGBM multi-class wrapper for the meta-labeling P(win) prediction.

    Parameters
    ----------
    model_path : str or Path
        Where to save / load the serialized model artifact.
    config : dict
        model section from config.json.
    """

    DEFAULT_PARAMS = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
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
        model_path: str = "models/meta_model_lgbm.pkl",
        config: Optional[Dict] = None,
    ):
        self._path = Path(model_path)
        self._cfg = (config or {}).get("model", config or {})
        self._model = None
        self._feature_names: List[str] = []
        self._trained_at: Optional[str] = None
        self._train_stats: Dict = {}

        if self._path.exists():
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

    def save(self) -> None:
        """Serialize model + metadata to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self._model,
            "feature_names": self._feature_names,
            "trained_at": self._trained_at,
            "train_stats": self._train_stats,
        }
        with open(self._path, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write a companion JSON for quick inspection without loading pickle
        meta_path = self._path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "trained_at": self._trained_at,
                    "n_features": len(self._feature_names),
                    "feature_names": self._feature_names,
                    **self._train_stats,
                },
                f, indent=2, default=str,
            )
        logger.info("MetaModel saved → %s", self._path)

    def load(self) -> None:
        """Load model artifact from disk."""
        try:
            with open(self._path, "rb") as f:
                artifact = pickle.load(f)
            self._model = artifact["model"]
            self._feature_names = artifact["feature_names"]
            self._trained_at = artifact.get("trained_at")
            self._train_stats = artifact.get("train_stats", {})
            logger.info(
                "MetaModel loaded from %s — %d features, trained %s",
                self._path, len(self._feature_names), self._trained_at,
            )
        except Exception:
            logger.exception("Failed to load MetaModel from %s", self._path)
            self._model = None

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
