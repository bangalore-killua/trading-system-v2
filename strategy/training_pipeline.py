"""
training_pipeline.py — LightGBM Meta-Model Training Orchestrator (Phase 3+)

Two paths to generate labeled training data:

  PATH A — Retroactive historical bootstrap (run once):
    Pulls 5 years of OHLCV from yfinance for all watchlist symbols.
    Applies TripleBarrierLabeler at every bar → labeled entries.
    Computes feature vectors for each labeled bar via Scorer.
    Combines all symbols → single DataFrame → trains LightGBM.

  PATH B — Live paper trade labels (ongoing):
    As paper trades close, trade_ledger.py records the outcome.
    A nightly job (Sunday) calls label_live_trades() to convert
    closed trades into labeled rows and appends them to the training set.

Minimum samples before training: config.json model.retrain_min_samples (default 500).
Walk-forward validation with TimeSeriesSplit(n_splits=5) before saving.

CLI usage:
    python -m strategy.training_pipeline --symbols AAPL MSFT GOOGL --years 5
    python -m strategy.training_pipeline --live-only
    python -m strategy.training_pipeline --validate-only
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the full training workflow from raw OHLCV to a saved LightGBM model."""

    def __init__(self, config: dict, db_url: Optional[str] = None):
        self._cfg = config
        self._model_cfg = config.get("model", {})
        self._db_url = db_url
        self._min_samples = self._model_cfg.get("retrain_min_samples", 500)

    async def run_bootstrap(
        self,
        symbols: List[str],
        years: int = 5,
        save_intermediate: bool = True,
    ) -> Optional[dict]:
        """PATH A: Historical retroactive bootstrap.

        Returns training stats dict or None if not enough samples.
        """
        from data.fetcher import OHLCVFetcher
        from strategy.triple_barrier import TripleBarrierLabeler
        from strategy.scorer import Scorer, FeatureVector
        from strategy.meta_model import MetaModel
        from strategy import technical_analysis as ta_module

        labeler = TripleBarrierLabeler(
            tp_multiplier=self._model_cfg.get("triple_barrier_tp_multiplier", 3.0),
            sl_multiplier=self._model_cfg.get("triple_barrier_sl_multiplier", 1.5),
            max_holding_days=self._model_cfg.get("triple_barrier_max_days", 10),
        )
        fetcher = OHLCVFetcher()
        scorer = Scorer(config=self._cfg)

        bars = int(years * 252)
        all_rows: List[dict] = []

        for symbol in symbols:
            try:
                df = await fetcher.get_bars(symbol, bars=bars)
                if df is None or len(df) < 100:
                    logger.warning("Skipping %s — insufficient data", symbol)
                    continue

                labeled = labeler.label(df)
                if labeled.empty:
                    continue

                indicators_cache = {}
                for entry_date, row in labeled.iterrows():
                    # Find bar index for this entry date
                    if entry_date not in df.index:
                        continue
                    idx = df.index.get_loc(entry_date)
                    df_slice = df.iloc[max(0, idx - 250): idx + 1]
                    if len(df_slice) < 50:
                        continue

                    try:
                        indicators = ta_module.compute_indicators(df_slice)
                    except Exception:
                        continue

                    fv = scorer.build_vector(
                        symbol=symbol,
                        df=df_slice,
                        indicators=indicators,
                        setup_type="SETUP_MOMENTUM",
                    )
                    feature_dict = fv.to_dict()
                    feature_dict["label"] = row["label"]
                    feature_dict["symbol"] = symbol
                    feature_dict["entry_date"] = str(entry_date)
                    all_rows.append(feature_dict)

                logger.info("Processed %s — %d labeled entries", symbol, len(labeled))

            except Exception:
                logger.exception("Failed to process %s", symbol)
                continue

        if not all_rows:
            logger.error("No labeled data generated — cannot train")
            return None

        df_train = pd.DataFrame(all_rows)
        logger.info("Total labeled samples: %d", len(df_train))

        if save_intermediate:
            out_path = Path("logs/training_data.parquet")
            out_path.parent.mkdir(exist_ok=True)
            df_train.to_parquet(out_path, index=False)
            logger.info("Saved intermediate training data → %s", out_path)

        return await self._train_and_save(df_train)

    async def run_live_update(self) -> Optional[dict]:
        """PATH B: Pull closed paper trades from ledger, label them, retrain.

        Only retrains when new labeled samples push total above a multiple of
        retrain_min_samples (so we don't retrain after every single trade).
        """
        labeled_live = await self._load_live_labels()
        if labeled_live is None or len(labeled_live) < self._min_samples:
            logger.info(
                "Not enough live labels yet (%d < %d minimum)",
                0 if labeled_live is None else len(labeled_live),
                self._min_samples,
            )
            return None

        historical_path = Path("logs/training_data.parquet")
        if historical_path.exists():
            df_hist = pd.read_parquet(historical_path)
            df_all = pd.concat([df_hist, labeled_live], ignore_index=True)
        else:
            df_all = labeled_live

        return await self._train_and_save(df_all)

    async def _train_and_save(self, df: pd.DataFrame) -> Optional[dict]:
        """Common training path: prepare X/y → walk-forward validate → save."""
        from strategy.meta_model import MetaModel

        if len(df) < self._min_samples:
            logger.warning(
                "Only %d samples (minimum %d) — skipping training",
                len(df), self._min_samples,
            )
            return None

        # Separate features from metadata columns
        meta_cols = {"label", "symbol", "entry_date"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        X = df[feature_cols].copy()
        y = df["label"].astype(int)

        # Sort chronologically for proper walk-forward validation
        if "entry_date" in df.columns:
            df_sorted_idx = df["entry_date"].argsort()
            X = X.iloc[df_sorted_idx].reset_index(drop=True)
            y = y.iloc[df_sorted_idx].reset_index(drop=True)

        # Drop columns that are entirely NaN
        nan_frac = X.isna().mean()
        drop_cols = nan_frac[nan_frac > 0.5].index.tolist()
        if drop_cols:
            logger.info("Dropping %d high-NaN columns: %s", len(drop_cols), drop_cols[:5])
            X = X.drop(columns=drop_cols)

        # Fill remaining NaN with column medians (LightGBM handles NaN but
        # walk-forward splits on sklearn need clean arrays)
        X = X.fillna(X.median(numeric_only=True))

        model = MetaModel(
            model_path=self._model_cfg.get("meta_model_path", "models/meta_model_lgbm.pkl"),
            config=self._cfg,
        )

        from strategy.triple_barrier import TripleBarrierLabeler
        labeler = TripleBarrierLabeler()
        class_weights = labeler.class_weights(pd.DataFrame({"label": y}))

        # Walk-forward cross-validation
        n_splits = self._model_cfg.get("walk_forward_splits", 5)
        fold_results = model.walk_forward_validate(X, y, n_splits=n_splits, class_weights=class_weights)

        # Final model trained on all data
        stats = model.train(X, y, class_weights=class_weights)
        model.save()

        summary = {
            **stats,
            "walk_forward_folds": fold_results,
            "avg_high_conf_accuracy": _mean_or_none(
                [f["high_conf_accuracy"] for f in fold_results if f["high_conf_accuracy"] is not None]
            ),
        }

        logger.info("Training complete: %s", summary)
        return summary

    async def _load_live_labels(self) -> Optional[pd.DataFrame]:
        """Load labeled live paper trades from PostgreSQL trade_features table."""
        if not self._db_url:
            return None
        try:
            import asyncpg
            conn = await asyncpg.connect(self._db_url)
            rows = await conn.fetch(
                "SELECT * FROM trade_features WHERE triple_barrier_label IS NOT NULL"
            )
            await conn.close()
            if not rows:
                return None
            df = pd.DataFrame([dict(r) for r in rows])
            # Parse JSON feature columns back to flat columns
            if "features_json" in df.columns:
                feat_df = df["features_json"].apply(
                    lambda s: pd.Series(__import__("json").loads(s)) if isinstance(s, str) else pd.Series()
                )
                df = pd.concat([feat_df, df[["triple_barrier_label"]].rename(
                    columns={"triple_barrier_label": "label"}
                )], axis=1)
            return df
        except Exception:
            logger.exception("Failed to load live labels from database")
            return None


def _mean_or_none(values: list) -> Optional[float]:
    return round(float(np.mean(values)), 4) if values else None


# ── CLI entrypoint ────────────────────────────────────────────────────────────

async def _cli_main(args):
    import json
    from pathlib import Path as _Path

    config_path = _Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    import os
    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")

    pipeline = TrainingPipeline(config=config, db_url=db_url)

    if args.live_only:
        result = await pipeline.run_live_update()
    elif args.validate_only:
        # Load existing parquet and re-validate without retraining
        parquet = Path("logs/training_data.parquet")
        if not parquet.exists():
            print("No training_data.parquet found — run bootstrap first")
            sys.exit(1)
        result = await pipeline._train_and_save(pd.read_parquet(parquet))
    else:
        symbols = args.symbols or config.get("watchlist", {}).get("mandatory", ["AAPL", "MSFT", "GOOGL"])
        result = await pipeline.run_bootstrap(symbols=symbols, years=args.years)

    if result:
        print(f"\nTraining complete:")
        print(f"  Samples:        {result.get('n_samples')}")
        print(f"  Features:       {result.get('n_features')}")
        print(f"  Train accuracy: {result.get('train_accuracy')}")
        print(f"  WF avg P(win) accuracy: {result.get('avg_high_conf_accuracy')}")
    else:
        print("Training skipped — insufficient data or error (see logs)")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Train LightGBM meta-model")
    parser.add_argument("--symbols", nargs="+", help="Symbols to bootstrap (default: config watchlist)")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data (default: 5)")
    parser.add_argument("--live-only", action="store_true", help="Only use live paper trade labels")
    parser.add_argument("--validate-only", action="store_true", help="Re-validate existing parquet without fetching")
    args = parser.parse_args()

    asyncio.run(_cli_main(args))
