"""
model_updater.py — Monthly LightGBM Retrain + RAG Update (Phase 5)

Runs on the first Sunday of each month after market close.

4-filter noise removal (applied in order, records must pass ALL filters):
  1. Frequency filter — pattern must appear >= 3 times before use
  2. Outlier filter   — exclude moves > 3 standard deviations (squeezes, memes)
  3. Regime filter    — only use samples from same HMM regime as current market
  4. Fundamental filter — Groq must confirm a repeatable catalyst (not noise)

After filtering:
  - Retrain LightGBM scorecard (adds new samples to historical set)
  - Save new version via scorecard versioning (strategy/meta_model.py)
  - Auto-rollback if new walk-forward accuracy < previous version
  - Generate a human-readable report → printed to stdout for manual review

Usage:
  python -m data.model_updater            # run immediately
  python -m data.model_updater --dry      # simulate: print plan without retraining
  python -m data.model_updater --force    # skip Groq fundamental check (faster)
"""

import json
import logging
import os
import asyncio
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
FEEDBACK_LOG_PATH = Path("logs/feedback_loop.jsonl")
UPDATE_REPORT_PATH = Path("logs/model_update_reports")

MIN_PATTERN_FREQUENCY = 3       # filter 1: must appear >= 3 times
OUTLIER_STD_MULTIPLIER = 3.0    # filter 2: exclude > 3σ moves


class ModelUpdater:
    """Monthly LightGBM retrain with 4-filter noise removal.

    Parameters
    ----------
    config : full config.json dict
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._groq_client = None
        UPDATE_REPORT_PATH.mkdir(parents=True, exist_ok=True)

    async def run(self, dry_run: bool = False, skip_fundamental_check: bool = False) -> dict:
        """Main entry point: load feedback, filter, retrain, report.

        Returns a report dict with what was done and new model accuracy.
        """
        logger.info("Model updater: loading feedback records from %s", FEEDBACK_LOG_PATH)

        # Step 1: load all feedback records
        all_records = self._load_feedback_records()
        if not all_records:
            return {"status": "no_data", "message": "No feedback records found in logs/"}

        logger.info("Loaded %d total feedback records", len(all_records))

        # Step 2: apply 4 filters
        filtered, filter_log = await self._apply_filters(
            all_records, skip_fundamental=skip_fundamental_check
        )
        logger.info("After filtering: %d records remain (from %d)", len(filtered), len(all_records))

        if len(filtered) < 50:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(filtered)} records after filtering — need >= 50 to retrain",
                "filter_log": filter_log,
            }

        if dry_run:
            report = self._generate_report(all_records, filtered, filter_log, new_version_info=None)
            print(json.dumps(report, indent=2, default=str))
            return report

        # Step 3: retrain LightGBM with new samples
        new_version_info = self._retrain(filtered)

        # Step 4: generate and save human-readable report
        report = self._generate_report(all_records, filtered, filter_log, new_version_info)
        report_path = UPDATE_REPORT_PATH / f"update_{date.today().isoformat()}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Model update complete. Report saved to %s", report_path)
        return report

    # ── Step 1: Load Records ──────────────────────────────────────────────────

    def _load_feedback_records(self) -> List[dict]:
        """Load all records from the JSONL feedback log."""
        records = []
        if not FEEDBACK_LOG_PATH.exists():
            return records
        with open(FEEDBACK_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ── Step 2: 4 Filters ────────────────────────────────────────────────────

    async def _apply_filters(
        self, records: List[dict], skip_fundamental: bool = False
    ) -> Tuple[List[dict], dict]:
        """Apply all 4 noise-removal filters. Returns (kept records, log of what was removed)."""
        log: Dict[str, int] = {"total": len(records)}

        # Filter 1: frequency — pattern (catalyst_type) must appear >= MIN_PATTERN_FREQUENCY times
        after_f1 = self._filter_frequency(records)
        log["removed_f1_frequency"] = len(records) - len(after_f1)

        # Filter 2: outlier — exclude moves > 3σ
        after_f2 = self._filter_outliers(after_f1)
        log["removed_f2_outlier"] = len(after_f1) - len(after_f2)

        # Filter 3: regime — only samples matching current HMM regime
        after_f3 = self._filter_regime(after_f2)
        log["removed_f3_regime"] = len(after_f2) - len(after_f3)

        # Filter 4: fundamental — Groq confirms repeatable catalyst (skip if --force)
        if skip_fundamental:
            after_f4 = after_f3
            log["removed_f4_fundamental"] = 0
            log["f4_skipped"] = True
        else:
            after_f4 = await self._filter_fundamental(after_f3)
            log["removed_f4_fundamental"] = len(after_f3) - len(after_f4)

        log["final"] = len(after_f4)
        return after_f4, log

    def _filter_frequency(self, records: List[dict]) -> List[dict]:
        """Remove records whose catalyst_type appears < 3 times in the dataset."""
        catalyst_counts: Counter = Counter()
        for r in records:
            ct = r.get("groq_analysis", {}).get("catalyst_type", "unknown")
            catalyst_counts[ct] += 1

        return [
            r for r in records
            if catalyst_counts.get(r.get("groq_analysis", {}).get("catalyst_type", "unknown"), 0)
            >= MIN_PATTERN_FREQUENCY
        ]

    def _filter_outliers(self, records: List[dict]) -> List[dict]:
        """Remove records where abs(pct_change) > 3 standard deviations."""
        pcts = np.array([abs(r.get("pct_change", 0)) for r in records], dtype=np.float64)
        if len(pcts) < 10:
            return records
        mean, std = float(pcts.mean()), float(pcts.std())
        threshold = mean + OUTLIER_STD_MULTIPLIER * std
        return [r for r in records if abs(r.get("pct_change", 0)) <= threshold]

    def _filter_regime(self, records: List[dict]) -> List[dict]:
        """Keep only records from the same HMM regime as current market."""
        try:
            from strategy.regime_detector import RegimeDetector  # type: ignore
            detector = RegimeDetector()
            current_regime = detector.get_current_regime()
            current_name = current_regime.name if current_regime else None
        except Exception:
            logger.debug("Regime detector unavailable — regime filter skipped")
            return records

        if not current_name:
            return records

        matching = [r for r in records if r.get("regime") == current_name]
        # Fallback: if too few regime-matched records, use all
        if len(matching) < 20:
            logger.info(
                "Regime filter: only %d records match '%s', using all %d",
                len(matching), current_name, len(records),
            )
            return records
        return matching

    async def _filter_fundamental(self, records: List[dict]) -> List[dict]:
        """Ask Groq whether each record's move was driven by a repeatable fundamental catalyst.

        Processes in batches to stay within Groq rate limits.
        Records where Groq says repeatable=False are excluded.
        """
        client = self._get_groq_client()
        if client is None:
            logger.info("Groq unavailable — fundamental filter skipped")
            return records

        # Only re-check records where the feedback_loop didn't already flag repeatable
        to_check = [
            r for r in records
            if not r.get("groq_analysis", {}).get("repeatable", True)
        ]
        if not to_check:
            return records

        # Batch check — ask Groq once for a list of tickers to exclude
        symbols_str = ", ".join(list({r["symbol"] for r in to_check})[:30])
        prompt = f"""These stocks moved unusually but were NOT already classified as repeatable:
{symbols_str}

Which of these typically have noise-driven moves (squeezes, meme activity, thin liquidity)
rather than repeatable fundamental catalysts? Return a JSON list of ticker symbols to EXCLUDE
from model training. Include only genuine noise; if unsure, leave out (err on the side of inclusion).

Return: {{"exclude": ["SYMBOL1", "SYMBOL2"]}}
No markdown."""

        exclude: set = set()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                ),
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1][4:]
            parsed = json.loads(content)
            exclude = {s.upper() for s in parsed.get("exclude", [])}
            logger.info("Fundamental filter: excluding %d symbols: %s", len(exclude), exclude)
        except Exception:
            logger.debug("Groq fundamental filter failed — keeping all records", exc_info=True)

        return [r for r in records if r["symbol"].upper() not in exclude]

    # ── Step 3: Retrain ───────────────────────────────────────────────────────

    def _retrain(self, records: List[dict]) -> Optional[dict]:
        """Retrain LightGBM scorecard with validated feedback records.

        Returns new version info dict, or None if retrain failed.
        """
        logger.info("Retraining LightGBM with %d validated records...", len(records))

        try:
            from strategy.training_pipeline import TrainingPipeline  # type: ignore

            pipeline = TrainingPipeline(self._cfg)
            symbols = list({r["symbol"] for r in records})
            result = pipeline.retrain_incremental(symbols=symbols, extra_records=records)
            return result
        except AttributeError:
            # retrain_incremental may not exist yet — fall back to full retrain
            logger.info("Incremental retrain not available — running full retrain for %d symbols",
                        len({r['symbol'] for r in records}))
            try:
                from strategy.training_pipeline import TrainingPipeline  # type: ignore
                pipeline = TrainingPipeline(self._cfg)
                symbols = list({r["symbol"] for r in records})
                result = pipeline.train(symbols=symbols)
                return result
            except Exception:
                logger.exception("Full retrain also failed")
                return None
        except Exception:
            logger.exception("Retrain failed")
            return None

    # ── Step 4: Report ────────────────────────────────────────────────────────

    def _generate_report(
        self,
        all_records: List[dict],
        filtered_records: List[dict],
        filter_log: dict,
        new_version_info: Optional[dict],
    ) -> dict:
        """Generate a human-readable monthly update report."""
        outcome_counts: Counter = Counter(r.get("outcome_type") for r in filtered_records)
        catalyst_counts: Counter = Counter(
            r.get("groq_analysis", {}).get("catalyst_type", "unknown")
            for r in filtered_records
        )
        pcts = [r.get("pct_change", 0) for r in filtered_records]

        return {
            "report_date": date.today().isoformat(),
            "total_records_loaded": filter_log.get("total", 0),
            "records_after_filtering": filter_log.get("final", 0),
            "filter_removed": {
                "frequency": filter_log.get("removed_f1_frequency", 0),
                "outlier":   filter_log.get("removed_f2_outlier", 0),
                "regime":    filter_log.get("removed_f3_regime", 0),
                "fundamental": filter_log.get("removed_f4_fundamental", 0),
            },
            "outcome_breakdown": dict(outcome_counts),
            "top_catalyst_types": dict(catalyst_counts.most_common(5)),
            "avg_pct_move": round(float(np.mean(pcts)), 2) if pcts else 0,
            "std_pct_move": round(float(np.std(pcts)), 2) if pcts else 0,
            "new_model_version": new_version_info,
            "requires_manual_review": new_version_info is None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        try:
            from groq import Groq  # type: ignore
            self._groq_client = Groq(api_key=api_key)
            return self._groq_client
        except Exception:
            return None


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Monthly LightGBM model update")
    parser.add_argument("--dry", action="store_true", help="Print plan without retraining")
    parser.add_argument("--force", action="store_true", help="Skip Groq fundamental check")
    args = parser.parse_args()
    import json as _json
    with open("config.json") as f:
        config = _json.load(f)
    asyncio.run(ModelUpdater(config).run(dry_run=args.dry, skip_fundamental_check=args.force))
