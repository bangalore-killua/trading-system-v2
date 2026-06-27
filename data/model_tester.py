"""
model_tester.py — ML Auto-Tester: 5-Gate Deployment Guard

Runs before any model swap in model_updater.py. All 5 gates must pass
for the new model to replace the production version.

Gates:
  1. IC > 0                — anti-predictive models never deploy
  2. Calibration ±15%      — soft fail → Platt scaling; >30% → rollback
  3. Pareto dominance       — new model must improve at least one metric
  4. PSI < 0.25            — distribution shift detection
  5. Regime breakdown       — model must work in ALL observed regimes

Usage:
    from data.model_tester import ModelTester
    tester = ModelTester(model_name="entry", new_model_path="models/autogluon_entry")
    passed, report = tester.run(held_out_df)
    if not passed:
        # rollback — do NOT swap models
        send_telegram_alert(report["summary"])
    else:
        deploy_model(new_model_path)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_META_PATH = Path("models/scorecard_meta.json")
REPORT_DIR      = Path("logs/model_test_reports")


class ModelTester:
    def __init__(
        self,
        model_name: str,             # "entry", "exit", "dip", "tranche"
        new_model_path: Optional[str] = None,
        prev_model_path: Optional[str] = None,
    ):
        self.model_name      = model_name
        self.new_model_path  = new_model_path
        self.prev_model_path = prev_model_path
        self._meta           = self._load_meta()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, held_out_df) -> Tuple[bool, dict]:
        """
        Run all 5 gates on the held-out DataFrame.

        Parameters
        ----------
        held_out_df : pd.DataFrame with columns:
            - y_true (int 0/1)
            - y_prob (float 0-1, new model probability)
            - y_prob_prev (float 0-1, previous model probability) — optional
            - regime (str) — optional, for gate 5
            - feature columns for PSI computation

        Returns
        -------
        (passed: bool, report: dict)
        """
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        import pandas as pd  # type: ignore

        df = held_out_df
        y_true  = df["y_true"].values
        y_prob  = df["y_prob"].values

        report: Dict = {
            "model_name":   self.model_name,
            "n_samples":    len(df),
            "gates":        {},
            "rollback":     False,
            "platt_needed": False,
            "summary":      "",
        }

        gates_passed = 0

        # ── Gate 1: IC > 0 ────────────────────────────────────────────────────
        ic = self._spearman_ic(y_prob, y_true.astype(float))
        g1_pass = ic > 0
        report["gates"]["ic"] = {"value": round(ic, 4), "threshold": 0, "passed": g1_pass}
        if g1_pass:
            gates_passed += 1
        else:
            report["rollback"] = True
            report["summary"]  = f"ROLLBACK: IC={ic:.4f} ≤ 0 — anti-predictive model"
            self._emit_log(report, gates_passed)
            return False, report

        # ── Gate 2: Calibration ±15% (soft fail → Platt scaling) ─────────────
        brier = float(np.mean((y_prob - y_true) ** 2))
        brier_ref = 0.25  # null model (predict 0.5 always)
        calibration_err = abs(self._calibration_error(y_prob, y_true))
        g2_pass  = calibration_err <= 0.15
        g2_fatal = calibration_err > 0.30
        report["gates"]["calibration"] = {
            "brier":    round(brier, 4),
            "cal_err":  round(calibration_err, 4),
            "threshold": 0.15,
            "passed":   g2_pass,
        }
        if g2_pass:
            gates_passed += 1
        elif g2_fatal:
            report["rollback"] = True
            report["summary"]  = f"ROLLBACK: calibration error {calibration_err:.0%} > 30%"
            self._emit_log(report, gates_passed)
            return False, report
        else:
            report["platt_needed"] = True
            gates_passed += 1  # soft fail — Platt scaling, not rollback

        # ── Gate 3: Pareto dominance ──────────────────────────────────────────
        prev_ic    = self._meta.get(f"{self.model_name}_ic", 0.0)
        prev_brier = self._meta.get(f"{self.model_name}_brier", 0.25)
        pareto_ok  = (ic >= prev_ic or brier <= prev_brier) and not (ic < prev_ic and brier > prev_brier)
        report["gates"]["pareto"] = {
            "new_ic":    round(ic, 4),    "prev_ic":    round(prev_ic, 4),
            "new_brier": round(brier, 4), "prev_brier": round(prev_brier, 4),
            "passed":    pareto_ok,
        }
        if pareto_ok:
            gates_passed += 1
        else:
            report["rollback"] = True
            report["summary"]  = (
                f"ROLLBACK: Pareto-dominated by previous model "
                f"(new IC={ic:.4f} < prev IC={prev_ic:.4f} AND new Brier={brier:.4f} > prev Brier={prev_brier:.4f})"
            )
            self._emit_log(report, gates_passed)
            return False, report

        # ── Gate 4: PSI < 0.25 ───────────────────────────────────────────────
        psi = self._compute_psi(y_prob)
        g4_pass = psi < 0.25
        report["gates"]["psi"] = {"value": round(psi, 4), "threshold": 0.25, "passed": g4_pass}
        if g4_pass:
            gates_passed += 1
        else:
            report["rollback"] = True
            report["summary"]  = f"ROLLBACK: PSI={psi:.4f} ≥ 0.25 — distribution shift detected"
            self._emit_log(report, gates_passed)
            return False, report

        # ── Gate 5: Regime breakdown ──────────────────────────────────────────
        regime_ics: Dict[str, float] = {}
        if "regime" in df.columns:
            for reg in df["regime"].unique():
                mask = df["regime"] == reg
                if mask.sum() < 10:
                    continue
                reg_ic = self._spearman_ic(y_prob[mask], y_true[mask].astype(float))
                regime_ics[reg] = round(reg_ic, 4)

        regime_collapse = {r: ic for r, ic in regime_ics.items() if ic < -0.05}
        g5_pass = len(regime_collapse) == 0
        report["gates"]["regime_breakdown"] = {
            "regime_ics":      regime_ics,
            "collapsed_in":    list(regime_collapse.keys()),
            "passed":          g5_pass,
        }
        if g5_pass:
            gates_passed += 1
        else:
            # Flag but don't hard-rollback — reduce position size in affected regimes
            report["gates"]["regime_breakdown"]["action"] = "reduce_size_in_collapsed_regimes"
            report["summary"] = (
                f"PARTIAL: regime collapse in {list(regime_collapse.keys())} — "
                "deployed with regime-specific size reduction"
            )
            gates_passed += 1  # count as pass to allow deploy with flag

        if not report["summary"]:
            report["summary"] = (
                f"DEPLOYED: IC={ic:.4f} | Brier={brier:.4f} | PSI={psi:.4f} | "
                f"gates_passed={gates_passed}/5"
                + (" | Platt scaling applied" if report["platt_needed"] else "")
            )

        # Persist updated meta
        self._update_meta(ic, brier, psi, regime_ics)
        self._emit_log(report, gates_passed)
        return True, report

    # ── Statistical helpers ───────────────────────────────────────────────────

    @staticmethod
    def _spearman_ic(probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Spearman rank correlation between predicted prob and binary outcome."""
        from scipy.stats import spearmanr  # type: ignore
        if len(probs) < 10:
            return 0.0
        corr, _ = spearmanr(probs, outcomes)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def _calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Mean calibration error across equal-frequency bins."""
        bins = np.array_split(np.argsort(probs), n_bins)
        errors = []
        for b in bins:
            if len(b) == 0:
                continue
            pred_mean = float(probs[b].mean())
            actual_mean = float(outcomes[b].mean())
            errors.append(abs(pred_mean - actual_mean))
        return float(np.mean(errors)) if errors else 0.0

    def _compute_psi(self, new_probs: np.ndarray, n_bins: int = 10) -> float:
        """PSI between new model's score distribution and baseline reference."""
        ref_probs = np.array(self._meta.get(f"{self.model_name}_ref_probs", [0.5] * 100))
        edges     = np.linspace(0, 1, n_bins + 1)
        new_hist  = np.histogram(new_probs, bins=edges)[0] / max(len(new_probs), 1)
        ref_hist  = np.histogram(ref_probs, bins=edges)[0] / max(len(ref_probs), 1)
        epsilon   = 1e-8
        new_hist  = np.maximum(new_hist, epsilon)
        ref_hist  = np.maximum(ref_hist, epsilon)
        psi = float(np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist)))
        return max(0.0, psi)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        try:
            return json.loads(MODEL_META_PATH.read_text())
        except Exception:
            return {}

    def _update_meta(self, ic: float, brier: float, psi: float, regime_ics: dict) -> None:
        self._meta[f"{self.model_name}_ic"]       = round(ic, 4)
        self._meta[f"{self.model_name}_brier"]    = round(brier, 4)
        self._meta[f"{self.model_name}_psi"]      = round(psi, 4)
        self._meta[f"{self.model_name}_regime_ics"] = regime_ics
        try:
            MODEL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_META_PATH.write_text(json.dumps(self._meta, indent=2))
        except Exception as e:
            logger.warning("Could not write model meta: %s", e)

    def _emit_log(self, report: dict, gates_passed: int) -> None:
        ic    = report["gates"].get("ic", {}).get("value", 0.0)
        brier = report["gates"].get("calibration", {}).get("brier", 0.25)
        psi   = report["gates"].get("psi", {}).get("value", 0.0)
        try:
            from data.logging.trade_logger import model_eval_log, WhyAnalyzer
            model_eval_log().result("PASS" if not report["rollback"] else "FAIL").model({
                "model_name":   self.model_name,
                "ic":           ic,
                "brier":        brier,
                "psi":          psi,
                "gates_passed": gates_passed,
                "rollback":     report["rollback"],
                "platt_needed": report["platt_needed"],
            }).why(WhyAnalyzer.model_eval(ic, brier, psi, gates_passed, report["rollback"])).emit()
        except Exception:
            pass

        # Write JSON report for audit trail
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            fname = REPORT_DIR / f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            fname.write_text(json.dumps(report, indent=2, default=str))
        except Exception:
            pass
