"""
model_health_reporter.py — Weekly Sunday 8 AM Model Health Digest

Queries trade_lifecycle.db for the last 30 days of closed trades and model
eval records, then sends a Telegram digest summarising model performance.

Scheduled by the main loop (e.g. APScheduler or asyncio-based cron):
    from data.model_health_reporter import ModelHealthReporter
    reporter = ModelHealthReporter()
    reporter.send_weekly_report()   # called every Sunday 08:00 AM

Or triggered on-demand via the Telegram bot /report command:
    msg = reporter.build_report()
    send_telegram_message(msg)
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH        = Path("logs/trade_lifecycle.db")
MODEL_META     = Path("models/scorecard_meta.json")


class ModelHealthReporter:

    def build_report(self, days: int = 30) -> str:
        """Build the weekly health report as an HTML Telegram message."""
        meta    = self._load_meta()
        trades  = self._load_recent_outcomes(days)
        evals   = self._load_model_evals(days)

        # ── Model IC table ───────────────────────────────────────────────────
        model_lines = []
        for m in ["entry", "exit", "dip", "tranche"]:
            ic      = meta.get(f"{m}_ic")
            brier   = meta.get(f"{m}_brier")
            psi     = meta.get(f"{m}_psi")
            if ic is None:
                model_lines.append(f"{m:8s}: no data yet")
                continue
            ic_icon = "✅" if ic > 0 else "❌"
            model_lines.append(
                f"{m:8s}: IC={ic:.3f}{ic_icon}  Brier={brier:.3f}  PSI={psi:.3f}"
            )

        # ── Rollbacks ────────────────────────────────────────────────────────
        rollbacks = [e for e in evals if e.get("result") == "FAIL"]

        # ── Closed trade stats ───────────────────────────────────────────────
        n_closed = len(trades)
        wins     = [t for t in trades if (t.get("alpha") or 0) > 0]
        win_rate = len(wins) / max(n_closed, 1)
        avg_alpha = sum(t.get("alpha", 0) or 0 for t in trades) / max(n_closed, 1)

        # Regime breakdown (group by regime)
        regime_groups: Dict[str, list] = {}
        for t in trades:
            r = t.get("regime", "UNKNOWN")
            regime_groups.setdefault(r, []).append(t.get("alpha", 0) or 0)
        regime_lines = []
        for reg, alphas in sorted(regime_groups.items()):
            avg = sum(alphas) / max(len(alphas), 1)
            regime_lines.append(f"  {reg:12s}: {len(alphas)} trades | avg alpha={avg:+.2%}")

        # ── Calibration summary ───────────────────────────────────────────────
        all_cal_ok = all(
            abs(meta.get(f"{m}_brier", 0.25) - 0.25) < 0.05
            for m in ["entry", "exit", "dip", "tranche"]
            if meta.get(f"{m}_brier") is not None
        )

        psi_vals = [meta.get(f"{m}_psi", 0) for m in ["entry", "exit", "dip", "tranche"]
                    if meta.get(f"{m}_psi") is not None]
        avg_psi = sum(psi_vals) / max(len(psi_vals), 1)

        now = datetime.now().strftime("%Y-%m-%d")
        msg = (
            f"<b>WEEKLY MODEL HEALTH — {now}</b>\n\n"
            + "\n".join(model_lines) + "\n\n"
            + f"Calibration:  {'all OK ✅' if all_cal_ok else '⚠️ check Brier'}\n"
            f"Rollbacks:    {len(rollbacks)} this {days}d {'✅' if not rollbacks else '❌'}\n"
            f"Distribution: PSI={avg_psi:.3f} {'(stable ✅)' if avg_psi < 0.10 else '(⚠️ watch)' if avg_psi < 0.25 else '(🔴 shift!)'}\n\n"
            f"<b>Closed trades ({days}d): {n_closed}</b>\n"
            f"Win rate: {win_rate:.1%} | Avg alpha: {avg_alpha:+.2%}\n\n"
            f"<b>Regime breakdown:</b>\n" + ("\n".join(regime_lines) if regime_lines else "  (no data)")
        )
        return msg

    def send_weekly_report(self) -> bool:
        """Build the report and send it via Telegram."""
        try:
            from data.notifications.telegram import notify
            msg = self.build_report()
            return notify(msg, level="SYSTEM")
        except Exception as e:
            logger.error("Weekly report send failed: %s", e)
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        try:
            return json.loads(MODEL_META.read_text())
        except Exception:
            return {}

    def _load_recent_outcomes(self, days: int) -> List[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_logs WHERE stage='OUTCOME' AND ts >= ? ORDER BY id DESC",
                (cutoff,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                model_out = json.loads(d.get("model_out") or "{}")
                d["alpha"]  = model_out.get("alpha")
                d["regime"] = model_out.get("regime")
                results.append(d)
            return results
        except Exception:
            return []

    def _load_model_evals(self, days: int) -> List[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_logs WHERE stage='MODEL_EVAL' AND ts >= ? ORDER BY id DESC",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
