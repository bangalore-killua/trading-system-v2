"""
signal_orchestrator.py — Composite scoring → Claude Sonnet gate → hard constraints.

Layer architecture (from plan):
  Layer 1: Specialized models in parallel
            LightGBM/AutoGluon (entry), LSTM (exit), AutoGluon (dip), LightGBM (tranche)
  Layer 2: Meta-stacking (AutoGluon on conflict cases — builds after 2000+ conflicts)
  Layer 3: Research Agent + Claude Sonnet (anomaly detection + final synthesis)
  Layer 4: Hard constraints (non-negotiable, non-ML)

Composite score formula:
  composite_score = Σ(signal_i × IC[signal_i][current_regime])
  position_size   = half_kelly × composite_score_normalised × regime_scalar

IC tracking (rolling 90-day, per signal per regime):
  After each closed trade: IC[signal][regime] = spearman(confidence_scores, alpha_outcomes)
  Stored in models/scorecard_meta.json

Signal registry (all contribute score, none individually block):
  technical:    RSI, MACD, Bollinger, Volume, ATR
  pattern:      algebraic patterns (cup, flag, H&S, etc.)
  news:         FinBERT + headline velocity
  fundamental:  EPS streak, FCF, institutional 13F
  alternative:  Congressional + Insider Form 4  ← highest IC empirically
  options:      Unusual Whales call/put ratio  ← leading indicator
  macro:        FRED yield curve, credit spread, VIX
  knowledge:    shared holders, corporate investments

Hard gates (Layer 4):
  ATR stop hit            → SELL_ALL
  Cash reserve < 10%      → no new buys
  Symbol exposure > 22%   → no adds
  Regime = CRASH          → no new entries
  VIX > 40                → all sizes × 0.5
  Portfolio drawdown > 12% → pause all entries
  5 consecutive losses    → pause + human alert
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SCORECARD_PATH   = Path("models/scorecard_meta.json")
SONNET_MODEL     = "claude-sonnet-4-6"
IC_WINDOW_DAYS   = 90
COMPOSITE_THRESHOLD = 0.35  # minimum composite for T1 entry


class SignalOrchestrator:
    """Composite signal scoring with IC weighting and Claude Sonnet synthesis.

    Usage:
        orch = SignalOrchestrator(portfolio_state)
        result = await orch.evaluate("NVDA", signal_inputs, regime="MID_BULL")

        result["composite_score"]  → 0-1 overall signal strength
        result["position_size_pct"]→ position size as % of portfolio
        result["action"]           → "enter_t1|skip|wait"
        result["hard_gate_reason"] → None or reason why a hard gate fired
    """

    def __init__(self, portfolio_state: Optional[dict] = None, config: Optional[dict] = None):
        self._portfolio = portfolio_state or {}
        self._cfg       = config or {}
        self._scorecard = self._load_scorecard()
        self._anthropic_client = None
        self._consecutive_losses = self._portfolio.get("consecutive_losses", 0)

    async def evaluate(
        self,
        symbol:          str,
        signal_inputs:   dict,
        regime:          str = "NEUTRAL",
        dry_run:         bool = False,
    ) -> dict:
        """Full orchestrator pipeline: scores → composite → Sonnet gate → hard constraints.

        Parameters
        ----------
        signal_inputs : {
          "rsi_14", "macd", "vol_ratio", "momentum", "pattern_score",
          "news_score", "fundamental_score", "insider_score", "options_score",
          "macro_score", "4h_trend", "1d_regime_slope",
          "entry_model_prob",       # from AutoGluon/LightGBM
          "headlines",              # list of headline strings
          "company_name",           # for LLM prompt
          "beta",                   # stock beta vs SPY
        }

        Returns
        -------
        {
          "symbol":           str,
          "composite_score":  float 0-1,
          "signal_scores":    dict,
          "position_size_pct": float,
          "action":           "enter_t1"|"skip"|"pause",
          "hard_gate_reason": str|None,
          "sonnet_anomaly":   str|None,
          "regime":           str,
          "vix_scaling":      float,
        }
        """
        trace_id = signal_inputs.get("_trace_id")
        cycle_id = signal_inputs.get("_cycle_id")

        try:
            from data.logging.trade_logger import stage as tl_stage, new_trace, WhyAnalyzer
            _tl = True
        except Exception:
            _tl = False

        # ── Layer 4 first: hard constraints (always check before computing) ───
        hard_gate = self._check_hard_gates(symbol, regime)
        if hard_gate:
            if _tl and trace_id:
                tl_stage(trace_id, symbol, "CONSTRAINTS", cycle_id).result("FAIL").reason(hard_gate).why(
                    WhyAnalyzer.constraints(False, hard_gate,
                        self._portfolio.get("cash_pct", 0.5),
                        self._portfolio.get("exposure_pct", 0))
                ).emit()
            return self._gate_result(symbol, regime, hard_gate)

        # ── Layer 1: IC-weighted composite score ──────────────────────────────
        signal_scores  = self._compute_signal_scores(signal_inputs, regime)
        composite      = self._ic_weighted_composite(signal_scores, regime)

        if _tl:
            if not trace_id:
                trace_id, _log = new_trace(symbol, "SIGNAL", cycle_id)
                _log.result("PASS").signals(signal_scores).model({"composite": composite}).why(
                    WhyAnalyzer.signal(composite, signal_scores, cycle_id or "")
                ).emit()
            else:
                tl_stage(trace_id, symbol, "SIGNAL", cycle_id).result("PASS").signals(signal_scores).model(
                    {"composite": composite}
                ).why(WhyAnalyzer.signal(composite, signal_scores, cycle_id or "")).emit()

        # ── VIX scaling ───────────────────────────────────────────────────────
        vix        = self._portfolio.get("vix", 20)
        vix_scale  = 0.5 if vix > 40 else 1.0

        # ── Half-Kelly position sizing ────────────────────────────────────────
        size_pct = self._compute_position_size(
            composite  = composite,
            regime     = regime,
            beta       = signal_inputs.get("beta", 1.0),
            vix_scale  = vix_scale,
        )

        # ── MODELS log (model probabilities from signal_inputs) ───────────────
        if _tl and trace_id:
            entry_p   = float(signal_inputs.get("entry_model_prob", 0.5))
            tl_stage(trace_id, symbol, "MODELS", cycle_id).result("PASS").model({
                "entry_P":   entry_p,
                "composite": composite,
            }).why(WhyAnalyzer.models(entry_p, 0.5, None, None)).emit()

        # ── CONSTRAINTS log (passed — would have returned early otherwise) ────
        if _tl and trace_id:
            tl_stage(trace_id, symbol, "CONSTRAINTS", cycle_id).result("PASS").why(
                WhyAnalyzer.constraints(True, None,
                    self._portfolio.get("cash_pct", 0.5),
                    self._portfolio.get("exposure_pct", 0))
            ).emit()

        # ── Layer 3: Claude Sonnet anomaly check (only for high-composite signals) ──
        sonnet_anomaly = None
        if composite >= 0.55 and not dry_run:
            sonnet_anomaly = await self._sonnet_anomaly_check(symbol, signal_inputs, composite, regime)

        action = "enter_t1" if composite >= COMPOSITE_THRESHOLD else "skip"

        # ── ENTRY log ─────────────────────────────────────────────────────────
        if _tl and trace_id:
            rsi = signal_inputs.get("rsi_14")
            tl_stage(trace_id, symbol, "ENTRY", cycle_id).result("PASS" if action != "skip" else "SKIP").model({
                "action": action, "composite": composite, "size_pct": round(size_pct * 100, 2), "vix_scale": vix_scale,
            }).decision(action).why(
                WhyAnalyzer.entry(action, composite, size_pct * 100, rsi, vix_scale)
            ).emit()

        return {
            "symbol":           symbol,
            "composite_score":  round(composite, 4),
            "signal_scores":    signal_scores,
            "position_size_pct": round(size_pct * 100, 2),
            "action":           action,
            "hard_gate_reason": None,
            "sonnet_anomaly":   sonnet_anomaly,
            "regime":           regime,
            "vix_scaling":      vix_scale,
            "evaluated_at":     datetime.now().isoformat(),
            "_trace_id":        trace_id,
        }

    # ── Signal scoring ────────────────────────────────────────────────────────

    def _compute_signal_scores(self, inputs: dict, regime: str) -> dict:
        """Map raw signal inputs to normalized 0-1 scores."""
        scores = {}

        # Technical: RSI (oversold = higher score)
        rsi = inputs.get("rsi_14")
        if rsi is not None:
            scores["technical"] = round(max(0, min(1, (70 - rsi) / 40)), 3)

        # Pattern
        if "pattern_score" in inputs:
            scores["pattern"] = float(inputs["pattern_score"])

        # News sentiment
        if "news_score" in inputs:
            scores["news"] = float(inputs["news_score"])

        # Fundamental quality
        if "fundamental_score" in inputs:
            scores["fundamental"] = float(inputs["fundamental_score"])

        # Alternative: v14 composite (congressional + insider + dark pool); fall back to v13 insider_score
        if "alternative_score" in inputs:
            scores["alternative"] = float(inputs["alternative_score"])
        elif "insider_score" in inputs:
            scores["alternative"] = float(inputs["insider_score"])

        # Options flow (Unusual Whales)
        if "options_score" in inputs:
            scores["options"] = float(inputs["options_score"])

        # Macro (FRED composite)
        if "macro_score" in inputs:
            scores["macro"] = float(inputs["macro_score"])

        # ML model output (entry model probability)
        if "entry_model_prob" in inputs:
            scores["ml_entry"] = float(inputs["entry_model_prob"])

        # 4H trend gate (Auto-Quant empirical finding — flips winter Sharpe)
        scores["4h_trend"] = float(inputs.get("4h_trend", 0))

        # 1D regime slope
        slope = inputs.get("1d_regime_slope", 0)
        scores["regime_slope"] = max(0, float(slope))  # only bullish slope adds score

        return scores

    def _ic_weighted_composite(self, signal_scores: dict, regime: str) -> float:
        """IC-weighted composite: Σ(signal_i × IC[signal_i][regime])."""
        ic_weights = self._scorecard.get("ic_weights", {}).get(regime, {})

        # Default IC weights (equal weighting if no empirical data yet)
        default_ic = {
            "technical":    0.12,
            "pattern":      0.08,
            "news":         0.12,
            "fundamental":  0.10,
            "alternative":  0.18,  # congressional + insider — highest IC
            "options":      0.15,  # leading indicator
            "macro":        0.08,
            "ml_entry":     0.20,  # model output — highest composite weight
            "4h_trend":     0.10,
            "regime_slope": 0.05,
        }

        total_weight  = 0.0
        weighted_sum  = 0.0
        for signal, score in signal_scores.items():
            ic = ic_weights.get(signal, default_ic.get(signal, 0.10))
            weighted_sum  += score * ic
            total_weight  += ic

        if total_weight == 0:
            return 0.5
        return round(min(1.0, weighted_sum / total_weight), 4)

    # ── Position sizing ───────────────────────────────────────────────────────

    def _compute_position_size(self, composite: float, regime: str, beta: float, vix_scale: float) -> float:
        """Half-Kelly sizing: kelly × 0.5 × composite × regime_scalar / beta_adj."""
        # Kelly from config (or default 10%)
        kelly    = self._cfg.get("model", {}).get("kelly_fraction", 0.10)
        half_k   = kelly * 0.5

        # Regime scalar (be more conservative in late bull / crash)
        regime_scalar = {
            "EARLY_BULL": 1.0, "MID_BULL": 0.9, "LATE_BULL": 0.7,
            "RECOVERY":   0.85, "NEUTRAL":  0.8, "SIDEWAYS":  0.75, "CRASH": 0.0,
        }.get(regime, 0.8)

        # Beta adjustment: high-beta stocks get smaller position
        beta_adj = min(max(beta, 0.5), 2.0)

        size = half_k * composite * regime_scalar * vix_scale / beta_adj
        return round(max(0.0, min(size, 0.15)), 4)  # hard cap at 15%

    # ── Hard gates (Layer 4) ──────────────────────────────────────────────────

    def _check_hard_gates(self, symbol: str, regime: str) -> Optional[str]:
        p = self._portfolio

        if regime == "CRASH":
            return "CRASH regime — no new entries"
        if p.get("cash_pct", 1.0) < 0.10:
            return "cash_reserve < 10%"
        if p.get("positions", {}).get(symbol, {}).get("total_pct", 0) >= 0.22:
            return f"combined_exposure[{symbol}] ≥ 22%"
        if p.get("drawdown_pct", 0) > 0.12:
            return "portfolio drawdown > 12% — entries paused"
        if self._consecutive_losses >= 5:
            return "5 consecutive losses — human review required"

        return None

    def _gate_result(self, symbol: str, regime: str, reason: str) -> dict:
        logger.info("Hard gate SKIP %s: %s", symbol, reason)
        return {
            "symbol":           symbol,
            "composite_score":  0.0,
            "signal_scores":    {},
            "position_size_pct": 0.0,
            "action":           "pause" if "drawdown" in reason or "loss" in reason else "skip",
            "hard_gate_reason": reason,
            "sonnet_anomaly":   None,
            "regime":           regime,
            "vix_scaling":      1.0,
            "evaluated_at":     datetime.now().isoformat(),
        }

    # ── Claude Sonnet anomaly check ───────────────────────────────────────────

    async def _sonnet_anomaly_check(
        self, symbol: str, inputs: dict, composite: float, regime: str
    ) -> Optional[str]:
        """Claude Sonnet: anomaly detection — what do the numbers miss?

        Role: synthesis and anomaly detection, NOT fact provider.
        Uses research_agent output (browse=False for speed, Haiku handles details).
        Cost: ~8 calls/day × 3000 tokens ≈ $2/month.
        """
        client = self._get_anthropic_client()
        if client is None:
            return None

        headlines = inputs.get("headlines", [])[:5]
        headline_str = "\n".join(f"- {h}" for h in headlines) if headlines else "(none)"

        system_prompt = (
            "You are a senior quantitative analyst reviewing a trade signal. "
            "Your role: identify qualitative anomalies the numerical model cannot see — "
            "regulatory risk, management change, narrative shift, competitive disruption. "
            "If nothing anomalous, say so. Be concise (1-2 sentences max)."
        )
        user_prompt = (
            f"Symbol: {inputs.get('company_name', symbol)} ({symbol})\n"
            f"Regime: {regime} | Composite: {composite:.3f}\n"
            f"Signal scores: {json.dumps({k: round(v, 3) for k, v in inputs.items() if k.endswith('_score')})}\n\n"
            f"Recent headlines:\n{headline_str}\n\n"
            f"Is there any qualitative anomaly that would change this signal's validity? "
            f"Answer in 1-2 sentences. If nothing anomalous, say 'No anomaly detected.'"
        )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            msg  = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model=SONNET_MODEL,
                    max_tokens=150,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            )
            text = msg.content[0].text.strip()
            if "no anomaly" in text.lower():
                return None
            logger.info("Sonnet anomaly %s: %s", symbol, text[:100])
            return text
        except Exception as e:
            logger.debug("Sonnet anomaly check failed: %s", e)
            return None

    # ── IC update ─────────────────────────────────────────────────────────────

    def update_ic(self, regime: str, signal_name: str, confidence_scores: list, alpha_outcomes: list) -> None:
        """Update IC[signal][regime] after trade closes. Rolling 90-day Spearman."""
        if len(confidence_scores) < 10:
            return

        from scipy.stats import spearmanr  # type: ignore
        ic, _ = spearmanr(confidence_scores, alpha_outcomes)

        if "ic_weights" not in self._scorecard:
            self._scorecard["ic_weights"] = {}
        if regime not in self._scorecard["ic_weights"]:
            self._scorecard["ic_weights"][regime] = {}

        self._scorecard["ic_weights"][regime][signal_name] = round(max(0.001, float(ic)), 4)
        self._save_scorecard()
        logger.info("IC update: [%s][%s] = %.4f", regime, signal_name, ic)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_scorecard(self) -> dict:
        try:
            if SCORECARD_PATH.exists():
                with open(SCORECARD_PATH) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug("Failed to load scorecard: %s", e)
        return {}

    def _save_scorecard(self) -> None:
        try:
            SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SCORECARD_PATH, "w") as f:
                json.dump(self._scorecard, f, indent=2)
        except Exception as e:
            logger.debug("Failed to save scorecard: %s", e)

    def _get_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            import anthropic  # type: ignore
            self._anthropic_client = anthropic.Anthropic(api_key=key)
            return self._anthropic_client
        except Exception as e:
            logger.debug("Anthropic client init failed: %s", e)
            return None
