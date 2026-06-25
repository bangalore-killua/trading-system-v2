"""
dip_detector.py — AutoGluon dip detection model.

Question: "Is this price drop a buying opportunity or a falling knife?"

Strategy from plan:
  Dip tranches are LARGER (0.8×–1.5×) than initial entry tranches.
  We buy more on fundamental dips, not less.
  Signal: dip_score ≥ 0.65 + fundamentals intact + regime bullish.

Features (for AutoGluon model):
  drop_pct_from_high:     how far from recent high (e.g. -0.15 = 15% drop)
  drop_speed_days:        how many days to reach current low
  regime:                 current regime string (EARLY_BULL, MID_BULL, etc.)
  finbert_delta:          news sentiment change since peak
  news_cause:             "earnings_miss" | "sector_rotation" | "macro" | "stock_specific"
  eps_beat_streak:        consecutive earnings beats (quality proxy)
  institutional_change:   change in institutional ownership (13F delta)
  insider_post_drop:      insider buy within 2 weeks of drop (high IC)
  quality_score:          composite fundamental quality (ROE, FCF, etc.)
  pre_drop_rsi:           RSI just before the drop started
  options_flow_score:     call/put ratio during dip (smart money positioning)
  macro_score:            yield curve + VIX composite (is macro supportive?)

Model path: models/autogluon_dip/
Built via: strategy/training_pipeline.py AutoGluonPipeline — NOT yet implemented
           (requires entry model IC > 0 + enough dip labeled records ≥ 500)

Fallback (before model is trained):
  Rule-based: drop > 8% + quality_score > 0.6 + regime not CRASH → score 0.65
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DIP_MODEL_PATH    = "models/autogluon_dip"
DIP_SCORE_THRESHOLD = 0.65  # minimum dip_score to trigger dip_verdict pipeline


class DipDetector:
    """AutoGluon-based dip quality classifier.

    Usage:
        detector = DipDetector()
        result   = detector.score("NVDA", dip_features)
        if result["dip_score"] >= 0.65:
            verdict = await consensus_engine.dip_verdict("NVDA", dip_features)
    """

    def __init__(self, model_path: str = DIP_MODEL_PATH):
        self._model_path = model_path
        self._model      = None

    def score(self, symbol: str, features: dict) -> dict:
        """Score a dip to determine if it's a buying opportunity.

        Parameters
        ----------
        features : {drop_pct_from_high, drop_speed_days, regime, finbert_delta,
                    news_cause, eps_beat_streak, institutional_change, insider_post_drop,
                    quality_score, pre_drop_rsi, options_flow_score, macro_score}

        Returns
        -------
        {
          "symbol":            str,
          "dip_score":         float 0-1,
          "is_opportunity":    bool,
          "recommended_size":  str ("skip"|"normal"|"large"),
          "model_source":      str,
        }
        """
        if self._is_model_available():
            try:
                dip_score = self._infer_autogluon(features)
                return self._build_result(symbol, dip_score, "autogluon")
            except Exception as e:
                logger.warning("DipDetector: AutoGluon inference failed for %s — %s. Using fallback.", symbol, e)

        dip_score = self._fallback_score(features)
        return self._build_result(symbol, dip_score, "rule_based")

    def _build_result(self, symbol: str, dip_score: float, source: str) -> dict:
        dip_score = round(float(dip_score), 4)
        is_opportunity = dip_score >= DIP_SCORE_THRESHOLD

        if dip_score >= 0.80:
            recommended_size = "large"   # 1.2×-1.5× T1 size
        elif dip_score >= DIP_SCORE_THRESHOLD:
            recommended_size = "normal"  # 0.8×-1.0× T1 size
        else:
            recommended_size = "skip"

        result = {
            "symbol":           symbol,
            "dip_score":        dip_score,
            "is_opportunity":   is_opportunity,
            "recommended_size": recommended_size,
            "model_source":     source,
        }
        logger.info("DipDetector %s: score=%.3f → %s [%s]", symbol, dip_score, recommended_size, source)
        return result

    def _is_model_available(self) -> bool:
        return os.path.isdir(self._model_path)

    def _infer_autogluon(self, features: dict) -> float:
        import pandas as pd
        from autogluon.tabular import TabularPredictor  # type: ignore

        if self._model is None:
            self._model = TabularPredictor.load(self._model_path)

        df   = pd.DataFrame([features])
        prob = self._model.predict_proba(df).iloc[0].get(1, 0.5)
        return float(prob)

    def _fallback_score(self, features: dict) -> float:
        """Rule-based fallback before model is trained."""
        drop        = abs(features.get("drop_pct_from_high", 0))
        quality     = features.get("quality_score", 0.5)
        regime      = features.get("regime", "NEUTRAL")
        insider     = features.get("insider_post_drop", False)
        macro       = features.get("macro_score", 0.5)
        eps_streak  = features.get("eps_beat_streak", 0)
        options_sc  = features.get("options_flow_score", 0.5)

        if regime in ("CRASH", "RISK_OFF"):
            return 0.2

        score = 0.3
        if drop >= 0.08:       score += 0.15  # meaningful drop
        if drop >= 0.15:       score += 0.10  # deeper dip
        if quality >= 0.6:     score += 0.15  # quality company
        if quality >= 0.75:    score += 0.05
        if insider:            score += 0.10  # insider buying = high IC signal
        if macro >= 0.55:      score += 0.05
        if eps_streak >= 3:    score += 0.05
        if options_sc >= 0.60: score += 0.05  # calls > puts during dip

        return round(min(score, 0.80), 3)  # cap fallback at 0.80
