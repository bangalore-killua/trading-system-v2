"""
fred_macro.py — FRED macro signals as ML feature scores (0-1) for the signal orchestrator.

Wraps data/fincept/fred_data.py with feature engineering:
  - yield_curve_score:  inverted curve → 0 (bearish), steep curve → 1 (bullish)
  - credit_spread_score: wide spread → 0 (risk-off), tight spread → 1 (risk-on)
  - vix_score:          VIX > 40 → 0, VIX < 15 → 1
  - macro_regime:       "RISK_ON" | "RISK_OFF" | "NEUTRAL"
  - vix_above_40:       bool — triggers portfolio-level circuit breaker

These feed directly into the signal orchestrator's macro_score and regime detection.

Uses existing data/fincept/fred_data.py for data fetching (no duplication).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_macro_signals() -> dict:
    """Return macro feature scores (0-1) ready for the signal orchestrator.

    Returns
    -------
    {
      "yield_curve_score":   float 0-1,  # 0=inverted/recession, 1=steep/bullish
      "credit_spread_score": float 0-1,  # 0=wide/risk-off, 1=tight/risk-on
      "vix_score":           float 0-1,  # 0=extreme fear, 1=complacency
      "macro_score":         float 0-1,  # composite of above three
      "macro_regime":        str,        # "RISK_ON" | "RISK_OFF" | "NEUTRAL"
      "vix_above_40":        bool,       # circuit breaker trigger
      "vix_raw":             float,
      "yield_spread_raw":    float,      # 10Y-2Y in % points
      "credit_spread_raw":   float,      # HY-IG in % points
    }
    """
    from data.fincept.fred_data import get_series, get_yield_curve  # type: ignore

    result = {
        "yield_curve_score":   0.5,
        "credit_spread_score": 0.5,
        "vix_score":           0.5,
        "macro_score":         0.5,
        "macro_regime":        "NEUTRAL",
        "vix_above_40":        False,
        "vix_raw":             None,
        "yield_spread_raw":    None,
        "credit_spread_raw":   None,
    }

    # ── VIX ──────────────────────────────────────────────────────────────────
    try:
        vix_series = get_series("VIXCLS", limit=1)
        if vix_series:
            vix = float(vix_series[-1]["value"])
            result["vix_raw"]      = round(vix, 2)
            result["vix_above_40"] = vix > 40
            # Normalize: VIX 10=score 1.0, VIX 40=score 0.0, VIX >40 → 0
            result["vix_score"]    = round(max(0.0, min(1.0, (40 - vix) / 30)), 3)
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)

    # ── Yield curve (10Y-2Y spread) ───────────────────────────────────────────
    try:
        yc = get_yield_curve()
        spread = yc.get("spread_10_2")
        if spread is not None:
            result["yield_spread_raw"] = round(spread, 3)
            # Normalize: spread -1.0 → 0, spread +2.0 → 1.0
            result["yield_curve_score"] = round(max(0.0, min(1.0, (spread + 1.0) / 3.0)), 3)
    except Exception as e:
        logger.warning("Yield curve fetch failed: %s", e)

    # ── Credit spread (HY-IG) ─────────────────────────────────────────────────
    try:
        hy_series = get_series("BAMLH0A0HYM2", limit=1)
        if hy_series:
            credit_spread = float(hy_series[-1]["value"])
            result["credit_spread_raw"] = round(credit_spread, 3)
            # Normalize: spread 10 → 0 (crisis), spread 2 → 1.0 (tight/benign)
            result["credit_spread_score"] = round(max(0.0, min(1.0, (10 - credit_spread) / 8.0)), 3)
    except Exception as e:
        logger.warning("Credit spread fetch failed: %s", e)

    # ── Composite macro score ─────────────────────────────────────────────────
    scores = [result["yield_curve_score"], result["credit_spread_score"], result["vix_score"]]
    result["macro_score"] = round(float(sum(scores) / len(scores)), 3)

    # ── Regime classification ─────────────────────────────────────────────────
    if result["macro_score"] >= 0.65:
        result["macro_regime"] = "RISK_ON"
    elif result["macro_score"] <= 0.35:
        result["macro_regime"] = "RISK_OFF"
    else:
        result["macro_regime"] = "NEUTRAL"

    return result


def get_4h_trend_gate(df_4h) -> int:
    """EMA50 > EMA200 on 4H data → 1 (bullish regime) else 0.

    Auto-Quant empirical finding: this single feature flipped winter Sharpe
    from -0.11 → +0.05 across momentum, MR, and counter-trend paradigms.

    Parameters
    ----------
    df_4h : pd.DataFrame with 'close' column, at least 205 4H bars
    """
    import numpy as np
    closes = df_4h["close"].values
    if len(closes) < 205:
        return 0  # insufficient data → neutral
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    return int(ema50[-1] > ema200[-1])


def get_1d_regime_slope(df_1d) -> int:
    """Sign of EMA200 slope on 1D data: +1 bull, -1 bear, 0 flat.

    Auto-Quant: zero-cost regime classifier from OHLCV alone.

    Parameters
    ----------
    df_1d : pd.DataFrame with 'close' column, at least 220 daily bars
    """
    import numpy as np
    closes = df_1d["close"].values
    if len(closes) < 220:
        return 0
    ema200 = _ema(closes, 200)
    slope = ema200[-1] - ema200[-20]
    if abs(slope / ema200[-1]) < 0.0005:  # < 0.05% → flat
        return 0
    return 1 if slope > 0 else -1


def _ema(values, period: int):
    """Exponential moving average using pandas ewm logic."""
    import numpy as np
    alpha = 2 / (period + 1)
    ema = np.zeros_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    signals = get_macro_signals()
    for k, v in signals.items():
        print(f"  {k}: {v}")
