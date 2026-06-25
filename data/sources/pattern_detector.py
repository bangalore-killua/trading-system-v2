"""
pattern_detector.py — Algebraic chart pattern detection. Replaces YOLOv8 entirely.

Why algebraic instead of ML:
  - YOLOv8 required: numbers → images → object detection → signal (number)
  - Algebraic: numbers → signal directly. No training data, no GPU, no image pipeline.
  - Deterministic, microseconds not milliseconds, fully transparent.
  - Same patterns TradingView computes internally.

All functions:
  - Accept a pandas DataFrame with columns: open, high, low, close, volume
  - Return a dict with {"detected": bool, "confidence": float 0-1, "details": dict}
  - Require at least 60 bars of history for reliable detection

CLI:
    python -m data.sources.pattern_detector --symbol NVDA
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PatternResult = dict  # {"detected": bool, "confidence": float, "details": dict}


# ── Individual pattern detectors ──────────────────────────────────────────────

def detect_cup_handle(df: pd.DataFrame, min_bars: int = 60) -> PatternResult:
    """Cup and Handle: U-shaped consolidation followed by brief pullback before breakout.

    Requirements: depth 15-35% of peak, recovery ≥85%, handle <50% of cup depth.
    """
    if len(df) < min_bars:
        return _no_detect("cup_handle", "insufficient bars")

    highs = df["high"].values
    lows  = df["low"].values

    # Find left peak, trough, right peak in trailing 60 bars
    window = highs[-60:]
    peak1_idx  = np.argmax(window[:30])
    peak1      = window[peak1_idx]
    trough_idx = np.argmin(window[peak1_idx:45]) + peak1_idx
    trough     = window[trough_idx]
    peak2      = np.max(window[trough_idx:])

    cup_depth  = (peak1 - trough) / peak1 if peak1 > 0 else 0
    recovery   = (peak2 - trough) / (peak1 - trough) if (peak1 - trough) > 0 else 0

    if not (0.15 < cup_depth < 0.35 and recovery > 0.85):
        return _no_detect("cup_handle", f"depth={cup_depth:.2f} recovery={recovery:.2f}")

    # Handle check: last 10 bars should show shallow pullback < 50% of cup depth
    handle_low = np.min(lows[-10:])
    handle_depth = (peak2 - handle_low) / peak2 if peak2 > 0 else 1.0
    if handle_depth > cup_depth * 0.5:
        return _no_detect("cup_handle", f"handle too deep ({handle_depth:.2f})")

    confidence = min(1.0, recovery * (1 - abs(cup_depth - 0.25) / 0.25))
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"cup_depth": round(cup_depth, 3), "recovery": round(recovery, 3),
                       "handle_depth": round(handle_depth, 3)},
    }


def detect_head_shoulders(df: pd.DataFrame) -> PatternResult:
    """Head and Shoulders: bearish reversal. Three peaks, middle highest.

    Requires left shoulder ≈ right shoulder (within 10%), head 10-30% above shoulders.
    """
    if len(df) < 60:
        return _no_detect("head_shoulders", "insufficient bars")

    highs = df["high"].values[-60:]
    # Find three local maxima
    peaks = _local_maxima(highs, distance=8)
    if len(peaks) < 3:
        return _no_detect("head_shoulders", f"only {len(peaks)} peaks found")

    ls, head, rs = peaks[-3], peaks[-2], peaks[-1]
    ls_h, head_h, rs_h = highs[ls], highs[head], highs[rs]

    symmetry     = 1 - abs(ls_h - rs_h) / max(ls_h, rs_h)
    head_premium = (head_h - max(ls_h, rs_h)) / max(ls_h, rs_h)

    if symmetry < 0.90 or not (0.05 < head_premium < 0.35):
        return _no_detect("head_shoulders",
                          f"symmetry={symmetry:.2f} head_premium={head_premium:.2f}")

    # Neckline: line through troughs between shoulders and head
    neckline = np.mean(df["low"].values[-60:][[ls + (head - ls)//2, head + (rs - head)//2]])
    confidence = symmetry * min(1.0, head_premium / 0.15)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"symmetry": round(symmetry, 3), "head_premium": round(head_premium, 3),
                       "neckline": round(float(neckline), 2)},
    }


def detect_bull_flag(df: pd.DataFrame) -> PatternResult:
    """Bull Flag: strong upward pole followed by tight parallel consolidation.

    Pole: ≥8% gain over 5-15 bars. Flag: ≤5% pullback, low volatility channel.
    """
    if len(df) < 30:
        return _no_detect("bull_flag", "insufficient bars")

    closes = df["close"].values

    # Pole: last 10-20 bars before flag
    pole_start  = closes[-25]
    pole_high   = np.max(closes[-20:-5])
    pole_gain   = (pole_high - pole_start) / pole_start if pole_start > 0 else 0

    # Flag: last 5-10 bars should be tight consolidation
    flag_close  = closes[-1]
    flag_high   = np.max(closes[-10:])
    flag_low    = np.min(closes[-10:])
    flag_range  = (flag_high - flag_low) / flag_high if flag_high > 0 else 1.0
    pullback    = (pole_high - flag_close) / pole_high if pole_high > 0 else 1.0

    if pole_gain < 0.08 or pullback > 0.05 or flag_range > 0.04:
        return _no_detect("bull_flag",
                          f"pole_gain={pole_gain:.2f} pullback={pullback:.2f} range={flag_range:.2f}")

    confidence = min(1.0, pole_gain / 0.15) * (1 - flag_range / 0.04)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"pole_gain": round(pole_gain, 3), "pullback": round(pullback, 3),
                       "flag_range": round(flag_range, 3)},
    }


def detect_double_bottom(df: pd.DataFrame) -> PatternResult:
    """Double Bottom: W-shape reversal — two troughs at approximately equal levels.

    Trough equality within 3%. Recovery between troughs ≥4%.
    """
    if len(df) < 40:
        return _no_detect("double_bottom", "insufficient bars")

    lows = df["low"].values[-40:]
    troughs = _local_minima(lows, distance=8)
    if len(troughs) < 2:
        return _no_detect("double_bottom", f"only {len(troughs)} troughs found")

    t1, t2 = troughs[-2], troughs[-1]
    l1, l2 = lows[t1], lows[t2]

    equality  = 1 - abs(l1 - l2) / max(l1, l2)
    mid_high  = np.max(lows[t1:t2]) if t2 > t1 else l1
    recovery  = (mid_high - min(l1, l2)) / min(l1, l2) if min(l1, l2) > 0 else 0

    if equality < 0.97 or recovery < 0.04:
        return _no_detect("double_bottom", f"equality={equality:.2f} recovery={recovery:.2f}")

    confidence = equality * min(1.0, recovery / 0.08)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"equality": round(equality, 3), "recovery": round(recovery, 3)},
    }


def detect_breakout(df: pd.DataFrame, lookback: int = 52) -> PatternResult:
    """52-week (or lookback-bar) high breakout with volume confirmation.

    Criteria: close > max(high[-lookback:-1]) AND volume > 1.5× avg volume.
    """
    if len(df) < lookback + 5:
        return _no_detect("breakout", "insufficient bars")

    prior_high  = df["high"].iloc[-(lookback + 1):-1].max()
    current_close = float(df["close"].iloc[-1])
    breakout_pct  = (current_close - prior_high) / prior_high if prior_high > 0 else 0

    vol_today = float(df["volume"].iloc[-1])
    vol_avg   = float(df["volume"].iloc[-20:-1].mean())
    vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0

    if breakout_pct <= 0 or vol_ratio < 1.5:
        return _no_detect("breakout", f"breakout_pct={breakout_pct:.3f} vol_ratio={vol_ratio:.2f}")

    confidence = min(1.0, breakout_pct / 0.02) * min(1.0, vol_ratio / 3.0)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"breakout_pct": round(breakout_pct, 4),
                       "vol_ratio": round(vol_ratio, 2),
                       "prior_high": round(prior_high, 2)},
    }


def detect_ascending_triangle(df: pd.DataFrame) -> PatternResult:
    """Ascending Triangle: flat resistance + rising support. Bullish continuation.

    Resistance: 3+ touches within 1.5% range. Support: rising lows.
    """
    if len(df) < 30:
        return _no_detect("ascending_triangle", "insufficient bars")

    highs = df["high"].values[-30:]
    lows  = df["low"].values[-30:]

    resistance = np.max(highs)
    resistance_touches = np.sum(highs > resistance * 0.985)

    # Rising lows: fit linear trend to lows
    x = np.arange(len(lows))
    slope, _ = np.polyfit(x, lows, 1)
    slope_pct = slope / np.mean(lows) if np.mean(lows) > 0 else 0

    if resistance_touches < 3 or slope_pct <= 0:
        return _no_detect("ascending_triangle",
                          f"touches={resistance_touches} slope={slope_pct:.4f}")

    confidence = min(1.0, resistance_touches / 5) * min(1.0, slope_pct / 0.002)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"resistance": round(resistance, 2),
                       "touches": int(resistance_touches),
                       "support_slope_pct": round(slope_pct, 4)},
    }


def detect_vwap_reclaim(df: pd.DataFrame) -> PatternResult:
    """VWAP Reclaim: price drops below VWAP intraday then closes above it.

    Bullish short-term signal. Requires intraday (1H or lower) data.
    Uses rolling VWAP over last 20 bars.
    """
    if len(df) < 20:
        return _no_detect("vwap_reclaim", "insufficient bars")
    if "volume" not in df.columns:
        return _no_detect("vwap_reclaim", "no volume column")

    closes  = df["close"].values[-20:]
    volumes = df["volume"].values[-20:]
    vwap    = np.sum(closes * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else closes[-1]

    prev_close  = float(df["close"].iloc[-2])
    curr_close  = float(df["close"].iloc[-1])
    prev_below  = prev_close < vwap
    curr_above  = curr_close > vwap
    reclaim_pct = (curr_close - vwap) / vwap if vwap > 0 else 0

    if not (prev_below and curr_above):
        return _no_detect("vwap_reclaim", f"no cross: prev={prev_close:.2f} vwap={vwap:.2f} curr={curr_close:.2f}")

    confidence  = min(1.0, reclaim_pct / 0.005)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"vwap": round(vwap, 2), "close": round(curr_close, 2),
                       "reclaim_pct": round(reclaim_pct, 4)},
    }


def detect_rsi_divergence(df: pd.DataFrame) -> PatternResult:
    """Bullish RSI divergence: price makes lower low, RSI makes higher low.

    Classic reversal signal. Looks at last 20 bars.
    """
    if len(df) < 30:
        return _no_detect("rsi_divergence", "insufficient bars")

    closes = df["close"].values
    rsi    = _compute_rsi(closes, period=14)
    if rsi is None or len(rsi) < 20:
        return _no_detect("rsi_divergence", "insufficient RSI data")

    price_window = closes[-20:]
    rsi_window   = rsi[-20:]

    # Find two troughs in price
    price_troughs = _local_minima(price_window, distance=5)
    if len(price_troughs) < 2:
        return _no_detect("rsi_divergence", "insufficient price troughs")

    t1, t2 = price_troughs[-2], price_troughs[-1]
    price_lower_low = price_window[t2] < price_window[t1]
    rsi_higher_low  = rsi_window[t2]  > rsi_window[t1]

    if not (price_lower_low and rsi_higher_low):
        return _no_detect("rsi_divergence", "no divergence")

    rsi_diff = rsi_window[t2] - rsi_window[t1]
    confidence = min(1.0, rsi_diff / 10)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"price_t1": round(float(price_window[t1]), 2),
                       "price_t2": round(float(price_window[t2]), 2),
                       "rsi_t1":   round(float(rsi_window[t1]), 1),
                       "rsi_t2":   round(float(rsi_window[t2]), 1)},
    }


def detect_golden_cross(df: pd.DataFrame) -> PatternResult:
    """Golden Cross: MA50 crosses above MA200. Bullish trend confirmation."""
    if len(df) < 205:
        return _no_detect("golden_cross", "insufficient bars for MA200")

    closes = df["close"].values
    ma50   = np.convolve(closes, np.ones(50) / 50, mode="valid")
    ma200  = np.convolve(closes, np.ones(200) / 200, mode="valid")

    # Align arrays
    offset = len(ma50) - len(ma200)
    ma50_aligned = ma50[offset:] if offset > 0 else ma50
    ma200_aligned = ma200

    if len(ma50_aligned) < 2 or len(ma200_aligned) < 2:
        return _no_detect("golden_cross", "alignment failed")

    prev_above = ma50_aligned[-2] > ma200_aligned[-2]
    curr_above = ma50_aligned[-1] > ma200_aligned[-1]
    just_crossed = (not prev_above) and curr_above

    # Also useful as a "state" signal even without fresh cross
    spread_pct = (ma50_aligned[-1] - ma200_aligned[-1]) / ma200_aligned[-1] if ma200_aligned[-1] > 0 else 0

    if not just_crossed and spread_pct <= 0:
        return _no_detect("golden_cross", f"no cross, spread={spread_pct:.3f}")

    confidence = 1.0 if just_crossed else min(0.7, spread_pct / 0.02)
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"just_crossed": just_crossed,
                       "ma50": round(float(ma50_aligned[-1]), 2),
                       "ma200": round(float(ma200_aligned[-1]), 2),
                       "spread_pct": round(spread_pct, 4)},
    }


def detect_support_bounce(df: pd.DataFrame) -> PatternResult:
    """Support Bounce: price touches a key support level and reverses up.

    Support: a price level tested 2+ times in last 60 bars.
    """
    if len(df) < 40:
        return _no_detect("support_bounce", "insufficient bars")

    lows   = df["low"].values[-60:] if len(df) >= 60 else df["low"].values
    closes = df["close"].values

    support_level = np.percentile(lows, 15)
    curr_low   = float(df["low"].iloc[-1])
    curr_close = float(df["close"].iloc[-1])

    touches = np.sum(np.abs(lows - support_level) / support_level < 0.015)
    near_support    = abs(curr_low - support_level) / support_level < 0.015
    closing_higher  = curr_close > support_level * 1.005

    if touches < 2 or not near_support or not closing_higher:
        return _no_detect("support_bounce",
                          f"touches={touches} near={near_support} closing_higher={closing_higher}")

    confidence = min(1.0, touches / 4) * 0.8
    return {
        "detected":   True,
        "confidence": round(confidence, 3),
        "details":    {"support": round(support_level, 2),
                       "touches": int(touches),
                       "curr_low": round(curr_low, 2)},
    }


# ── Master aggregator ─────────────────────────────────────────────────────────

ALL_DETECTORS = {
    "cup_handle":         detect_cup_handle,
    "head_shoulders":     detect_head_shoulders,
    "bull_flag":          detect_bull_flag,
    "double_bottom":      detect_double_bottom,
    "breakout":           detect_breakout,
    "ascending_triangle": detect_ascending_triangle,
    "vwap_reclaim":       detect_vwap_reclaim,
    "rsi_divergence":     detect_rsi_divergence,
    "golden_cross":       detect_golden_cross,
    "support_bounce":     detect_support_bounce,
}


def detect_all(df: pd.DataFrame) -> dict:
    """Run all pattern detectors and return aggregated results.

    Returns dict with:
      "detected_patterns": list of pattern names that fired
      "pattern_score":     0-1 composite (weighted by confidence)
      "raw":               {pattern_name: PatternResult}
    """
    raw = {}
    for name, fn in ALL_DETECTORS.items():
        try:
            raw[name] = fn(df)
        except Exception as e:
            logger.warning("Pattern %s raised: %s", name, e)
            raw[name] = _no_detect(name, str(e))

    detected = [name for name, r in raw.items() if r["detected"]]
    if not detected:
        pattern_score = 0.0
    else:
        pattern_score = float(np.mean([raw[n]["confidence"] for n in detected]))

    return {
        "detected_patterns": detected,
        "pattern_score":     round(pattern_score, 3),
        "raw":               raw,
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _no_detect(name: str, reason: str = "") -> PatternResult:
    return {"detected": False, "confidence": 0.0, "details": {"reason": reason}}


def _local_maxima(arr: np.ndarray, distance: int = 5) -> np.ndarray:
    from scipy.signal import find_peaks  # type: ignore
    peaks, _ = find_peaks(arr, distance=distance)
    return peaks


def _local_minima(arr: np.ndarray, distance: int = 5) -> np.ndarray:
    return _local_maxima(-arr, distance=distance)


def _compute_rsi(closes: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.convolve(gains,  np.ones(period) / period, mode="valid")
    avg_loss = np.convolve(losses, np.ones(period) / period, mode="valid")
    rs  = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Run algebraic pattern detection")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--bars",   type=int, default=300)
    args = parser.parse_args()

    from data.sources.tvdatafeed_client import get_ohlcv  # type: ignore
    df = get_ohlcv(args.symbol, interval="1D", n_bars=args.bars)
    results = detect_all(df)

    print(f"\n{args.symbol} — Pattern Score: {results['pattern_score']}")
    print(f"Detected: {results['detected_patterns'] or 'none'}")
    for name in results["detected_patterns"]:
        r = results["raw"][name]
        print(f"  {name}: confidence={r['confidence']} | {r['details']}")
