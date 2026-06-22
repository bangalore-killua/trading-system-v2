"""
technical_analysis.py — pandas-ta Indicator + Candlestick Pattern Engine

Replaces TradingSystem's `ta` library with `pandas-ta` which gives us:
  - 130+ technical indicators (same API, more complete)
  - 60+ candlestick patterns as binary flags (0/1) via df.ta.cdl_pattern()
  - Full OHLC geometry features for LightGBM feature vector (Phase 3+)

Output format: a flat dict of all indicator values for a given symbol's OHLCV
DataFrame. This dict is used in two ways:
  1. Signal basket: specific named keys checked for binary thresholds
  2. Feature vector: all raw numeric values passed to LightGBM (Phase 3+)

Context-aware combinations (e.g. hammer_at_200ma) are computed here so that
the signal basket can treat them as single named signals.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# All 60+ pandas-ta candlestick pattern names — generated via pandas_ta.cdl_pattern("all")
# Stored here explicitly so we can iterate them without a live DataFrame
CDL_PATTERNS = [
    "cdl_2crows", "cdl_3blackcrows", "cdl_3inside", "cdl_3linestrike",
    "cdl_3outside", "cdl_3starsinsouth", "cdl_3whitesoldiers", "cdl_abandonedbaby",
    "cdl_advanceblock", "cdl_belthold", "cdl_breakaway", "cdl_closingmarubozu",
    "cdl_concealbabyswall", "cdl_counterattack", "cdl_darkcloudcover", "cdl_doji",
    "cdl_dojistar", "cdl_dragonflydoji", "cdl_engulfing", "cdl_eveningdojistar",
    "cdl_eveningstar", "cdl_gapsidesidewhite", "cdl_gravestonedoji", "cdl_hammer",
    "cdl_hangingman", "cdl_harami", "cdl_haramicross", "cdl_highwave",
    "cdl_hikkake", "cdl_hikkakemod", "cdl_homingpigeon", "cdl_identical3crows",
    "cdl_inneck", "cdl_inside", "cdl_invertedhammer", "cdl_kicking",
    "cdl_kickingbylength", "cdl_ladderbottom", "cdl_longleggeddoji", "cdl_longline",
    "cdl_marubozu", "cdl_matchinglow", "cdl_mathold", "cdl_morningdojistar",
    "cdl_morningstar", "cdl_onneck", "cdl_piercing", "cdl_rickshawman",
    "cdl_risefall3methods", "cdl_separatinglines", "cdl_shootingstar",
    "cdl_shortline", "cdl_spinningtop", "cdl_stalledpattern", "cdl_sticksandwich",
    "cdl_takuri", "cdl_tasukigap", "cdl_thrusting", "cdl_tristar",
    "cdl_unique3river", "cdl_upsidegap2crows", "cdl_xsidegap3methods",
]


def compute_indicators(df: pd.DataFrame, atr_period: int = 14) -> Dict:
    """Compute all indicators and candlestick patterns for a OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
            Must have at least 200 rows for all MAs to be valid.
        atr_period: period for ATR and related calculations.

    Returns:
        Flat dict with all indicator values and binary flags.
        Returns an empty dict if the DataFrame is too short.
    """
    if len(df) < 50:
        logger.warning("compute_indicators: DataFrame too short (%d rows)", len(df))
        return {}

    try:
        import pandas_ta as ta  # type: ignore
    except ImportError:
        logger.error("pandas-ta not installed — run: pip install pandas-ta")
        return {}

    # Ensure standard column names that pandas-ta expects
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns:
        df["close"] = df["adj close"]

    result: Dict = {}

    # ── Price & Moving Averages ─────────────────────────────────────────────
    close = df["close"]
    last_close = float(close.iloc[-1])

    sma20 = df.ta.sma(length=20).iloc[-1]
    sma50 = df.ta.sma(length=50).iloc[-1]
    sma200 = df.ta.sma(length=200).iloc[-1] if len(df) >= 200 else None
    ema9 = df.ta.ema(length=9).iloc[-1]
    ema21 = df.ta.ema(length=21).iloc[-1]

    result["price"] = last_close
    result["sma20"] = float(sma20) if pd.notna(sma20) else None
    result["sma50"] = float(sma50) if pd.notna(sma50) else None
    result["sma200"] = float(sma200) if sma200 is not None and pd.notna(sma200) else None
    result["ema9"] = float(ema9) if pd.notna(ema9) else None
    result["ema21"] = float(ema21) if pd.notna(ema21) else None

    # Percent distance from MAs (signed — positive = price above MA)
    result["price_vs_20ma"] = _pct_diff(last_close, result["sma20"])
    result["price_vs_50ma"] = _pct_diff(last_close, result["sma50"])
    result["price_vs_200ma"] = _pct_diff(last_close, result["sma200"])

    # Binary: price above MAs
    result["price_above_20ma"] = int(result["price_vs_20ma"] > 0) if result["price_vs_20ma"] is not None else 0
    result["price_above_50ma"] = int(result["price_vs_50ma"] > 0) if result["price_vs_50ma"] is not None else 0
    result["price_above_200ma"] = int(result["price_vs_200ma"] > 0) if result["price_vs_200ma"] is not None else 0

    # ── Trend: daily trend classification ──────────────────────────────────
    # Bullish daily trend = price > SMA20 > SMA50 (all aligned)
    if result["sma20"] and result["sma50"]:
        result["daily_trend_bullish"] = int(last_close > result["sma20"] > result["sma50"])
    else:
        result["daily_trend_bullish"] = 0

    # ── RSI ─────────────────────────────────────────────────────────────────
    rsi_series = df.ta.rsi(length=14)
    rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and pd.notna(rsi_series.iloc[-1]) else None
    result["rsi"] = rsi
    result["rsi_oversold"] = int(rsi < 30) if rsi is not None else 0
    result["rsi_overbought"] = int(rsi > 70) if rsi is not None else 0
    # Momentum zone = RSI between 50-70 (trending but not overextended)
    result["rsi_momentum_zone"] = int(50 <= rsi <= 70) if rsi is not None else 0

    # ── MACD ────────────────────────────────────────────────────────────────
    macd_df = df.ta.macd(fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_line = float(macd_df["MACD_12_26_9"].iloc[-1])
        macd_signal = float(macd_df["MACDs_12_26_9"].iloc[-1])
        macd_hist = float(macd_df["MACDh_12_26_9"].iloc[-1])
        macd_hist_prev = float(macd_df["MACDh_12_26_9"].iloc[-2]) if len(macd_df) > 1 else 0.0
        result["macd"] = macd_line
        result["macd_signal"] = macd_signal
        result["macd_hist"] = macd_hist
        # Bullish: histogram positive and growing
        result["macd_bullish"] = int(macd_hist > 0 and macd_hist > macd_hist_prev)
        # Bullish cross: line crossed above signal in last 2 bars
        result["macd_bullish_cross"] = int(macd_line > macd_signal and macd_hist_prev <= 0)
        result["macd_bearish_cross"] = int(macd_line < macd_signal and macd_hist_prev >= 0)
    else:
        result.update({"macd": None, "macd_signal": None, "macd_hist": None,
                       "macd_bullish": 0, "macd_bullish_cross": 0, "macd_bearish_cross": 0})

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    # Column names vary by pandas-ta version (0.4.71b0 appends std twice:
    # "BBU_20_2.0_2.0" instead of "BBU_20_2.0"). Use startswith() lookup.
    bbands = df.ta.bbands(length=20, std=2)
    if bbands is not None and not bbands.empty:
        _bb = {p: next((c for c in bbands.columns if c.startswith(p)), None)
               for p in ("BBU", "BBL", "BBM", "BBB", "BBP")}
        bb_upper = float(bbands[_bb["BBU"]].iloc[-1]) if _bb["BBU"] else None
        bb_lower = float(bbands[_bb["BBL"]].iloc[-1]) if _bb["BBL"] else None
        bb_mid   = float(bbands[_bb["BBM"]].iloc[-1]) if _bb["BBM"] else None
        bb_width = float(bbands[_bb["BBB"]].iloc[-1]) if _bb["BBB"] else None  # bandwidth
        bb_pct   = float(bbands[_bb["BBP"]].iloc[-1]) if _bb["BBP"] else None  # %B
        result["bb_upper"] = bb_upper
        result["bb_lower"] = bb_lower
        result["bb_mid"] = bb_mid
        result["bb_width"] = bb_width
        result["bb_position"] = bb_pct  # 0=at lower band, 1=at upper band
        # Squeeze: bands extremely tight (width < 5% of price is historically low)
        result["bb_squeeze"] = int(bb_width < 5.0) if pd.notna(bb_width) else 0
    else:
        result.update({"bb_upper": None, "bb_lower": None, "bb_mid": None,
                       "bb_width": None, "bb_position": None, "bb_squeeze": 0})

    # ── Volume ──────────────────────────────────────────────────────────────
    volume = df["volume"]
    vol_sma20 = float(volume.rolling(20).mean().iloc[-1])
    last_vol = float(volume.iloc[-1])
    result["volume"] = last_vol
    result["volume_sma20"] = vol_sma20
    result["volume_ratio"] = round(last_vol / vol_sma20, 2) if vol_sma20 > 0 else 1.0
    # Volume surge: today's volume > 2x 20-day average
    result["volume_surge"] = int(result["volume_ratio"] > 2.0)
    # Volume confirming: price up + volume above average
    price_change = close.iloc[-1] - close.iloc[-2] if len(close) > 1 else 0
    result["volume_confirming"] = int(price_change > 0 and result["volume_ratio"] > 1.0)
    # Distribution: price down + volume above average (institutional selling)
    result["volume_distribution"] = int(price_change < 0 and result["volume_ratio"] > 1.3)

    # ── ATR ─────────────────────────────────────────────────────────────────
    atr_series = df.ta.atr(length=atr_period)
    atr = float(atr_series.iloc[-1]) if atr_series is not None and pd.notna(atr_series.iloc[-1]) else None
    result["atr"] = atr
    result["atr_pct"] = round(atr / last_close * 100, 2) if atr and last_close > 0 else None
    # Extreme ATR: if daily ATR > 5% of price, stock is too volatile for new entry
    result["atr_not_extreme"] = int((result["atr_pct"] or 0) < 5.0)

    # ── ADX (Trend strength) ─────────────────────────────────────────────────
    adx_df = df.ta.adx(length=14)
    if adx_df is not None and not adx_df.empty:
        result["adx"] = float(adx_df["ADX_14"].iloc[-1])
        result["dmp"] = float(adx_df["DMP_14"].iloc[-1])  # +DI
        result["dmn"] = float(adx_df["DMN_14"].iloc[-1])  # -DI
        # Strong trend: ADX > 25 with +DI > -DI = directional bull trend
        result["adx_strong_trend"] = int(
            result["adx"] > 25 and result["dmp"] > result["dmn"]
        )
    else:
        result.update({"adx": None, "dmp": None, "dmn": None, "adx_strong_trend": 0})

    # ── Stochastic ──────────────────────────────────────────────────────────
    stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        result["stoch_k"] = float(stoch["STOCHk_14_3_3"].iloc[-1])
        result["stoch_d"] = float(stoch["STOCHd_14_3_3"].iloc[-1])
        result["stoch_oversold"] = int(result["stoch_k"] < 20)
        result["stoch_overbought"] = int(result["stoch_k"] > 80)
    else:
        result.update({"stoch_k": None, "stoch_d": None,
                       "stoch_oversold": 0, "stoch_overbought": 0})

    # ── OHLC Geometry (for LightGBM feature vector) ──────────────────────────
    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    candle_range = h - l
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    result["body_size"] = round(body / candle_range, 4) if candle_range > 0 else 0
    result["body_direction"] = 1 if c >= o else -1
    result["upper_wick"] = round(upper_wick / candle_range, 4) if candle_range > 0 else 0
    result["lower_wick"] = round(lower_wick / candle_range, 4) if candle_range > 0 else 0
    result["wick_ratio"] = round(upper_wick / lower_wick, 2) if lower_wick > 0 else 0
    result["candle_range_vs_atr"] = round(candle_range / atr, 2) if atr and atr > 0 else None
    # Gap detection
    if len(df) > 1:
        prev_close = float(df["close"].iloc[-2])
        result["gap_up"] = int(o > prev_close * 1.005)
        result["gap_down"] = int(o < prev_close * 0.995)
    else:
        result["gap_up"] = 0
        result["gap_down"] = 0

    # ── Candlestick Patterns ─────────────────────────────────────────────────
    _add_candlestick_patterns(df, result)

    # ── Context-Aware Combinations ───────────────────────────────────────────
    # These synthetic signals check candlestick pattern + location context.
    # They matter for signal basket: a hammer at 200MA support is more significant
    # than a hammer in the middle of a range.
    _add_context_combinations(result)

    return result


def _add_candlestick_patterns(df: pd.DataFrame, result: Dict) -> None:
    """Run all pandas-ta cdl_pattern checks and store as 0/1 binary flags."""
    try:
        import pandas_ta as ta  # type: ignore
        # pandas-ta computes all patterns at once and returns a DataFrame
        patterns_df = df.ta.cdl_pattern(name="all")
        if patterns_df is not None and not patterns_df.empty:
            last_row = patterns_df.iloc[-1]
            for col in patterns_df.columns:
                # pandas-ta returns 100/-100 for bullish/bearish; normalize to 1/0/-1
                val = last_row[col]
                col_lower = col.lower()
                if pd.notna(val) and val != 0:
                    result[col_lower] = 1 if val > 0 else -1
                else:
                    result[col_lower] = 0
    except Exception:
        logger.exception("Candlestick pattern computation error")
        # Fill all patterns with 0 rather than crashing
        for pat in CDL_PATTERNS:
            result.setdefault(pat, 0)


def _add_context_combinations(result: Dict) -> None:
    """Add context-aware candlestick combinations as named boolean features.

    These are inputs to the LightGBM feature vector and also used as named
    signals in the signal basket (though the basket addresses them by key name).
    """
    hammer = result.get("cdl_hammer", 0)
    engulfing = result.get("cdl_engulfing", 0)
    marubozu = result.get("cdl_marubozu", 0)
    doji = result.get("cdl_doji", 0)
    morning_star = result.get("cdl_morningstar", 0)
    evening_star = result.get("cdl_eveningstar", 0)

    at_200ma = result.get("price_above_200ma") == 0 and result.get("price_vs_200ma") is not None and abs(result["price_vs_200ma"]) < 2.0
    at_50ma = result.get("price_above_50ma") == 0 and result.get("price_vs_50ma") is not None and abs(result["price_vs_50ma"]) < 2.0
    volume_surge = result.get("volume_surge", 0) == 1

    # Hammer at 200 MA support — high-conviction bounce signal
    result["hammer_at_200ma"] = int(hammer == 1 and at_200ma)

    # Hammer at 50 MA support
    result["hammer_at_50ma"] = int(hammer == 1 and at_50ma)

    # Bullish engulfing at support (50MA or 200MA)
    result["engulfing_at_support"] = int(engulfing == 1 and (at_200ma or at_50ma))

    # Strong green candle with high volume — institutional buying confirmation
    result["marubozu_on_volume_surge"] = int(marubozu == 1 and volume_surge)

    # Morning star near support = reversal with location confirmation
    result["morning_star_at_support"] = int(morning_star == 1 and (at_200ma or at_50ma))

    # Evening star near recent highs = reversal warning
    price_above_200 = result.get("price_above_200ma", 0) == 1
    result["evening_star_near_highs"] = int(evening_star == -1 and price_above_200)


def _pct_diff(price: float, reference: Optional[float]) -> Optional[float]:
    """Signed percentage difference: (price - ref) / ref * 100"""
    if reference is None or reference == 0:
        return None
    return round((price - reference) / reference * 100, 2)
