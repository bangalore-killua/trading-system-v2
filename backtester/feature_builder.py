"""
feature_builder.py — Canonical Feature Registry + Shared Feature Construction

Single source of truth for:
  - FEATURE_NAMES: ordered list of all 65 canonical feature names
  - FEATURE_DEFAULTS: neutral fill values for features unavailable historically
  - _build_indicators(): compute all technical columns from OHLCV DataFrame
  - _build_feature_row(): assemble one complete feature dict

Imported by:
  backtester/ml_backtest.py    — inference during portfolio simulation
  backtester/intelligent_backtest.py — training data label generation
  strategy/training_pipeline.py — validation guard

IMPORTANT: Every feature added here must match scorer.py FeatureVector.to_dict()
exactly. If names drift, AutoGluon sees all-NaN columns at inference time.
"""

from typing import Optional
import numpy as np
import pandas as pd

# ── Canonical feature list (65 features, same order as FeatureVector.to_dict()) ─

FEATURE_NAMES: list[str] = [
    # Technical — continuous
    "rsi", "macd_hist", "macd_line", "macd_signal",
    "price_vs_20ma", "price_vs_50ma", "price_vs_200ma",
    "bb_position", "bb_width", "volume_ratio", "atr_pct", "adx", "stoch_k",
    "ema9", "ema21",
    # Technical — binary flags
    "price_above_20ma", "price_above_50ma", "price_above_200ma",
    "rsi_oversold", "rsi_overbought", "rsi_momentum_zone",
    "macd_bullish", "macd_bullish_cross", "macd_bearish_cross",
    "volume_surge", "volume_confirming", "volume_distribution",
    "bb_squeeze", "adx_strong_trend", "daily_trend_bullish",
    "stoch_oversold", "stoch_overbought", "gap_up", "gap_down",
    # OHLC geometry
    "body_size", "body_direction", "upper_wick", "lower_wick",
    "wick_ratio", "candle_range_vs_atr",
    # Candlestick patterns (set to 0 in backtester — no TA-Lib)
    "hammer_at_200ma", "hammer_at_50ma", "engulfing_at_support",
    "marubozu_on_volume_surge", "morning_star_at_support", "evening_star_near_highs",
    # Fundamentals
    "pe_ratio", "roe", "revenue_growth_yoy", "earnings_revision_3m",
    "debt_equity", "insider_buying_30d",
    # News (fixed neutral in backtester)
    "finbert_score", "news_volume_7d", "threat_level_encoded",
    # Macro
    "vix", "yield_spread_10y2y", "spy_above_200ma",
    "inverted_yield_curve", "fear_greed_encoded",
    # Context
    "regime_encoded", "setup_type_encoded", "sector_bucket",
    "mktcap_bucket_encoded", "days_since_earnings",
]

FEATURE_DEFAULTS: dict = {
    "finbert_score":       0.0,
    "news_volume_7d":      5,
    "threat_level_encoded": 0,
    "vix":                 20.0,
    "yield_spread_10y2y":  0.5,
    "spy_above_200ma":     1,
    "inverted_yield_curve": 0,
    "fear_greed_encoded":  2,
    "regime_encoded":      4,
    "setup_type_encoded":  0,
    "sector_bucket":       "OTHER",
    "mktcap_bucket_encoded": 3,
    "days_since_earnings": 45,
    "earnings_revision_3m": None,
    "insider_buying_30d":  0,
    "hammer_at_200ma":     0,
    "hammer_at_50ma":      0,
    "engulfing_at_support": 0,
    "marubozu_on_volume_surge": 0,
    "morning_star_at_support": 0,
    "evening_star_near_highs": 0,
}

CATEGORICAL_FEATURES: list[str] = ["sector_bucket"]

# ── Sector / encoding lookups (keep in sync with scorer.py) ───────────────────

try:
    from strategy.scorer import SECTOR_BUCKETS, REGIME_ENCODING, MKTCAP_ENCODING
except ImportError:
    SECTOR_BUCKETS  = {}
    REGIME_ENCODING = {"NEUTRAL": 4}
    MKTCAP_ENCODING = {"large": 4, "mega": 5, "mid": 3, "small": 2, "micro": 1}


# ── Technical indicators (pure pandas/numpy — no TA-Lib) ─────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    low_n  = low.rolling(period).min()
    high_n = high.rolling(period).max()
    denom  = (high_n - low_n).replace(0, np.nan)
    return (close - low_n) / denom * 100


def _adx_simple(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    dm_plus  = (high.diff()).clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    atr_s    = _atr(high, low, close, period)
    di_plus  = dm_plus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan) * 100
    di_minus = dm_minus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan) * 100
    dx       = ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100)
    return dx.ewm(com=period - 1, adjust=False).mean()


def _rolling_pct_fast(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile approximation via min-max normalisation.

    Returns values in [0, 1]: 0 = at rolling minimum, 1 = at rolling maximum.
    Uses pure rolling min/max — O(n) and fully vectorised.
    min_periods = window // 4 so early bars still get a value.
    """
    min_p = max(1, window // 4)
    r_min = series.rolling(window, min_periods=min_p).min()
    r_max = series.rolling(window, min_periods=min_p).max()
    span  = (r_max - r_min).replace(0, np.nan)
    return ((series - r_min) / span).fillna(0.5)


def _build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features and add as columns.

    Expects Title Case OHLCV columns: Close, High, Low, Open, Volume.
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi14       = _rsi(c, 14)
    macd_line, macd_sig, macd_hist = _macd(c)
    atr14       = _atr(h, l, c, 14)
    stoch_k     = _stoch_k(h, l, c, 14)
    adx         = _adx_simple(h, l, c, 14)
    ema9        = c.ewm(span=9,  adjust=False).mean()
    ema21       = c.ewm(span=21, adjust=False).mean()
    sma20       = c.rolling(20).mean()
    sma50       = c.rolling(50).mean()
    sma200      = c.rolling(200).mean()
    vol20       = v.rolling(20).mean()
    bb_std      = c.rolling(20).std()
    bb_up       = sma20 + 2 * bb_std
    bb_dn       = sma20 - 2 * bb_std
    bb_width    = (bb_up - bb_dn) / sma20.replace(0, np.nan)

    df = df.copy()
    df["rsi"]              = rsi14
    df["macd_hist"]        = macd_hist
    df["macd_line"]        = macd_line
    df["macd_signal"]      = macd_sig
    df["price_vs_20ma"]    = c / sma20.replace(0, np.nan) - 1
    df["price_vs_50ma"]    = c / sma50.replace(0, np.nan) - 1
    df["price_vs_200ma"]   = c / sma200.replace(0, np.nan) - 1
    df["sma200"]           = sma200
    df["bb_position"]      = (c - bb_dn) / (bb_up - bb_dn).replace(0, np.nan)
    df["bb_width"]         = bb_width
    df["volume_ratio"]     = v / vol20.replace(0, np.nan)
    df["atr_pct"]          = atr14 / c.replace(0, np.nan)
    df["adx"]              = adx
    df["stoch_k"]          = stoch_k
    df["ema9"]             = ema9
    df["ema21"]            = ema21
    df["atr14"]            = atr14

    df["price_above_20ma"]    = (c > sma20).astype(int)
    df["price_above_50ma"]    = (c > sma50).astype(int)
    df["price_above_200ma"]   = (c > sma200).astype(int)

    # Percentile-ranked binary flags — self-adjusting across regimes
    # RSI: 252-day window (one trading year of context)
    rsi_pct   = _rolling_pct_fast(rsi14,           252)
    # Volume / ADX / Stochastic: 60-day window (one trading quarter)
    vol_pct   = _rolling_pct_fast(df["volume_ratio"], 60)
    adx_pct   = _rolling_pct_fast(adx,              60)
    stoch_pct = _rolling_pct_fast(stoch_k,           60)

    df["rsi_oversold"]        = (rsi_pct < 0.20).astype(int)   # bottom 20% of yearly history
    df["rsi_overbought"]      = (rsi_pct > 0.80).astype(int)   # top 20% of yearly history
    df["rsi_momentum_zone"]   = ((rsi14 >= 40) & (rsi14 <= 60)).astype(int)  # absolute zone kept
    df["macd_bullish"]        = (macd_line > macd_sig).astype(int)
    df["macd_bullish_cross"]  = ((macd_line > macd_sig) & (macd_line.shift() <= macd_sig.shift())).astype(int)
    df["macd_bearish_cross"]  = ((macd_line < macd_sig) & (macd_line.shift() >= macd_sig.shift())).astype(int)
    df["volume_surge"]        = (vol_pct > 0.85).astype(int)   # top 15% of quarterly volume
    df["volume_confirming"]   = ((vol_pct > 0.70) & (c > c.shift())).astype(int)
    df["volume_distribution"] = ((vol_pct > 0.70) & (c < c.shift())).astype(int)
    df["bb_squeeze"]          = (bb_width < bb_width.rolling(20).mean() * 0.7).astype(int)
    df["adx_strong_trend"]    = (adx_pct > 0.60).astype(int)   # top 40% of quarterly ADX
    df["daily_trend_bullish"] = (c > c.shift()).astype(int)
    df["stoch_oversold"]      = (stoch_pct < 0.15).astype(int) # bottom 15% of quarterly stoch
    df["stoch_overbought"]    = (stoch_pct > 0.85).astype(int) # top 15% of quarterly stoch
    df["gap_up"]              = (df["Open"] > c.shift() * 1.01).astype(int)
    df["gap_down"]            = (df["Open"] < c.shift() * 0.99).astype(int)

    o          = df["Open"]
    body       = (c - o).abs()
    candle_rng = h - l
    df["body_size"]           = body / c.replace(0, np.nan)
    df["body_direction"]      = np.sign(c - o)
    df["upper_wick"]          = (h - c.clip(lower=o)) / c.replace(0, np.nan)
    df["lower_wick"]          = (c.clip(upper=o) - l) / c.replace(0, np.nan)
    df["wick_ratio"]          = (h - l - body) / body.replace(0, np.nan)
    df["candle_range_vs_atr"] = candle_rng / atr14.replace(0, np.nan)

    # ── Candlestick pattern features (pandas OHLCV rules, no TA-Lib) ─────────
    # The live system uses YOLOv8 chart-pattern detection; the backtester uses
    # these rule-based approximations instead. Retrain models periodically once
    # enough live-labelled candle data accumulates.
    sma50      = c.rolling(50, min_periods=10).mean()
    raw_rng    = candle_rng.replace(0, np.nan)
    low_wick_r = o.clip(upper=c) - l          # lower boundary of real body → low
    up_wick_r  = h - o.clip(lower=c)          # high → upper boundary of real body

    _hammer = (
        (body / raw_rng < 0.35) &
        (low_wick_r > body * 2.0) &
        (up_wick_r  < raw_rng * 0.15)
    )
    df["hammer_at_200ma"] = (
        _hammer & ((c - sma200).abs() / sma200.replace(0, np.nan) < 0.03)
    ).fillna(False).astype(int)
    df["hammer_at_50ma"] = (
        _hammer & ((c - sma50).abs() / sma50.replace(0, np.nan) < 0.03)
    ).fillna(False).astype(int)

    _prior_bear = c.shift(1) < o.shift(1)
    _curr_bull  = c > o
    _engulf     = (
        (o <= c.shift(1)) & (c >= o.shift(1)) &
        (body > (o.shift(1) - c.shift(1)).abs())
    )
    _at_sup_20  = c < c.rolling(20, min_periods=5).min().shift(1) * 1.05
    df["engulfing_at_support"] = (
        _prior_bear & _curr_bull & _engulf & _at_sup_20
    ).fillna(False).astype(int)

    df["marubozu_on_volume_surge"] = (
        (body / raw_rng > 0.85) & (df["volume_ratio"] > 2.0)
    ).fillna(False).astype(int)

    _d1_bear_big = (o.shift(2) > c.shift(2)) & (
        (o.shift(2) - c.shift(2)) > (h.shift(2) - l.shift(2)) * 0.50
    )
    _d2_small   = (c.shift(1) - o.shift(1)).abs() < (h.shift(1) - l.shift(1)) * 0.30
    _d3_bull_mid = (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
    df["morning_star_at_support"] = (
        _d1_bear_big & _d2_small & _d3_bull_mid &
        (c < c.rolling(20, min_periods=5).min().shift(1) * 1.07)
    ).fillna(False).astype(int)

    _d1_bull_big = (c.shift(2) > o.shift(2)) & (
        (c.shift(2) - o.shift(2)) > (h.shift(2) - l.shift(2)) * 0.50
    )
    _d3_bear_big = (o > c) & ((o - c) > raw_rng * 0.50)
    df["evening_star_near_highs"] = (
        _d1_bull_big & _d2_small & _d3_bear_big &
        (c > c.rolling(252, min_periods=50).max().shift(1) * 0.95)
    ).fillna(False).astype(int)

    return df


# ── Fundamentals helper ───────────────────────────────────────────────────────

def _mktcap_bucket(market_cap: Optional[float]) -> str:
    if market_cap is None:
        return "mid"
    if market_cap > 200e9:
        return "mega"
    if market_cap > 10e9:
        return "large"
    if market_cap > 2e9:
        return "mid"
    if market_cap > 300e6:
        return "small"
    return "micro"


def _fetch_fundamentals(symbol: str) -> dict:
    """Pull current yfinance snapshot as a stable proxy for all history."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        return {
            "pe_ratio":            info.get("trailingPE"),
            "roe":                 info.get("returnOnEquity"),
            "revenue_growth_yoy":  info.get("revenueGrowth"),
            "earnings_revision_3m": None,
            "debt_equity":         info.get("debtToEquity"),
            "insider_buying_30d":  0,
            "market_cap":          info.get("marketCap"),
        }
    except Exception:
        return {}


# ── Feature-vector assembler ──────────────────────────────────────────────────

def _build_feature_row(
    sym:      str,
    row:      pd.Series,
    fund:     dict,
    vix:      Optional[float],
    ys:       Optional[float],
    spy_row:  pd.Series,
    regime:   int,
) -> dict:
    """Assemble one complete feature dict matching FeatureVector.to_dict().

    Parameters
    ----------
    sym     : ticker symbol (for sector lookup)
    row     : DataFrame row already processed by _build_indicators()
    fund    : dict from _fetch_fundamentals()
    vix     : VIX close (None → uses FEATURE_DEFAULTS)
    ys      : 10Y–2Y yield spread (None → uses FEATURE_DEFAULTS)
    spy_row : SPY indicator row (same date)
    regime  : regime_encoded integer from _approx_regime()
    """
    f: dict = {}

    # Technical continuous + binary (including pandas-computed candlestick patterns)
    for col in (
        "rsi", "macd_hist", "macd_line", "macd_signal",
        "price_vs_20ma", "price_vs_50ma", "price_vs_200ma",
        "bb_position", "bb_width", "volume_ratio", "atr_pct", "adx",
        "stoch_k", "ema9", "ema21",
        "price_above_20ma", "price_above_50ma", "price_above_200ma",
        "rsi_oversold", "rsi_overbought", "rsi_momentum_zone",
        "macd_bullish", "macd_bullish_cross", "macd_bearish_cross",
        "volume_surge", "volume_confirming", "volume_distribution",
        "bb_squeeze", "adx_strong_trend", "daily_trend_bullish",
        "stoch_oversold", "stoch_overbought", "gap_up", "gap_down",
        "body_size", "body_direction", "upper_wick", "lower_wick",
        "wick_ratio", "candle_range_vs_atr",
        "hammer_at_200ma", "hammer_at_50ma", "engulfing_at_support",
        "marubozu_on_volume_surge", "morning_star_at_support", "evening_star_near_highs",
    ):
        val = row.get(col)
        f[col] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    # Fundamentals
    f["pe_ratio"]             = fund.get("pe_ratio")
    f["roe"]                  = fund.get("roe")
    f["revenue_growth_yoy"]   = fund.get("revenue_growth_yoy")
    f["earnings_revision_3m"] = None
    f["debt_equity"]          = fund.get("debt_equity")
    f["insider_buying_30d"]   = 0

    # News — fixed neutral (cannot replay sentiment historically)
    f["finbert_score"]         = 0.0
    f["news_volume_7d"]        = 5
    f["threat_level_encoded"]  = 0

    # Macro
    f["vix"]                  = vix if vix is not None else FEATURE_DEFAULTS["vix"]
    f["yield_spread_10y2y"]   = ys  if ys  is not None else FEATURE_DEFAULTS["yield_spread_10y2y"]
    spy_c   = spy_row.get("Close", 400.0)
    spy_200 = spy_row.get("sma200")
    f["spy_above_200ma"]      = int(spy_c > spy_200) if spy_200 else 1
    f["inverted_yield_curve"] = int(ys < 0) if ys is not None else 0
    f["fear_greed_encoded"]   = 2  # NEUTRAL proxy

    # Context
    f["regime_encoded"]        = regime
    f["setup_type_encoded"]    = 0   # SETUP_MOMENTUM default
    f["sector_bucket"]         = SECTOR_BUCKETS.get(sym, "OTHER")
    f["mktcap_bucket_encoded"] = MKTCAP_ENCODING.get(
        _mktcap_bucket(fund.get("market_cap")), 3
    )
    f["days_since_earnings"]   = 45  # mid-quarter proxy

    return f
