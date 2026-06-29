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
    df["rsi_oversold"]        = (rsi14 < 30).astype(int)
    df["rsi_overbought"]      = (rsi14 > 70).astype(int)
    df["rsi_momentum_zone"]   = ((rsi14 >= 40) & (rsi14 <= 60)).astype(int)
    df["macd_bullish"]        = (macd_line > macd_sig).astype(int)
    df["macd_bullish_cross"]  = ((macd_line > macd_sig) & (macd_line.shift() <= macd_sig.shift())).astype(int)
    df["macd_bearish_cross"]  = ((macd_line < macd_sig) & (macd_line.shift() >= macd_sig.shift())).astype(int)
    df["volume_surge"]        = (df["volume_ratio"] > 2.0).astype(int)
    df["volume_confirming"]   = ((df["volume_ratio"] > 1.3) & (c > c.shift())).astype(int)
    df["volume_distribution"] = ((df["volume_ratio"] > 1.3) & (c < c.shift())).astype(int)
    df["bb_squeeze"]          = (bb_width < bb_width.rolling(20).mean() * 0.7).astype(int)
    df["adx_strong_trend"]    = (adx > 25).astype(int)
    df["daily_trend_bullish"] = (c > c.shift()).astype(int)
    df["stoch_oversold"]      = (stoch_k < 20).astype(int)
    df["stoch_overbought"]    = (stoch_k > 80).astype(int)
    df["gap_up"]              = (df["Open"] > c.shift() * 1.01).astype(int)
    df["gap_down"]            = (df["Open"] < c.shift() * 0.99).astype(int)

    body       = (c - df["Open"]).abs()
    candle_rng = h - l
    df["body_size"]           = body / c.replace(0, np.nan)
    df["body_direction"]      = np.sign(c - df["Open"])
    df["upper_wick"]          = (h - c.clip(lower=df["Open"])) / c.replace(0, np.nan)
    df["lower_wick"]          = (c.clip(upper=df["Open"]) - l) / c.replace(0, np.nan)
    df["wick_ratio"]          = (h - l - body) / body.replace(0, np.nan)
    df["candle_range_vs_atr"] = candle_rng / atr14.replace(0, np.nan)

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

    # Technical continuous + binary
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
    ):
        val = row.get(col)
        f[col] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    # Candlestick patterns — no TA-Lib in backtester, fixed to 0
    f["hammer_at_200ma"]          = 0
    f["hammer_at_50ma"]           = 0
    f["engulfing_at_support"]     = 0
    f["marubozu_on_volume_surge"] = 0
    f["morning_star_at_support"]  = 0
    f["evening_star_near_highs"]  = 0

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
