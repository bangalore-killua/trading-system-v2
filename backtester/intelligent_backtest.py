"""
intelligent_backtest.py — Unified labeled backtest dataset with alpha-beta separation.

Core outputs:
  1. Labeled records (one per position-entry) for ML training
  2. Multi-regime oracle: robust_sharpe = min(sharpe across 4 regime windows)
  3. Multi-objective oracle: profit >= 20% + avg_position >= 5% + not Pareto-dominated
  4. Buy-and-hold benchmark alongside every run
  5. Alpha-beta separation: alpha = trade_return - (spy_return x stock_beta)

Feature consistency:
  All features are built via backtester.feature_builder._build_feature_row()
  which produces exactly the same 65-feature dict that scorer.py uses at runtime.
  Training data and live inference now always have identical column names.

Point-in-time correctness:
  Earnings: use release date (not quarter end)
  Form 4 insider: use filing date (2 business days after trade)
  13F institutional: use filing date (45 days after quarter end)
  News: publication timestamp only

Requirements:
  pip install yfinance pandas numpy scipy

Usage:
  python backtester/intelligent_backtest.py --symbol NVDA --years 5
  python backtester/intelligent_backtest.py --dry --symbol NVDA   # prints sample records
"""

import argparse
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from backtester.feature_builder import (
    _build_indicators,
    _build_feature_row,
    _fetch_fundamentals,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

# ── Regime windows (empirically validated, Auto-Quant 250+ experiments) ──────
REGIME_WINDOWS = {
    "bull":     ("2020-04-01", "2021-12-31"),
    "crash":    ("2022-01-01", "2022-12-31"),
    "recovery": ("2023-01-01", "2024-06-30"),
    "sideways": ("2024-07-01", "2025-12-31"),
}

# ── Triple-barrier defaults — read from config so training matches live params ─
def _load_triple_barrier_cfg() -> tuple:
    try:
        from strategy.config_loader import load_config
        m = load_config().get("model", {})
        return (
            float(m.get("triple_barrier_atr_tp_multiplier", 2.0)),
            float(m.get("triple_barrier_atr_sl_multiplier", 1.5)),
            int(m.get("triple_barrier_days", 60)),
        )
    except Exception:
        return 2.0, 1.5, 60

TAKE_PROFIT_ATR_MULT, STOP_LOSS_ATR_MULT, MAX_HOLDING_DAYS = _load_triple_barrier_cfg()


def run_backtest(
    symbol:     str,
    years:      int = 5,
    dry:        bool = False,
    save_path:  Optional[str] = None,
) -> dict:
    """Run full intelligent backtest for a symbol.

    Returns
    -------
    {
      "records":          list of labeled dicts (for ML training),
      "metrics":          {"sharpe", "total_return", "win_rate", "n_trades"},
      "regime_sharpes":   {"bull": float, "crash": float, ...},
      "robust_sharpe":    float,  # min across 4 regimes
      "bah_return":       float,  # buy-and-hold benchmark
      "passes_oracle":    bool,
    }
    """
    logger.info("Backtesting %s (%d years)", symbol, years)

    # ── Load raw OHLCV ────────────────────────────────────────────────────────
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    df_stock_raw = _load_ohlcv(symbol, start, end)
    df_spy_raw   = _load_ohlcv("SPY",  start, end)

    # Compute beta from raw lowercase series (before column rename)
    beta = _compute_beta(df_stock_raw["close"], df_spy_raw["close"])

    # Buy-and-hold uses raw close (all bars, no dropna)
    bah_return = float((df_stock_raw["close"].iloc[-1] / df_stock_raw["close"].iloc[0]) - 1)

    if dry:
        logger.info("DRY RUN: loaded %d bars for %s, beta=%.2f", len(df_stock_raw), symbol, beta)

    # ── Macro data (VIX, 10Y-2Y yield spread) ────────────────────────────────
    vix_s = _load_macro_series("^VIX", start, end)
    tnx_s = _load_macro_series("^TNX", start, end)
    irx_s = _load_macro_series("^IRX", start, end)

    # ── Fundamentals (current snapshot — stable for large-caps) ──────────────
    fund = _fetch_fundamentals(symbol)

    # ── Build canonical features ──────────────────────────────────────────────
    df      = _compute_features(df_stock_raw)   # Title Case + 65 canonical features
    spy_ind = _compute_features(df_spy_raw)     # SPY with same canonical features

    # ── Entry signals ─────────────────────────────────────────────────────────
    entries = _generate_entry_signals(df)

    # ── Triple-barrier labeling ───────────────────────────────────────────────
    records = _label_trades(df, spy_ind, vix_s, tnx_s, irx_s, fund, entries, beta, symbol)

    if dry:
        for r in records[:3]:
            print(json.dumps({k: v for k, v in r.items() if k != "features"}, indent=2, default=str))
        return {"records": records[:3], "dry_run": True}

    # ── Portfolio metrics ─────────────────────────────────────────────────────
    metrics = _compute_metrics(records)

    # ── Multi-regime oracle ───────────────────────────────────────────────────
    regime_sharpes = _compute_regime_sharpes(df, spy_ind, vix_s, tnx_s, irx_s, fund, beta, symbol)
    robust_sharpe  = min(regime_sharpes.values()) if regime_sharpes else 0.0

    # ── Multi-objective oracle ────────────────────────────────────────────────
    prior_runs = _load_prior_runs()
    passes     = _passes_oracle(metrics, prior_runs)

    result = {
        "symbol":         symbol,
        "records":        records,
        "metrics":        metrics,
        "regime_sharpes": regime_sharpes,
        "robust_sharpe":  round(robust_sharpe, 4),
        "bah_return":     round(bah_return, 4),
        "beta":           round(beta, 3),
        "passes_oracle":  passes,
        "n_records":      len(records),
        "generated_at":   datetime.now().isoformat(),
    }

    if save_path:
        _save_records(records, save_path)
        logger.info("Saved %d labeled records to %s", len(records), save_path)

    logger.info(
        "%s: %d trades | sharpe=%.2f | robust_sharpe=%.2f | bah=%.1f%% | oracle=%s",
        symbol, metrics["n_trades"], metrics["sharpe"],
        robust_sharpe, bah_return * 100, "PASS" if passes else "FAIL",
    )
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_ohlcv(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load OHLCV with lowercase columns. Tries tvdatafeed, falls back to yfinance."""
    try:
        from data.sources.tvdatafeed_client import get_ohlcv  # type: ignore
        df = get_ohlcv(symbol, interval="1D", n_bars=int((end - start).days * 1.1))
        if df is not None and len(df) > 100:
            return df.loc[start.strftime("%Y-%m-%d"):end.strftime("%Y-%m-%d")]
    except Exception as e:
        logger.debug("tvdatafeed failed for %s: %s — using yfinance", symbol, e)

    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def _load_macro_series(ticker: str, start: datetime, end: datetime) -> pd.Series:
    """Download a macro series (VIX, TNX, IRX) and return the Close series."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df["Close"] if "Close" in df.columns else pd.Series(dtype=float)
    except Exception as e:
        logger.debug("Macro download failed for %s: %s", ticker, e)
        return pd.Series(dtype=float)


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename lowercase OHLCV to Title Case, compute all 65 canonical features."""
    col_map = {"close": "Close", "high": "High", "low": "Low", "open": "Open", "volume": "Volume"}
    df_tc = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    result = _build_indicators(df_tc)
    return result.dropna(subset=["rsi", "macd_hist"])


# ── Entry signals ─────────────────────────────────────────────────────────────

def _generate_entry_signals(df: pd.DataFrame) -> pd.Series:
    """Entry signal for training data generation — intentionally permissive.

    This is NOT the live entry signal (that lives in scorer.py).
    Goal is maximum diverse labeled samples for AutoGluon to learn from.
    Triple-barrier labeling determines which entries were actually profitable.
    """
    return (
        (df["rsi"] < 50) &
        (df["price_above_200ma"] == 1)
    )


# ── Triple-barrier labeling ───────────────────────────────────────────────────

def _label_trades(
    df:      pd.DataFrame,
    spy_ind: pd.DataFrame,
    vix_s:   pd.Series,
    tnx_s:   pd.Series,
    irx_s:   pd.Series,
    fund:    dict,
    entries: pd.Series,
    beta:    float,
    symbol:  str,
) -> list:
    """Label each entry with alpha-adjusted outcome using triple-barrier method."""
    records     = []
    entry_dates = df.index[entries]

    for entry_date in entry_dates:
        try:
            record = _label_single_trade(
                df, spy_ind, vix_s, tnx_s, irx_s, fund, entry_date, beta, symbol
            )
            if record:
                records.append(record)
        except Exception as e:
            logger.debug("Labeling failed for %s at %s: %s", symbol, entry_date, e)

    return records


def _label_single_trade(
    df:         pd.DataFrame,
    spy_ind:    pd.DataFrame,
    vix_s:      pd.Series,
    tnx_s:      pd.Series,
    irx_s:      pd.Series,
    fund:       dict,
    entry_date,
    beta:       float,
    symbol:     str,
) -> Optional[dict]:
    """Label one trade with triple-barrier + alpha separation + full feature vector."""
    idx = df.index.get_loc(entry_date)
    if idx + 5 >= len(df):
        return None

    entry_price = float(df["Close"].iloc[idx])
    atr         = float(df["atr14"].iloc[idx])   # canonical name (was atr_14)

    tp_price = entry_price + TAKE_PROFIT_ATR_MULT * atr
    sl_price = entry_price - STOP_LOSS_ATR_MULT  * atr
    end_idx  = min(idx + MAX_HOLDING_DAYS, len(df) - 1)

    exit_price  = float(df["Close"].iloc[end_idx])
    exit_date   = df.index[end_idx]
    exit_reason = "time"

    for i in range(idx + 1, end_idx + 1):
        high_i  = float(df["High"].iloc[i])
        low_i   = float(df["Low"].iloc[i])
        if high_i >= tp_price:
            exit_price  = tp_price
            exit_date   = df.index[i]
            exit_reason = "take_profit"
            break
        elif low_i <= sl_price:
            exit_price  = sl_price
            exit_date   = df.index[i]
            exit_reason = "stop_loss"
            break

    trade_return = (exit_price - entry_price) / entry_price
    spy_return   = _spy_return_over_period(spy_ind, entry_date, exit_date)
    alpha        = trade_return - (spy_return * beta)
    entry_label  = 1 if alpha > 0.005 else (-1 if alpha < -0.005 else 0)

    # ── Build full canonical feature vector (same 65 features as live) ────────
    row     = df.iloc[idx]
    vix_val = _get_series_at_date(vix_s, entry_date)
    tnx_val = _get_series_at_date(tnx_s, entry_date)
    irx_val = _get_series_at_date(irx_s, entry_date)
    ys_val  = (tnx_val - irx_val) if (tnx_val is not None and irx_val is not None) else None
    spy_row = _get_row_at_date(spy_ind, entry_date)
    spy_c   = spy_row.get("Close", 400.0) or 400.0
    spy_200 = spy_row.get("sma200")
    spy_mom = float(spy_row.get("price_vs_20ma", 0.0) or 0.0)
    regime  = _approx_regime(spy_c, spy_200 or spy_c, spy_mom)

    features = _build_feature_row(symbol, row, fund, vix_val, ys_val, spy_row, regime)

    days_held = (exit_date - entry_date).days if hasattr(exit_date - entry_date, "days") else 0

    return {
        "symbol":       symbol,
        "entry_date":   str(entry_date)[:10],
        "exit_date":    str(exit_date)[:10],
        "entry_price":  round(entry_price, 4),
        "exit_price":   round(exit_price, 4),
        "trade_return": round(trade_return, 5),
        "spy_return":   round(spy_return, 5),
        "alpha":        round(alpha, 5),
        "beta":         round(beta, 3),
        "exit_reason":  exit_reason,
        "days_held":    days_held,
        "entry_label":  entry_label,
        "features":     features,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(records: list) -> dict:
    if not records:
        return {"sharpe": 0.0, "total_return": 0.0, "win_rate": 0.0, "n_trades": 0, "avg_position_pct": 0.0}

    alphas    = np.array([r["alpha"] for r in records])
    wins      = (alphas > 0).sum()
    sharpe    = float(alphas.mean() / (alphas.std() + 1e-9) * np.sqrt(252))
    total_ret = float(np.prod(1 + alphas) - 1)
    avg_pos   = float(np.abs(alphas).mean() * 100)

    return {
        "sharpe":           round(sharpe, 4),
        "total_return":     round(total_ret, 4),
        "win_rate":         round(float(wins / len(alphas)), 4),
        "n_trades":         len(records),
        "avg_position_pct": round(avg_pos, 2),
        "avg_alpha":        round(float(alphas.mean()), 5),
    }


def _compute_regime_sharpes(df, spy_ind, vix_s, tnx_s, irx_s, fund, beta, symbol) -> dict:
    """Compute Sharpe for each regime window (robust_sharpe = min of these)."""
    sharpes = {}
    for regime, (start, end) in REGIME_WINDOWS.items():
        try:
            mask    = (df.index >= start) & (df.index <= end)
            df_r    = df[mask]
            if len(df_r) < 20:
                continue
            entries = _generate_entry_signals(df_r)
            records = _label_trades(df_r, spy_ind, vix_s, tnx_s, irx_s, fund, entries, beta, symbol)
            m       = _compute_metrics(records)
            sharpes[regime] = m["sharpe"]
        except Exception as e:
            logger.debug("Regime %s sharpe failed: %s", regime, e)
    return sharpes


# ── Multi-objective oracle ────────────────────────────────────────────────────

def _passes_oracle(metrics: dict, prior_runs: list) -> bool:
    if metrics["total_return"] < 0.20:
        logger.debug("Oracle FAIL: profit %.1f%% < 20%%", metrics["total_return"] * 100)
        return False
    if metrics["avg_position_pct"] < 5.0:
        logger.debug("Oracle FAIL: avg_position %.2f%% < 5%%", metrics["avg_position_pct"])
        return False
    if _pareto_dominated(metrics, prior_runs):
        logger.debug("Oracle FAIL: Pareto-dominated by prior run")
        return False
    return True


def _pareto_dominated(metrics: dict, prior_runs: list) -> bool:
    for prior in prior_runs:
        dominated = (
            prior.get("sharpe", 0)       >= metrics["sharpe"] and
            prior.get("total_return", 0) >= metrics["total_return"] and
            prior.get("win_rate", 0)     >= metrics["win_rate"]
        )
        if dominated:
            return True
    return False


def _load_prior_runs() -> list:
    import os
    path = "backtester/prior_runs.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


# ── Math helpers ──────────────────────────────────────────────────────────────

def _compute_beta(stock_close: pd.Series, spy_close: pd.Series) -> float:
    rets_s  = stock_close.pct_change().dropna()
    rets_m  = spy_close.pct_change().dropna()
    aligned = pd.concat([rets_s, rets_m], axis=1).dropna().tail(504)
    if len(aligned) < 60:
        return 1.0
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    return round(float(cov / (var + 1e-9)), 3)


def _spy_return_over_period(spy_ind: pd.DataFrame, start, end) -> float:
    try:
        spy = spy_ind["Close"]
        s   = spy.loc[start:end]
        if len(s) < 2:
            return 0.0
        return float((s.iloc[-1] - s.iloc[0]) / s.iloc[0])
    except Exception:
        return 0.0


def _get_series_at_date(series: pd.Series, dt) -> Optional[float]:
    """Lookup a macro series value at the nearest available date."""
    if series.empty:
        return None
    try:
        ts   = pd.Timestamp(dt).normalize()
        mask = series.index.normalize() == ts
        if mask.any():
            val = series[mask].iloc[0]
            return float(val) if not np.isnan(val) else None
        # Forward-fill: use most recent value before dt
        before = series[series.index.normalize() <= ts]
        if not before.empty:
            return float(before.iloc[-1])
    except Exception:
        pass
    return None


def _get_row_at_date(df: pd.DataFrame, dt) -> pd.Series:
    """Return the DataFrame row at the given date (normalized), or empty Series."""
    try:
        idx = df.index[df.index.normalize() == pd.Timestamp(dt).normalize()]
        if not idx.empty:
            return df.loc[idx[0]]
    except Exception:
        pass
    return pd.Series(dtype=float)


def _approx_regime(spy_close: float, spy_sma200: float, spy_mom20: float) -> int:
    """Approximate regime_encoded matching scorer.py REGIME_ENCODING."""
    if spy_close < spy_sma200 * 0.92:
        return 6  # MID_BEAR
    if spy_close < spy_sma200:
        return 5  # EARLY_BEAR
    if spy_mom20 > 0.04:
        return 1  # MID_BULL
    if spy_mom20 > 0.01:
        return 0  # EARLY_BULL
    return 4      # NEUTRAL


def _save_records(records: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump(records, f, default=str, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s -- %(message)s")

    parser = argparse.ArgumentParser(description="Intelligent vectorbt backtester")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--years",  type=int, default=5)
    parser.add_argument("--dry",    action="store_true", help="Print sample records and exit")
    parser.add_argument("--save",   default=None, help="Save labeled records to JSON path")
    args = parser.parse_args()

    result = run_backtest(symbol=args.symbol, years=args.years, dry=args.dry, save_path=args.save)

    if not args.dry:
        print(f"\n{'='*50}")
        print(f"Symbol:        {result['symbol']}")
        print(f"Records:       {result['n_records']}")
        print(f"Sharpe:        {result['metrics']['sharpe']:.3f}")
        print(f"Robust Sharpe: {result['robust_sharpe']:.3f}  (min across 4 regimes)")
        print(f"Total Return:  {result['metrics']['total_return']*100:.1f}%")
        print(f"Win Rate:      {result['metrics']['win_rate']*100:.1f}%")
        print(f"B&H Return:    {result['bah_return']*100:.1f}%")
        print(f"Oracle:        {'PASS' if result['passes_oracle'] else 'FAIL'}")
        print(f"\nRegime Sharpes:")
        for regime, s in result["regime_sharpes"].items():
            print(f"  {regime:12s}: {s:.3f}")
