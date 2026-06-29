"""
ml_backtest.py — ML-Integrated Strategy Backtester

Reconstructs historical feature vectors from yfinance OHLCV and runs the
actual AutoGluon entry/exit models.  Significantly more realistic than
system_backtest.py (RSI-only) because every entry/exit decision goes through
the real models.

Feature coverage
  ✓ Full technical layer  (RSI, MACD, Bollinger, ATR, ADX, Stochastic, SMAs)
  ✓ OHLC geometry         (body, wicks, candle_range_vs_atr)
  ✓ Macro context         (historical VIX, 10Y–2Y yield spread, SPY 200-SMA)
  ✓ Fundamentals          (yfinance snapshot — stable for large-caps year to year)
  ✓ Sector / market-cap   (static lookup from scorer.py SECTOR_BUCKETS)
  ~ Regime                (approximated: SPY vs 200-SMA + monthly momentum)
  ✗ News score            (fixed at 0.0 — can't replay FinBERT historically)
  ✗ Insider / dark-pool   (fixed at neutral)
  ✗ Candlestick patterns  (fixed 0 — no TA-Lib in backtester)

Requirement: AutoGluon must be installed (runs on VPS, not dev machine).
  pip install "autogluon.tabular[lightgbm,xgboost]"

Options simulation (--options):
  When entry model P > 0.82, allocates a portion of the position budget to an
  ATM long call using Black-Scholes pricing.  IV estimated from 30-day realized
  volatility × 1.25 (typical vol risk premium).  Option P&L is tracked
  separately from equity P&L.

Usage:
  # 2yr backtest, 3 symbols
  python -m backtester.ml_backtest --years 2 --symbols NVDA AAPL MSFT

  # Full watchlist, with options simulation
  python -m backtester.ml_backtest --years 2 --full-universe --options

  # 10-day smoke test
  python -m backtester.ml_backtest --dry --symbols NVDA

  # Specific date range
  python -m backtester.ml_backtest --start 2024-01-01 --end 2025-01-01 --symbols NVDA AMD
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="autogluon")

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.feature_builder import (
    _build_indicators,
    _build_feature_row,
    _fetch_fundamentals,
    _mktcap_bucket,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────────

BENCH_SYMBOL   = "SPY"
RISK_FREE_RATE = 0.045           # approximate 10Y yield used for B-S pricing
ENTRY_THRESH   = 0.60            # model P(win) to trigger entry
EXIT_THRESH    = 0.65            # model P(exit) to trigger full sell
MAX_HOLD_DAYS  = 40              # forced exit if model never fires
STOP_LOSS_PCT  = 0.08            # hard stop -8%
POS_SIZE_MAX   = 0.10            # max 10% of portfolio per position
OPTION_THRESH  = 0.82            # min entry_P to also buy calls
OPTION_BUDGET  = 0.40            # fraction of equity position budget used for calls

# Approximate regime from SPY vs 200-SMA + 20-day momentum
def _approx_regime(spy_close: float, spy_sma200: float, spy_mom20: float) -> int:
    """Return regime_encoded (matches scorer.py REGIME_ENCODING)."""
    if spy_close < spy_sma200 * 0.92:
        return 6  # MID_BEAR
    if spy_close < spy_sma200:
        return 5  # EARLY_BEAR
    if spy_mom20 > 0.04:
        return 1  # MID_BULL
    if spy_mom20 > 0.01:
        return 0  # EARLY_BULL
    return 4      # NEUTRAL


# ── Black-Scholes call pricing ────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price via Black-Scholes."""
    from scipy.stats import norm
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# _build_feature_row, _build_indicators, _fetch_fundamentals, _mktcap_bucket
# are imported from backtester.feature_builder at the top of this file.


# ── Portfolio simulation ──────────────────────────────────────────────────────

class _Position:
    __slots__ = ("symbol", "shares", "entry_price", "entry_date",
                 "entry_P", "hold_days", "opt_contracts", "opt_strike",
                 "opt_entry_premium", "opt_dte_at_entry", "opt_sigma_at_entry")

    def __init__(self, symbol, shares, entry_price, entry_date, entry_P):
        self.symbol            = symbol
        self.shares            = shares
        self.entry_price       = entry_price
        self.entry_date        = entry_date
        self.entry_P           = entry_P
        self.hold_days         = 0
        self.opt_contracts     = 0
        self.opt_strike        = 0.0
        self.opt_entry_premium = 0.0
        self.opt_dte_at_entry  = 0
        self.opt_sigma_at_entry= 0.0


class _Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.cash       = initial_cash
        self.initial    = initial_cash
        self.positions: Dict[str, _Position] = {}
        self.trades:    List[dict] = []
        self.equity_curve: List[float] = []

    def value(self, prices: Dict[str, float]) -> float:
        pos_val = sum(
            p.shares * (prices.get(p.symbol) or p.entry_price)
            for p in self.positions.values()
        )
        return self.cash + pos_val

    def enter(
        self,
        sym: str, price: float, date_str: str, entry_P: float,
        portfolio_val: float, simulate_options: bool,
        hist_vol: float,
    ) -> None:
        if sym in self.positions:
            return
        pos_frac = min(POS_SIZE_MAX, max(0.03, entry_P * 0.12))
        budget   = portfolio_val * pos_frac

        # Optionally split budget between equity and calls
        eq_budget = budget
        if simulate_options and entry_P >= OPTION_THRESH:
            eq_budget  = budget * (1 - OPTION_BUDGET)
            opt_budget = budget * OPTION_BUDGET
        else:
            opt_budget = 0.0

        shares = int(eq_budget / price)
        if shares < 1 or eq_budget > self.cash:
            return

        cost = shares * price
        self.cash -= cost

        pos = _Position(sym, shares, price, date_str, entry_P)

        # Options simulation
        if opt_budget > 0 and hist_vol > 0:
            sigma  = hist_vol * 1.25          # vol risk premium
            T      = 30 / 252                  # 30 DTE
            strike = price                     # ATM
            prem   = _bs_call(price, strike, T, RISK_FREE_RATE, sigma)
            if prem > 0.05:
                n_contracts = int(opt_budget / (prem * 100))
                if n_contracts >= 1:
                    opt_cost = n_contracts * prem * 100
                    if opt_cost <= self.cash:
                        self.cash -= opt_cost
                        pos.opt_contracts      = n_contracts
                        pos.opt_strike         = strike
                        pos.opt_entry_premium  = prem
                        pos.opt_dte_at_entry   = 30
                        pos.opt_sigma_at_entry = sigma

        self.positions[sym] = pos
        logger.debug("ENTER  %s  %d sh @ %.2f  P=%.2f", sym, shares, price, entry_P)

    def exit(
        self, sym: str, price: float, date_str: str, reason: str,
        spy_ret: float, hist_vol: float,
    ) -> Optional[dict]:
        pos = self.positions.pop(sym, None)
        if pos is None:
            return None
        price = price or pos.entry_price  # guard: fall back to entry if price missing

        # Equity P&L
        equity_pnl  = (price - pos.entry_price) * pos.shares
        equity_ret  = (price / pos.entry_price) - 1
        self.cash  += pos.shares * price

        # Options P&L
        opt_pnl = 0.0
        if pos.opt_contracts > 0:
            dte_remaining = max(0, pos.opt_dte_at_entry - pos.hold_days)
            T_rem = dte_remaining / 252
            current_sigma = hist_vol * 1.25 if hist_vol > 0 else pos.opt_sigma_at_entry
            exit_prem = _bs_call(price, pos.opt_strike, T_rem,
                                  RISK_FREE_RATE, current_sigma)
            opt_pnl = (exit_prem - pos.opt_entry_premium) * 100 * pos.opt_contracts
            self.cash += max(0.0, exit_prem * 100 * pos.opt_contracts)

        alpha = equity_ret - spy_ret
        trade = {
            "symbol":      sym,
            "entry_price": pos.entry_price,
            "exit_price":  price,
            "shares":      pos.shares,
            "hold_days":   pos.hold_days,
            "equity_pnl":  round(equity_pnl, 2),
            "opt_pnl":     round(opt_pnl, 2),
            "total_pnl":   round(equity_pnl + opt_pnl, 2),
            "equity_ret":  round(equity_ret, 4),
            "alpha":       round(alpha, 4),
            "entry_P":     round(pos.entry_P, 4),
            "entry_date":  pos.entry_date,
            "exit_date":   date_str,
            "reason":      reason,
            "opt_contracts": pos.opt_contracts,
        }
        self.trades.append(trade)
        logger.debug("EXIT   %s  @ %.2f  eq_pnl=%.0f  opt_pnl=%.0f  (%s)",
                     sym, price, equity_pnl, opt_pnl, reason)
        return trade

    def tick(self, prices: Dict[str, float]) -> None:
        """Advance hold_days for each open position (call once per trading day)."""
        for pos in self.positions.values():
            pos.hold_days += 1

    def check_stops(self, prices: Dict[str, float], date_str: str,
                    spy_ret: float, hist_vols: Dict[str, float]) -> None:
        """Hard stop-loss at -STOP_LOSS_PCT."""
        to_exit = []
        for sym, pos in self.positions.items():
            price  = prices.get(sym)
            if price is None:
                continue
            ret = (price / pos.entry_price) - 1
            if ret <= -STOP_LOSS_PCT:
                to_exit.append(sym)
        for sym in to_exit:
            price = prices[sym]
            self.exit(sym, price, date_str, "stop_loss",
                      spy_ret, hist_vols.get(sym, 0.20))

    def check_max_hold(self, prices: Dict[str, float], date_str: str,
                       spy_ret: float, hist_vols: Dict[str, float]) -> None:
        to_exit = [sym for sym, pos in self.positions.items()
                   if pos.hold_days >= MAX_HOLD_DAYS]
        for sym in to_exit:
            price = prices.get(sym, self.positions[sym].entry_price)
            self.exit(sym, price, date_str, "max_hold",
                      spy_ret, hist_vols.get(sym, 0.20))


# ── Main backtest engine ──────────────────────────────────────────────────────

class MLBacktest:
    def __init__(
        self,
        symbols:          List[str],
        years:            int  = 2,
        start_date:       Optional[date] = None,
        end_date:         Optional[date] = None,
        dry_run:          bool = False,
        simulate_options: bool = False,
        initial_cash:     float = 100_000.0,
    ):
        today = date.today()
        self.end_date   = end_date   or today
        self.start_date = start_date or (self.end_date - timedelta(days=int(years * 365.25)))
        self.symbols    = list(dict.fromkeys(symbols))  # deduplicate, keep order
        self.dry_run    = dry_run
        self.simulate_options = simulate_options
        self.initial_cash     = initial_cash

        if dry_run:
            self.end_date   = self.start_date + timedelta(days=14)

    # ── Public ───────────────────────────────────────────────────────────────

    async def run(self) -> dict:
        logger.info("ML Backtest: %d symbols | %s to %s | options=%s",
                    len(self.symbols), self.start_date, self.end_date,
                    self.simulate_options)

        entry_model, exit_model = self._load_models()

        warmup_start = self.start_date - timedelta(days=365)
        all_syms     = self.symbols + [BENCH_SYMBOL, "^VIX", "^TNX", "^IRX"]
        logger.info("Downloading OHLCV data…")
        raw = yf.download(
            [s for s in all_syms if not s.startswith("^")] + ["^VIX", "^TNX", "^IRX"],
            start=str(warmup_start),
            end=str(self.end_date + timedelta(days=1)),
            auto_adjust=True,
            progress=False,
        )

        # Handle MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            data: Dict[str, pd.DataFrame] = {}
            for sym in all_syms:
                try:
                    df = raw.xs(sym, level=1, axis=1).dropna(how="all")
                    if not df.empty:
                        data[sym] = df
                except KeyError:
                    pass
        else:
            data = {list(all_syms)[0]: raw}

        # Build indicators for each equity symbol
        logger.info("Computing technical indicators…")
        ind: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            if sym in data:
                ind[sym] = _build_indicators(data[sym])

        # Build SPY macro features
        spy_df = _build_indicators(data[BENCH_SYMBOL]) if BENCH_SYMBOL in data else pd.DataFrame()
        vix_df = data.get("^VIX", pd.DataFrame())
        tnx_df = data.get("^TNX", pd.DataFrame())
        irx_df = data.get("^IRX", pd.DataFrame())

        # Fetch fundamentals (once per symbol — current snapshot)
        logger.info("Fetching fundamentals snapshots…")
        funds: Dict[str, dict] = {}
        for sym in self.symbols:
            funds[sym] = _fetch_fundamentals(sym)

        # Trading simulation
        portfolio   = _Portfolio(self.initial_cash)
        all_dates   = pd.bdate_range(str(self.start_date), str(self.end_date))
        n_days      = len(all_dates)
        logger.info("Simulating %d trading days…", n_days)

        # Get feature names from model (if available)
        entry_feat_names = None
        if entry_model is not None:
            try:
                entry_feat_names = entry_model.feature_metadata_in.get_features()
            except Exception:
                pass

        exit_feat_names = None
        if exit_model is not None:
            try:
                exit_feat_names = exit_model.feature_metadata_in.get_features()
            except Exception:
                pass

        for i, dt in enumerate(all_dates):
            dt_str  = dt.strftime("%Y-%m-%d")
            spy_row = self._get_row(spy_df, dt)
            spy_ret = self._get_1d_ret(spy_df, dt)

            vix_val = self._get_scalar(vix_df, dt, "Close")
            tnx_val = self._get_scalar(tnx_df, dt, "Close")
            irx_val = self._get_scalar(irx_df, dt, "Close")
            ys_val  = (tnx_val - irx_val) if (tnx_val and irx_val) else None

            regime  = _approx_regime(
                spy_row.get("Close", 400.0),
                spy_row.get("sma200", 380.0),
                self._get_mom20(spy_df, dt),
            )

            prices = {
                sym: self._get_scalar(ind.get(sym, pd.DataFrame()), dt, "Close")
                for sym in self.symbols
            }

            hist_vols = {
                sym: self._get_vol30(ind.get(sym, pd.DataFrame()), dt)
                for sym in self.symbols
            }

            if i % 50 == 0:
                logger.info("Day %d/%d  %s  positions=%d  cash=%.0f",
                            i, n_days, dt_str, len(portfolio.positions), portfolio.cash)

            # Advance hold counters
            portfolio.tick(prices)

            # Stop-loss check
            portfolio.check_stops(prices, dt_str, spy_ret, hist_vols)

            # Max-hold forced exit
            portfolio.check_max_hold(prices, dt_str, spy_ret, hist_vols)

            # ── Build all feature vectors for the day in one pass ──────────────
            sym_fvs: Dict[str, dict] = {}
            for sym in self.symbols:
                if not prices.get(sym):
                    continue
                row = self._get_row(ind.get(sym, pd.DataFrame()), dt)
                if row.empty:
                    continue
                sym_fvs[sym] = _build_feature_row(
                    sym, row, funds[sym], vix_val, ys_val, spy_row, regime
                )

            # ── Batch exit: one predict_proba call for all held positions ──────
            if exit_model is not None and portfolio.positions:
                held = [s for s in portfolio.positions if s in sym_fvs]
                if held:
                    exit_probs = self._batch_predict(
                        exit_model, [sym_fvs[s] for s in held], exit_feat_names
                    )
                    for sym, exit_P in zip(held, exit_probs):
                        if exit_P is not None and exit_P >= EXIT_THRESH:
                            portfolio.exit(sym, prices[sym], dt_str, "model_exit",
                                           spy_ret, hist_vols.get(sym, 0.20))

            # ── Batch entry: one predict_proba call for all candidates ─────────
            if entry_model is not None:
                pv   = portfolio.value(prices)
                cands = [s for s in self.symbols
                         if s not in portfolio.positions and s in sym_fvs]
                if cands:
                    entry_probs = self._batch_predict(
                        entry_model, [sym_fvs[s] for s in cands], entry_feat_names
                    )
                    for sym, entry_P in zip(cands, entry_probs):
                        if entry_P is not None and entry_P >= ENTRY_THRESH:
                            portfolio.enter(sym, prices[sym], dt_str, entry_P, pv,
                                            self.simulate_options, hist_vols.get(sym, 0.20))

            portfolio.equity_curve.append(portfolio.value(prices))

        # Flush open positions at end
        last_dt = all_dates[-1]
        spy_ret = self._get_1d_ret(spy_df, last_dt)
        for sym in list(portfolio.positions.keys()):
            price = prices.get(sym) or portfolio.positions[sym].entry_price
            portfolio.exit(sym, price, last_dt.strftime("%Y-%m-%d"),
                           "sim_end", spy_ret, hist_vols.get(sym, 0.20))

        return self._compile_report(portfolio)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _load_models(self):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            logger.error(
                "AutoGluon not installed.  Run on VPS:\n"
                "  pip install 'autogluon.tabular[lightgbm,xgboost]'"
            )
            sys.exit(1)

        entry, exit_ = None, None
        entry_path = Path("models/autogluon_entry")
        exit_path  = Path("models/autogluon_exit")

        if entry_path.exists():
            logger.info("Loading entry model…")
            entry = TabularPredictor.load(str(entry_path), verbosity=0)
        else:
            logger.warning("Entry model not found at %s — entry signals disabled", entry_path)

        if exit_path.exists():
            logger.info("Loading exit model…")
            exit_ = TabularPredictor.load(str(exit_path), verbosity=0)
        else:
            logger.warning("Exit model not found at %s — exit signals disabled", exit_path)

        return entry, exit_

    @staticmethod
    def _get_row(df: pd.DataFrame, dt) -> pd.Series:
        try:
            idx = df.index[df.index.normalize() == pd.Timestamp(dt).normalize()]
            if idx.empty:
                return pd.Series(dtype=float)
            return df.loc[idx[0]]
        except Exception:
            return pd.Series(dtype=float)

    @staticmethod
    def _get_scalar(df: pd.DataFrame, dt, col: str) -> Optional[float]:
        row = MLBacktest._get_row(df, dt)
        val = row.get(col) if not row.empty else None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    @staticmethod
    def _get_1d_ret(df: pd.DataFrame, dt) -> float:
        try:
            row = MLBacktest._get_row(df, dt)
            if row.empty:
                return 0.0
            c = row.get("Close")
            p = row.get("prev_close")
            if c and p and p > 0:
                return float(c / p - 1)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _get_mom20(df: pd.DataFrame, dt) -> float:
        try:
            row = MLBacktest._get_row(df, dt)
            return float(row.get("price_vs_20ma", 0.0) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _get_vol30(df: pd.DataFrame, dt) -> float:
        try:
            row = MLBacktest._get_row(df, dt)
            atr_pct = row.get("atr_pct")
            if atr_pct and not np.isnan(atr_pct):
                return float(atr_pct) * np.sqrt(252)
        except Exception:
            pass
        return 0.25

    @staticmethod
    def _batch_predict(
        model, fv_list: List[dict], feat_names: Optional[List[str]]
    ) -> List[Optional[float]]:
        """Single predict_proba call for N feature vectors — avoids per-row overhead."""
        if not fv_list:
            return []
        try:
            df = pd.DataFrame(fv_list)
            if feat_names:
                for f in feat_names:
                    if f not in df.columns:
                        df[f] = None
                df = df[feat_names]
            proba = model.predict_proba(df)
            if hasattr(proba, "iloc"):
                cols    = proba.columns.tolist()
                pos_col = 1 if 1 in cols else True if True in cols else cols[-1]
                return [float(proba.iloc[j][pos_col]) for j in range(len(proba))]
            return [float(p) for p in proba]
        except Exception as e:
            logger.debug("batch_predict failed: %s", e)
            return [None] * len(fv_list)

    @staticmethod
    def _predict_proba(
        model, fv: dict, feat_names: Optional[List[str]]
    ) -> Optional[float]:
        """Single-row wrapper — kept for compatibility, not used in main loop."""
        result = MLBacktest._batch_predict(model, [fv], feat_names)
        return result[0] if result else None

    def _compile_report(self, portfolio: _Portfolio) -> dict:
        trades = portfolio.trades
        if not trades:
            return {"error": "no_trades", "n_trades": 0}

        wins      = [t for t in trades if t["equity_ret"] > 0]
        win_rate  = len(wins) / len(trades)
        avg_alpha = float(np.mean([t["alpha"] for t in trades]))
        total_eq  = sum(t["equity_pnl"] for t in trades)
        total_opt = sum(t["opt_pnl"] for t in trades)
        total_ret = (portfolio.cash + sum(
            t["shares"] * t["exit_price"] for t in trades if "exit_price" in t
        ) - portfolio.initial) / portfolio.initial if trades else 0.0

        # Use equity curve for Sharpe / drawdown
        curve = np.array(portfolio.equity_curve)
        if len(curve) > 1:
            daily_rets = np.diff(curve) / curve[:-1]
            sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-9) * np.sqrt(252))
            peak   = np.maximum.accumulate(curve)
            max_dd = float(np.min((curve - peak) / (peak + 1e-9)))
        else:
            sharpe = 0.0
            max_dd = 0.0

        total_pnl = sum(t["total_pnl"] for t in trades)
        total_pct = total_pnl / portfolio.initial

        opt_trades = [t for t in trades if t["opt_contracts"] > 0]
        report = {
            "run_date":           datetime.now().isoformat(),
            "symbols":            self.symbols,
            "start_date":         str(self.start_date),
            "end_date":           str(self.end_date),
            "simulate_options":   self.simulate_options,
            "n_trades":           len(trades),
            "win_rate":           round(win_rate, 4),
            "avg_alpha":          round(avg_alpha, 4),
            "total_equity_pnl":   round(total_eq, 2),
            "total_options_pnl":  round(total_opt, 2),
            "total_pnl":          round(total_pnl, 2),
            "total_return":       round(total_pct, 4),
            "sharpe":             round(sharpe, 3),
            "max_drawdown":       round(max_dd, 4),
            "n_option_trades":    len(opt_trades),
            "avg_hold_days":      round(float(np.mean([t["hold_days"] for t in trades])), 1),
        }

        # Save to logs/
        Path("logs").mkdir(exist_ok=True)
        ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix  = "_options" if self.simulate_options else ""
        j_path  = Path(f"logs/ml_backtest{suffix}_{ts_str}.json")
        j_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        summary = [
            f"ML Backtest Results {'(options)' if self.simulate_options else ''}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Symbols: {len(self.symbols)}",
            "─" * 44,
            f"Trades:          {len(trades)}",
            f"Win rate:        {win_rate:.1%}",
            f"Avg alpha:       {avg_alpha:+.2%}",
            f"Total return:    {total_pct:+.2%}",
            f"Sharpe:          {sharpe:.3f}",
            f"Max drawdown:    {max_dd:.1%}",
            f"Avg hold days:   {report['avg_hold_days']}",
        ]
        if self.simulate_options and opt_trades:
            summary += [
                "─" * 44,
                f"Option trades:   {len(opt_trades)}",
                f"Equity P&L:      ${total_eq:,.0f}",
                f"Options P&L:     ${total_opt:,.0f}",
                f"Total P&L:       ${total_pnl:,.0f}",
            ]

        print("\n".join(summary))
        t_path = Path(f"logs/ml_backtest{suffix}_{ts_str}.txt")
        t_path.write_text("\n".join(summary), encoding="utf-8")
        logger.info("Results saved → %s", j_path)
        return report


async def main():
    parser = argparse.ArgumentParser(description="ML-integrated strategy backtester")
    parser.add_argument("--years",        type=int,   default=2)
    parser.add_argument("--start",        type=str,   default=None)
    parser.add_argument("--end",          type=str,   default=None)
    parser.add_argument("--symbols",      nargs="+",  default=["NVDA", "AAPL", "MSFT"])
    parser.add_argument("--full-universe",action="store_true")
    parser.add_argument("--oos",          action="store_true", help="Use config.json oos_test_watchlist (symbols never seen during training)")
    parser.add_argument("--options",      action="store_true", help="Simulate long call positions")
    parser.add_argument("--dry",          action="store_true", help="14-day smoke test")
    parser.add_argument("--cash",         type=float, default=100_000.0)
    args = parser.parse_args()

    if args.oos:
        cfg     = json.loads(Path("config.json").read_text())
        symbols = cfg.get("oos_test_watchlist", {}).get("symbols", [])
        if not symbols:
            logger.error("oos_test_watchlist.symbols is empty in config.json")
            sys.exit(1)
        logger.info("OOS test mode: %d symbols (out-of-sample, not used during training)", len(symbols))
    elif args.full_universe:
        cfg = json.loads(Path("config.json").read_text())
        wl  = cfg.get("watchlist", {})
        symbols = (wl.get("mandatory", []) + wl.get("screener_picks", [])
                   + wl.get("manual_additions", [])) or ["NVDA", "AAPL"]
    else:
        symbols = args.symbols

    start = date.fromisoformat(args.start) if args.start else None
    end   = date.fromisoformat(args.end)   if args.end   else None

    bt = MLBacktest(
        symbols=symbols,
        years=args.years,
        start_date=start,
        end_date=end,
        dry_run=args.dry,
        simulate_options=args.options,
        initial_cash=args.cash,
    )
    result = await bt.run()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
