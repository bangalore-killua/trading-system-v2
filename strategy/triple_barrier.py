"""
triple_barrier.py — Training Label Generator (Marcos López de Prado, AFML Ch. 3)

Applies the triple-barrier method to a price series to generate trade outcome labels
for LightGBM training. Run this on historical OHLCV data to bootstrap Phase 3 before
enough live paper trades accumulate.

Three barriers per trade entry:
  Upper  (take-profit) → entry_price × (1 + ATR × tp_multiplier)   → label +1
  Lower  (stop-loss)   → entry_price × (1 − ATR × sl_multiplier)   → label -1
  Vertical (time)      → t + max_holding_days bars                  → label  0

The first barrier touched determines the label.

Usage — retroactive bootstrap:
    from data.fetcher import OHLCVFetcher
    from strategy.triple_barrier import TripleBarrierLabeler

    fetcher = OHLCVFetcher()
    df = await fetcher.fetch("AAPL", bars=1260)   # 5 years
    labeler = TripleBarrierLabeler()
    labeled = labeler.label(df)
    # labeled: DataFrame with columns [entry_date, label, return_pct, barrier_hit, days_held]

Usage — single live trade:
    label = labeler.label_one(df, entry_idx=200, entry_price=150.0)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BarrierResult:
    entry_date: pd.Timestamp
    entry_price: float
    tp_price: float
    sl_price: float
    exit_date: Optional[pd.Timestamp]
    exit_price: Optional[float]
    label: int            # +1 / -1 / 0
    barrier_hit: str      # "tp" / "sl" / "time" / "open"
    return_pct: float
    days_held: int
    atr_at_entry: float


class TripleBarrierLabeler:
    """Applies triple-barrier labeling to an OHLCV DataFrame.

    Parameters
    ----------
    tp_multiplier : float
        Take-profit distance as a multiple of ATR. Default 3.0 (3× ATR).
    sl_multiplier : float
        Stop-loss distance as a multiple of ATR. Default 1.5 (1.5× ATR).
    max_holding_days : int
        Vertical barrier — max bars before time-exit. Default 10.
    atr_period : int
        ATR lookback for barrier width. Default 14.
    min_atr_pct : float
        Minimum ATR as % of price. Prevents degenerate barriers on low-vol data.
    """

    def __init__(
        self,
        tp_multiplier: float = 3.0,
        sl_multiplier: float = 1.5,
        max_holding_days: int = 10,
        atr_period: int = 14,
        min_atr_pct: float = 0.005,
    ):
        self.tp_mult = tp_multiplier
        self.sl_mult = sl_multiplier
        self.max_hold = max_holding_days
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct

    def label(self, df: pd.DataFrame, min_bars: int = 50) -> pd.DataFrame:
        """Label every bar in df as a potential trade entry.

        Returns a DataFrame indexed by entry date with columns:
          entry_price, tp_price, sl_price, exit_date, exit_price,
          label, barrier_hit, return_pct, days_held, atr_at_entry
        """
        df = self._validate(df)
        if len(df) < min_bars:
            logger.warning("DataFrame too short (%d bars < %d minimum)", len(df), min_bars)
            return pd.DataFrame()

        atr = self._compute_atr(df)
        results = []

        # Skip early bars where ATR is NaN and leave trailing bars with no room
        for i in range(self.atr_period, len(df) - 1):
            entry_price = df["close"].iloc[i]
            atr_val = atr.iloc[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # Enforce minimum ATR floor
            atr_val = max(atr_val, entry_price * self.min_atr_pct)

            result = self._apply_barriers(df, i, entry_price, atr_val)
            if result is not None:
                results.append(result)

        if not results:
            return pd.DataFrame()

        out = pd.DataFrame([
            {
                "entry_date":    r.entry_date,
                "entry_price":   round(r.entry_price, 4),
                "tp_price":      round(r.tp_price, 4),
                "sl_price":      round(r.sl_price, 4),
                "exit_date":     r.exit_date,
                "exit_price":    round(r.exit_price, 4) if r.exit_price else None,
                "label":         r.label,
                "barrier_hit":   r.barrier_hit,
                "return_pct":    round(r.return_pct, 4),
                "days_held":     r.days_held,
                "atr_at_entry":  round(r.atr_at_entry, 4),
            }
            for r in results
        ])
        out.set_index("entry_date", inplace=True)

        label_counts = out["label"].value_counts()
        logger.info(
            "Labeled %d entries: +1=%d, -1=%d, 0=%d",
            len(out),
            label_counts.get(1, 0),
            label_counts.get(-1, 0),
            label_counts.get(0, 0),
        )
        return out

    def label_one(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: Optional[float] = None,
    ) -> Optional[BarrierResult]:
        """Label a single trade entry at df.iloc[entry_idx].

        Used in live paper trading to retroactively label trades as they close.
        """
        df = self._validate(df)
        if entry_idx >= len(df) - 1:
            return None

        atr = self._compute_atr(df)
        atr_val = atr.iloc[entry_idx]
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        price = entry_price or df["close"].iloc[entry_idx]
        atr_val = max(atr_val, price * self.min_atr_pct)
        return self._apply_barriers(df, entry_idx, price, atr_val)

    def _apply_barriers(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, atr_val: float
    ) -> Optional[BarrierResult]:
        tp_price = entry_price + atr_val * self.tp_mult
        sl_price = entry_price - atr_val * self.sl_mult

        end_idx = min(entry_idx + self.max_hold, len(df) - 1)
        forward = df.iloc[entry_idx + 1 : end_idx + 1]

        if forward.empty:
            return None

        label = 0
        barrier_hit = "time"
        exit_idx = end_idx
        exit_price = forward["close"].iloc[-1]

        for j, (_, bar) in enumerate(forward.iterrows()):
            tp_hit = bar["high"] >= tp_price
            sl_hit = bar["low"] <= sl_price

            # Both hit in same bar — use intraday order heuristic:
            # if gap-up past TP, TP is first; else SL considered first
            if tp_hit and sl_hit:
                if bar["open"] >= tp_price:
                    label, barrier_hit = 1, "tp"
                    exit_price = tp_price
                else:
                    label, barrier_hit = -1, "sl"
                    exit_price = sl_price
                exit_idx = entry_idx + 1 + j
                break
            elif tp_hit:
                label, barrier_hit = 1, "tp"
                exit_price = tp_price
                exit_idx = entry_idx + 1 + j
                break
            elif sl_hit:
                label, barrier_hit = -1, "sl"
                exit_price = sl_price
                exit_idx = entry_idx + 1 + j
                break

        return BarrierResult(
            entry_date=df.index[entry_idx],
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            exit_date=df.index[exit_idx],
            exit_price=exit_price,
            label=label,
            barrier_hit=barrier_hit,
            return_pct=(exit_price - entry_price) / entry_price,
            days_held=(df.index[exit_idx] - df.index[entry_idx]).days,
            atr_at_entry=atr_val,
        )

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, min_periods=self.atr_period).mean()

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        df = df.copy()
        df.columns = df.columns.str.lower()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def class_weights(self, labeled: pd.DataFrame) -> dict:
        """Return balanced class weights for LightGBM (handles class imbalance)."""
        counts = labeled["label"].value_counts()
        total = len(labeled)
        return {
            label: round(total / (len(counts) * count), 4)
            for label, count in counts.items()
        }
