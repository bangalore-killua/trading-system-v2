"""
percentile_utils.py — Shared Percentile Scoring Utilities

Every metric in the screener and factor-scoring pipeline is compared against
the stock's own history OR sector peers — never against fixed global thresholds.

Two scoring modes:
  stock_percentile(value, history)   → 0–100 where 100 = highest the stock has
                                       ever been (e.g. RSI vs own 1-yr history)
  sector_percentile(value, peers)    → 0–100 relative to sector peer group
                                       (e.g. P/E vs industry median)

Both return float in [0, 100]. NaN inputs return 50.0 (neutral).
"""

import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

Number = Union[int, float]


def stock_percentile(
    value: Optional[Number],
    history: List[Optional[Number]],
    higher_is_better: bool = True,
) -> float:
    """Rank *value* within the stock's own recent history.

    Parameters
    ----------
    value : current metric value (e.g. RSI today)
    history : list of historical values for this stock (e.g. RSI for last 252 days)
    higher_is_better : if False, lower values rank higher (e.g. P/E — lower is cheaper)

    Returns
    -------
    float in [0, 100]. 100 means the value is at the historical maximum.
    50 is returned for null / insufficient data.
    """
    if value is None or np.isnan(float(value)):
        return 50.0

    clean = [float(v) for v in history if v is not None and not np.isnan(float(v))]
    if len(clean) < 5:
        return 50.0

    arr = np.array(clean, dtype=np.float64)
    pct = float(np.sum(arr <= float(value)) / len(arr) * 100)

    return pct if higher_is_better else (100.0 - pct)


def sector_percentile(
    value: Optional[Number],
    peers: List[Optional[Number]],
    higher_is_better: bool = True,
) -> float:
    """Rank *value* within a sector peer group.

    Parameters
    ----------
    value : the stock's current metric (e.g. its P/E ratio)
    peers : list of the same metric across sector peers (may include the stock itself)
    higher_is_better : if False, lower values rank higher (e.g. P/E, debt/equity)

    Returns
    -------
    float in [0, 100]. 50 for null / insufficient data.
    """
    if value is None or np.isnan(float(value)):
        return 50.0

    clean = [float(v) for v in peers if v is not None and not np.isnan(float(v))]
    if len(clean) < 3:
        return 50.0

    arr = np.array(clean, dtype=np.float64)
    pct = float(np.sum(arr <= float(value)) / len(arr) * 100)

    return pct if higher_is_better else (100.0 - pct)


def composite_score(
    scores: List[Optional[float]],
    weights: Optional[List[float]] = None,
) -> float:
    """Weighted average of multiple percentile scores, ignoring None values.

    Parameters
    ----------
    scores  : list of 0–100 percentile scores (None = missing, excluded)
    weights : optional list of weights (same length as scores). Equal weights
              are used if omitted. Weights are renormalized after dropping NaN.

    Returns
    -------
    float in [0, 100]. 50.0 if all scores are None.
    """
    valid_pairs = [
        (s, (weights[i] if weights else 1.0))
        for i, s in enumerate(scores)
        if s is not None and not np.isnan(float(s))
    ]
    if not valid_pairs:
        return 50.0

    total_w = sum(w for _, w in valid_pairs)
    if total_w == 0:
        return 50.0

    return float(sum(s * w for s, w in valid_pairs) / total_w)


def winsorize(values: List[Optional[Number]], lower: float = 0.05, upper: float = 0.95) -> List[float]:
    """Clip values at lower/upper quantiles to reduce outlier impact.

    Returns a list of floats the same length as *values*. None inputs are
    passed through as NaN.
    """
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if len(clean) < 5:
        return [float(v) if v is not None else float("nan") for v in values]

    lo = float(np.quantile(clean, lower))
    hi = float(np.quantile(clean, upper))

    result = []
    for v in values:
        if v is None or np.isnan(float(v)):
            result.append(float("nan"))
        else:
            result.append(float(np.clip(float(v), lo, hi)))
    return result
