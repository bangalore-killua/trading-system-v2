"""Tests for strategy.triple_barrier — label generation."""

import numpy as np
import pandas as pd
import pytest

from strategy.triple_barrier import TripleBarrierLabeler, BarrierResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def labeler():
    return TripleBarrierLabeler(
        tp_multiplier=3.0,
        sl_multiplier=1.5,
        max_holding_days=10,
        atr_period=14,
    )


@pytest.fixture
def trending_up_df():
    """Steadily rising OHLCV — most entries should hit TP."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.ones(n) * 0.6 + np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open":   close - 0.3,
        "high":   close + 1.5,
        "low":    close - 0.8,
        "close":  close,
        "volume": np.ones(n) * 1_000_000,
    }, index=dates)


@pytest.fixture
def crashing_df():
    """Sharply falling OHLCV — most entries should hit SL."""
    np.random.seed(7)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 - np.cumsum(np.ones(n) * 0.8 + np.random.randn(n) * 0.1)
    close = np.maximum(close, 5)
    return pd.DataFrame({
        "open":   close + 0.3,
        "high":   close + 0.5,
        "low":    close - 2.0,
        "close":  close,
        "volume": np.ones(n) * 1_000_000,
    }, index=dates)


@pytest.fixture
def flat_df():
    """Flat OHLCV — most entries should time out (label 0)."""
    np.random.seed(0)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    noise = np.random.randn(n) * 0.05
    close = np.full(n, 100.0) + noise
    return pd.DataFrame({
        "open":   close - 0.01,
        "high":   close + 0.15,
        "low":    close - 0.15,
        "close":  close,
        "volume": np.ones(n) * 1_000_000,
    }, index=dates)


# ── Core label() tests ────────────────────────────────────────────────────────

def test_label_returns_dataframe(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_label_columns_present(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    for col in ("entry_price", "tp_price", "sl_price", "label", "barrier_hit",
                "return_pct", "days_held", "atr_at_entry"):
        assert col in result.columns, f"Missing column: {col}"


def test_label_values_are_valid(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert set(result["label"].unique()).issubset({-1, 0, 1})
    assert set(result["barrier_hit"].unique()).issubset({"tp", "sl", "time"})


def test_label_index_is_datetime(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_trending_up_majority_tp(labeler, trending_up_df):
    """Strongly rising market should produce mostly +1 labels."""
    result = labeler.label(trending_up_df)
    tp_count = (result["label"] == 1).sum()
    assert tp_count > len(result) * 0.4, f"Expected >40% TP hits, got {tp_count}/{len(result)}"


def test_crashing_df_majority_sl(labeler, crashing_df):
    """Strongly falling market should produce mostly -1 labels."""
    result = labeler.label(crashing_df)
    sl_count = (result["label"] == -1).sum()
    assert sl_count > len(result) * 0.4, f"Expected >40% SL hits, got {sl_count}/{len(result)}"


def test_days_held_bounded(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert (result["days_held"] <= labeler.max_hold + 5).all()
    assert (result["days_held"] >= 0).all()


def test_tp_price_above_sl_price(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert (result["tp_price"] > result["sl_price"]).all()


def test_tp_price_above_entry(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert (result["tp_price"] > result["entry_price"]).all()


def test_sl_price_below_entry(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    assert (result["sl_price"] < result["entry_price"]).all()


def test_too_short_returns_empty(labeler):
    short_df = pd.DataFrame({
        "open": [100.0] * 20,
        "high": [101.0] * 20,
        "low":  [99.0]  * 20,
        "close":[100.0] * 20,
    }, index=pd.date_range("2024-01-01", periods=20, freq="D"))
    result = labeler.label(short_df)
    assert result.empty


def test_label_is_consistent_with_barrier_hit(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    tp_rows = result[result["barrier_hit"] == "tp"]
    sl_rows = result[result["barrier_hit"] == "sl"]
    time_rows = result[result["barrier_hit"] == "time"]
    assert (tp_rows["label"] == 1).all()
    assert (sl_rows["label"] == -1).all()
    assert (time_rows["label"] == 0).all()


# ── label_one() tests ─────────────────────────────────────────────────────────

def test_label_one_returns_barrier_result(labeler, trending_up_df):
    result = labeler.label_one(trending_up_df, entry_idx=50)
    assert isinstance(result, BarrierResult)


def test_label_one_with_explicit_price(labeler, trending_up_df):
    result = labeler.label_one(trending_up_df, entry_idx=50, entry_price=110.0)
    assert result is not None
    assert result.entry_price == 110.0


def test_label_one_out_of_bounds_returns_none(labeler, trending_up_df):
    result = labeler.label_one(trending_up_df, entry_idx=len(trending_up_df) - 1)
    assert result is None


# ── class_weights() tests ─────────────────────────────────────────────────────

def test_class_weights_all_labels_present(labeler, trending_up_df):
    result = labeler.label(trending_up_df)
    # Mix of labels; get weights for whatever labels appear
    weights = labeler.class_weights(result)
    assert isinstance(weights, dict)
    assert all(v > 0 for v in weights.values())


def test_class_weights_minority_higher(labeler, trending_up_df):
    """Rare class should get higher weight than common class."""
    result = labeler.label(trending_up_df)
    if {-1, 1}.issubset(result["label"].unique()):
        counts = result["label"].value_counts()
        weights = labeler.class_weights(result)
        common = counts.idxmax()
        rare = counts.idxmin()
        assert weights[rare] >= weights[common]


# ── Custom parameters tests ───────────────────────────────────────────────────

def test_custom_tp_sl_multipliers(trending_up_df):
    labeler_tight = TripleBarrierLabeler(tp_multiplier=1.0, sl_multiplier=0.5, max_holding_days=5)
    labeler_wide = TripleBarrierLabeler(tp_multiplier=5.0, sl_multiplier=3.0, max_holding_days=20)
    result_tight = labeler_tight.label(trending_up_df)
    result_wide = labeler_wide.label(trending_up_df)
    # Tight barriers should resolve faster (smaller days_held on average)
    assert result_tight["days_held"].mean() <= result_wide["days_held"].mean() + 3


def test_case_insensitive_columns(labeler):
    """Labeler should accept uppercase column names."""
    n = 100
    df = pd.DataFrame({
        "Open":  [100.0] * n,
        "High":  [101.0] * n,
        "Low":   [99.0]  * n,
        "Close": [100.5] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))
    result = labeler.label(df)
    assert not result.empty
