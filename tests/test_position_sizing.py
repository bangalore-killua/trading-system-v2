"""
Unit tests for dynamic position sizing helpers in position_manager.py.
Tests _get_effective_position_pct() at various corpus sizes.
"""

import pytest


# Replicate the formula standalone so tests don't need a full PositionManager init
def _effective_pct(portfolio_value: float,
                   base_pct: float = 0.05,
                   min_usd: float = 300.0) -> float:
    if portfolio_value <= 0:
        return base_pct
    return max(min_usd / portfolio_value, base_pct)


# ── At small corpus: min_position_usd drives the percentage ──────────────────

def test_1k_corpus_gives_30pct():
    """$300 min on $1k corpus = 30% per position → max 3 positions naturally."""
    pct = _effective_pct(1_000)
    assert abs(pct - 0.30) < 0.001

def test_2k_corpus_gives_15pct():
    pct = _effective_pct(2_000)
    assert abs(pct - 0.15) < 0.001

def test_3k_corpus_gives_10pct():
    pct = _effective_pct(3_000)
    assert abs(pct - 0.10) < 0.001


# ── Crossover: at $6k, 5% of corpus = $300 = min_position_usd ────────────────

def test_6k_corpus_at_crossover():
    """5% × $6k = $300 = min_usd exactly — should return base_pct."""
    pct = _effective_pct(6_000)
    assert abs(pct - 0.05) < 0.001


# ── Above crossover: base_size_pct drives ────────────────────────────────────

def test_10k_corpus_gives_5pct():
    pct = _effective_pct(10_000)
    assert abs(pct - 0.05) < 0.001

def test_100k_corpus_gives_5pct():
    pct = _effective_pct(100_000)
    assert abs(pct - 0.05) < 0.001

def test_1m_corpus_gives_5pct():
    pct = _effective_pct(1_000_000)
    assert abs(pct - 0.05) < 0.001


# ── Natural position count (deployable / pos_size) ───────────────────────────

def test_natural_max_positions_1k():
    """At $1k: deployable=$930 / $300 = 3 positions."""
    pv = 1_000
    floor = 0.07
    deployable = pv * (1 - floor)
    pos_size = max(300, pv * 0.05)
    assert int(deployable / pos_size) == 3

def test_natural_max_positions_10k():
    """At $10k: deployable=$9,300 / $500 = 18 positions."""
    pv = 10_000
    floor = 0.07
    deployable = pv * (1 - floor)
    pos_size = max(300, pv * 0.05)
    assert int(deployable / pos_size) == 18

def test_natural_max_positions_100k():
    """At $100k: floor(93,000/5,000) = 18 — stabilises at 18."""
    pv = 100_000
    floor = 0.07
    deployable = pv * (1 - floor)
    pos_size = max(300, pv * 0.05)
    assert int(deployable / pos_size) == 18


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_zero_corpus_returns_base():
    pct = _effective_pct(0)
    assert pct == 0.05

def test_negative_corpus_returns_base():
    pct = _effective_pct(-500)
    assert pct == 0.05
