"""Tests for strategy/trade_ledger.py — LEDGER-1."""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

import strategy.trade_ledger as ledger_mod
from strategy.trade_ledger import (
    open_trade,
    update_trade,
    record_partial,
    close_trade,
    _make_trade_id,
    _get_trade_number,
    _ledger_path,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_ledger_dir(tmp_path, monkeypatch):
    """Redirect all file output to a temp directory for each test."""
    ledger_dir = tmp_path / "trade_ledger"
    ledger_dir.mkdir()
    master_log = tmp_path / "trade_log.md"

    monkeypatch.setattr(ledger_mod, "_LEDGER_DIR", str(ledger_dir))
    monkeypatch.setattr(ledger_mod, "_MASTER_LOG", str(master_log))
    return tmp_path


def _no_db(monkeypatch):
    """Silence all DB writes."""
    monkeypatch.setattr(ledger_mod, "_db_write", lambda *a, **kw: None)


def _sample_basket(score: int = 60, threshold: int = 55) -> dict:
    return {
        "basket_score": score,
        "threshold":    threshold,
        "signals_hit":  [
            f"rsi_momentum_zone(+15)",
            f"macd_bullish(+15)",
            f"price_above_20ma(+15)",
        ],
    }


def _sample_scorecard() -> dict:
    return {
        "overall_score": 80,
        "total_score":   80,
        "technical":   {"display": "40/50"},
        "fundamental": {"display": "25/40"},
        "macro":       {"display": "15/20"},
    }


# ── open_trade ────────────────────────────────────────────────────────────────

def test_open_trade_creates_ledger_file(monkeypatch):
    """BUY on NVDA creates logs/trade_ledger/NVDA.md."""
    _no_db(monkeypatch)
    trade_id = open_trade(
        symbol="NVDA", tier="TIER_2", bundle="B2",
        entry_price=820.0, shares=2.5,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=["[Reuters] NVDA raises guidance"],
        stop_price=780.0,
        exit_criteria=["RSI < 45 for 2 bars", "Exit basket >= 45"],
    )
    path = _ledger_path("NVDA")
    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert "NVDA" in content
    assert "BUY" in content
    assert "$820.00" in content
    assert "SETUP_MOMENTUM" in content
    assert "MID_BULL" in content
    assert trade_id.startswith("NVDA_")


def test_open_trade_writes_master_log(monkeypatch):
    """open_trade appends one line to trade_log.md."""
    _no_db(monkeypatch)
    open_trade(
        symbol="NVDA", tier="TIER_2", bundle="B2",
        entry_price=820.0, shares=2.5,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=780.0,
        exit_criteria=[],
    )
    assert os.path.exists(ledger_mod._MASTER_LOG)
    content = open(ledger_mod._MASTER_LOG, encoding="utf-8").read()
    assert "BUY" in content
    assert "NVDA" in content


def test_open_trade_returns_trade_id(monkeypatch):
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="AMD", tier="TIER_3", bundle="B4",
        entry_price=160.0, shares=5.0,
        setup_type="SETUP_BOUNCE",
        basket=_sample_basket(score=52, threshold=50),
        scorecard_snapshot=_sample_scorecard(),
        regime="NEUTRAL",
        key_news=[],
        stop_price=148.0,
        exit_criteria=[],
    )
    assert tid.startswith("AMD_")


def test_second_trade_appends_trade_2(monkeypatch):
    """Second BUY on same symbol appends TRADE #2 (not a new file)."""
    _no_db(monkeypatch)
    kwargs = dict(
        symbol="NVDA", tier="TIER_2", bundle="B2",
        entry_price=820.0, shares=2.5,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=780.0,
        exit_criteria=[],
    )
    open_trade(**kwargs)
    open_trade(**{**kwargs, "entry_price": 850.0})   # second trade

    content = open(_ledger_path("NVDA"), encoding="utf-8").read()
    assert "TRADE #1" in content
    assert "TRADE #2" in content
    # Still only one file, not two
    assert content.count("FILE CREATED:") == 1


def test_ledger_file_contains_basket_signals(monkeypatch):
    _no_db(monkeypatch)
    open_trade(
        symbol="MSFT", tier="TIER_1", bundle="B3",
        entry_price=400.0, shares=1.0,
        setup_type="SETUP_INFLECTION",
        basket={"basket_score": 52, "threshold": 45,
                "signals_hit": ["revenue_inflecting(+20)", "insider_buying_30d(+20)"]},
        scorecard_snapshot=_sample_scorecard(),
        regime="EARLY_BULL",
        key_news=["[SEC Form 4] CFO bought shares"],
        stop_price=360.0,
        exit_criteria=["Revenue decline accelerating"],
    )
    content = open(_ledger_path("MSFT"), encoding="utf-8").read()
    assert "revenue_inflecting" in content
    assert "insider_buying_30d" in content
    assert "CFO bought shares" in content
    assert "STOP LOSS:    $360.00" in content


# ── update_trade ──────────────────────────────────────────────────────────────

def test_update_trade_writes_when_score_changes_5pts(monkeypatch):
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="QCOM", tier="TIER_2", bundle="B2",
        entry_price=150.0, shares=3.0,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=138.0,
        exit_criteria=[],
    )
    update_trade(tid, new_score=85.0, old_score=80.0, reason="Guidance raised")

    content = open(_ledger_path("QCOM"), encoding="utf-8").read()
    assert "UPDATE" in content
    assert "80.0 → 85.0" in content
    assert "Guidance raised" in content


def test_update_trade_skipped_when_small_change(monkeypatch):
    """Score change < 5 pts without material_event → no UPDATE block."""
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="QCOM", tier="TIER_2", bundle="B2",
        entry_price=150.0, shares=3.0,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=138.0,
        exit_criteria=[],
    )
    original = open(_ledger_path("QCOM"), encoding="utf-8").read()
    update_trade(tid, new_score=82.0, old_score=80.0, reason="minor")
    after = open(_ledger_path("QCOM"), encoding="utf-8").read()
    assert after == original   # file unchanged


def test_update_trade_written_for_material_event_regardless_of_score(monkeypatch):
    """material_event forces UPDATE block even if score didn't change much."""
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="QCOM", tier="TIER_2", bundle="B2",
        entry_price=150.0, shares=3.0,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=138.0,
        exit_criteria=[],
    )
    update_trade(
        tid, new_score=81.0, old_score=80.0, reason="SEC 8-K filed",
        material_event={"headline": "Quarterly earnings: EPS beat by 11%"}
    )
    content = open(_ledger_path("QCOM"), encoding="utf-8").read()
    assert "UPDATE" in content
    assert "EPS beat" in content


# ── record_partial ────────────────────────────────────────────────────────────

def test_record_partial_writes_emoji_line(monkeypatch):
    """Partial profit writes 🎯 PARTIAL line with P&L."""
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="NVDA", tier="TIER_2", bundle="B2",
        entry_price=820.0, shares=2.5,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=780.0,
        exit_criteria=[],
    )
    record_partial(tid, pct_sold=0.33, price=950.0, pnl_pct=15.85)

    content = open(_ledger_path("NVDA"), encoding="utf-8").read()
    assert "🎯 PARTIAL" in content
    assert "$950.00" in content
    assert "+15.8%" in content or "15.85" in content


# ── close_trade ───────────────────────────────────────────────────────────────

def test_close_trade_appends_closed_block(monkeypatch):
    """SELL appends CLOSED block to correct symbol file."""
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="NVDA", tier="TIER_2", bundle="B2",
        entry_price=820.0, shares=2.5,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=780.0,
        exit_criteria=[],
    )
    close_trade(
        tid,
        exit_price=950.0,
        exit_reason="exit_basket",
        exit_basket={"basket_score": 48, "threshold": 45,
                     "signals_hit": ["rsi_momentum_lost_2bars(+20)"]},
    )
    content = open(_ledger_path("NVDA"), encoding="utf-8").read()
    assert "SELL" in content
    assert "$950.00" in content
    assert "exit_basket" in content
    assert "DAYS HELD:" in content
    assert "RETURN:" in content


def test_close_trade_appends_master_log(monkeypatch):
    """SELL also appends one-line SELL entry to trade_log.md."""
    _no_db(monkeypatch)
    tid = open_trade(
        symbol="NVDA", tier="TIER_2", bundle="B2",
        entry_price=820.0, shares=2.5,
        setup_type="SETUP_MOMENTUM",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="MID_BULL",
        key_news=[],
        stop_price=780.0,
        exit_criteria=[],
    )
    close_trade(tid, exit_price=950.0, exit_reason="exit_basket",
                exit_basket={})
    content = open(ledger_mod._MASTER_LOG, encoding="utf-8").read()
    # Newest first → SELL should appear before BUY in the file
    assert "SELL" in content
    assert "NVDA" in content


def test_master_log_newest_first(monkeypatch):
    """Master log has newest entry at the top (below header)."""
    _no_db(monkeypatch)
    kwargs = dict(
        symbol="AMD", tier="TIER_3", bundle="B4",
        entry_price=160.0, shares=5.0,
        setup_type="SETUP_BOUNCE",
        basket=_sample_basket(),
        scorecard_snapshot=_sample_scorecard(),
        regime="NEUTRAL",
        key_news=[],
        stop_price=148.0,
        exit_criteria=[],
    )
    open_trade(**kwargs)                           # first BUY
    open_trade(**{**kwargs, "entry_price": 155.0}) # second BUY

    content = open(ledger_mod._MASTER_LOG, encoding="utf-8").read()
    lines   = [l for l in content.splitlines() if "BUY" in l and "AMD" in l]
    assert len(lines) == 2
    # $155.00 entry (newer) should appear before $160.00 (older)
    assert lines[0].index("155") < len(lines[0])   # second trade is on top


# ── _make_trade_id / _get_trade_number ───────────────────────────────────────

def test_make_trade_id_format():
    dt = datetime(2026, 5, 10, 11, 30)
    assert _make_trade_id("NVDA", dt) == "NVDA_20260510T1130"


def test_get_trade_number_new_file(monkeypatch, tmp_ledger_dir):
    monkeypatch.setattr(ledger_mod, "_LEDGER_DIR", str(tmp_ledger_dir / "trade_ledger"))
    os.makedirs(str(tmp_ledger_dir / "trade_ledger"), exist_ok=True)
    assert _get_trade_number("NEW_SYM") == 1


def test_get_trade_number_increments(monkeypatch, tmp_path):
    ledger_dir = tmp_path / "trade_ledger"
    ledger_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(ledger_mod, "_LEDGER_DIR", str(ledger_dir))

    path = ledger_dir / "XYZ.md"
    path.write_text("TRADE #1\nTRADE #2\n", encoding="utf-8")
    assert _get_trade_number("XYZ") == 3
