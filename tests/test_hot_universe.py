"""Tests for data.hot_universe — TTL cache for event-driven symbol discovery."""

import time
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from data.hot_universe import HotUniverse, TriggerSource


def test_add_new_symbol_returns_true():
    hu = HotUniverse()
    is_new = hu.add("AAPL", TriggerSource.NEWS_EVENT)
    assert is_new is True


def test_add_duplicate_symbol_returns_false():
    hu = HotUniverse()
    hu.add("AAPL", TriggerSource.NEWS_EVENT)
    is_new = hu.add("AAPL", TriggerSource.NEWS_EVENT)
    assert is_new is False


def test_get_active_symbols_returns_added():
    hu = HotUniverse()
    hu.add("TSLA", TriggerSource.EDGAR_8K)
    hu.add("NVDA", TriggerSource.TV_SCREENER)
    active = hu.get_active_symbols()
    assert "TSLA" in active
    assert "NVDA" in active


def test_expired_entry_not_returned():
    hu = HotUniverse(ttl_config={TriggerSource.NEWS_EVENT: 0.0001})  # ~0.36s TTL
    hu.add("MSFT", TriggerSource.NEWS_EVENT)
    time.sleep(0.5)
    assert "MSFT" not in hu.get_active_symbols()


def test_is_active_false_after_expiry():
    hu = HotUniverse(ttl_config={TriggerSource.NEWS_EVENT: 0.0001})
    hu.add("GOOG", TriggerSource.NEWS_EVENT)
    time.sleep(0.5)
    assert hu.is_active("GOOG") is False


def test_remove_symbol():
    hu = HotUniverse()
    hu.add("AMD", TriggerSource.VOLUME_SPIKE)
    hu.remove("AMD")
    assert hu.is_active("AMD") is False


def test_callback_fired_on_new_entry():
    hu = HotUniverse()
    cb = MagicMock()
    hu.on_symbol_added(cb)
    hu.add("META", TriggerSource.EDGAR_FORM4)
    cb.assert_called_once_with("META", TriggerSource.EDGAR_FORM4)


def test_callback_not_fired_on_duplicate():
    hu = HotUniverse()
    cb = MagicMock()
    hu.on_symbol_added(cb)
    hu.add("META", TriggerSource.NEWS_EVENT)
    hu.add("META", TriggerSource.NEWS_EVENT)  # duplicate
    assert cb.call_count == 1


def test_size_excludes_expired():
    hu = HotUniverse(ttl_config={TriggerSource.NEWS_EVENT: 0.0001})
    hu.add("X", TriggerSource.NEWS_EVENT)
    hu.add("Y", TriggerSource.EDGAR_8K)  # default TTL = 6h, won't expire
    time.sleep(0.5)
    # X expired, Y still active
    assert hu.size() == 1


def test_get_entry_returns_metadata():
    hu = HotUniverse()
    hu.add("PLTR", TriggerSource.NEWS_EVENT, headline="Q3 earnings beat", sentiment_score=0.85)
    entry = hu.get_entry("PLTR")
    assert entry is not None
    assert entry.headline == "Q3 earnings beat"
    assert entry.sentiment_score == 0.85
    assert entry.source == TriggerSource.NEWS_EVENT


def test_different_ttl_per_source():
    hu = HotUniverse()
    hu.add("A", TriggerSource.VOLUME_SPIKE)   # TTL = 2h
    hu.add("B", TriggerSource.EDGAR_8K)        # TTL = 6h
    entry_a = hu.get_entry("A")
    entry_b = hu.get_entry("B")
    # EDGAR_8K TTL should be longer
    assert entry_b.ttl_remaining_seconds() > entry_a.ttl_remaining_seconds()
