"""Tests for strategy.threat_classifier — keyword NLP threat levels."""

import pytest

from strategy.threat_classifier import ThreatClassifier


@pytest.fixture
def clf():
    return ThreatClassifier()


def test_bankruptcy_is_critical(clf):
    assert clf.classify("Company files for Chapter 11 bankruptcy") == "CRITICAL"


def test_bank_run_is_critical(clf):
    assert clf.classify("Bank run accelerates as depositors withdraw funds") == "CRITICAL"


def test_going_concern_is_critical(clf):
    assert clf.classify("Auditors raise going concern doubt over the company") == "CRITICAL"


def test_recession_is_high(clf):
    assert clf.classify("US economy enters recession according to NBER") == "HIGH"


def test_fraud_is_high(clf):
    assert clf.classify("SEC investigation launched into accounting fraud at firm") == "HIGH"


def test_earnings_miss_is_high(clf):
    assert clf.classify("Company reports massive earnings miss, guidance cut sharply") == "HIGH"


def test_uncertainty_is_medium(clf):
    result = clf.classify("Market faces heightened uncertainty amid volatility")
    assert result in ("MEDIUM", "HIGH")


def test_positive_news_is_low(clf):
    assert clf.classify("Company reports record revenue, raises guidance") == "LOW"


def test_empty_string_is_low(clf):
    assert clf.classify("") == "LOW"


def test_state_media_downgrade(clf):
    # Same CRITICAL headline but from state media → should be downgraded
    result_neutral = clf.classify_full("Bank run accelerates")
    result_state = clf.classify_full("Bank run accelerates", source_name="RT News")
    assert result_state.score < result_neutral.score
    assert result_state.source_modifier == "STATE_MEDIA"


def test_is_blocking_critical(clf):
    assert clf.is_blocking("CRITICAL") is True


def test_is_blocking_high(clf):
    assert clf.is_blocking("HIGH") is True


def test_is_blocking_medium(clf):
    assert clf.is_blocking("MEDIUM") is False


def test_is_blocking_low(clf):
    assert clf.is_blocking("LOW") is False


def test_is_exit_trigger_only_critical(clf):
    assert clf.is_exit_trigger("CRITICAL") is True
    assert clf.is_exit_trigger("HIGH") is False
    assert clf.is_exit_trigger("MEDIUM") is False


def test_matched_terms_reported(clf):
    result = clf.classify_full("Market crash and circuit breaker triggered today")
    assert len(result.matched_terms) > 0
    assert result.score > 0
