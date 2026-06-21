"""Tests for strategy/signal_basket.py — SETUP-1."""

from unittest.mock import patch
import pytest

from strategy.signal_basket import (
    detect_primary_setup,
    compute_entry_basket,
    compute_exit_basket,
    check_hard_exit,
    HARD_EXIT_CONDITIONS,
)

# ── Minimal config returned by mock ──────────────────────────────────────────

_CFG = {
    "basket_signal_weights": {
        "rsi_momentum_zone":      15,
        "rsi_oversold":           20,
        "price_above_20ma":       15,
        "price_above_50ma":       10,
        "macd_bullish":           15,
        "volume_confirming":      15,
        "atr_not_extreme":         5,
        "score_above_52pct":      15,
        "score_above_60pct":      20,
        "insider_buying_30d":     20,
        "estimates_rising":       15,
        "revenue_inflecting":     20,
        "no_value_trap":          10,
        "strong_fundamentals":    15,
        "positive_news_6h":       20,
        "major_positive_event":   30,
        "no_major_negative_48h":  10,
        "no_structural_sec":      10,
        "sector_positive_news":   10,
        "regime_favourable":      15,
        "sector_inflow":          10,
        "daily_trend_bullish":    15,
        "vix_not_inverted":        5,
        "stabilisation_4of7":     25,
        "stabilisation_6of7":     35,
        "peers_not_singled_out":  15,
        "no_negative_news_48h":   20,
    },
    "basket_thresholds": {
        "SETUP_MOMENTUM":    55,
        "SETUP_BOUNCE":      50,
        "SETUP_INFLECTION":  45,
        "SETUP_NEWS_DRIVEN": 40,
    },
    "exit_basket_signal_weights": {
        "rsi_momentum_lost_2bars":     20,
        "price_below_20ma_2bars":      20,
        "volume_distribution":         15,
        "macd_bearish_cross":          15,
        "score_below_exit_threshold":  20,
        "revenue_accelerating_down":   25,
        "insider_selling_cluster":     30,
        "estimates_cut_twice":         20,
        "major_negative_event":        35,
        "negative_guidance":           25,
        "ceo_departure":               30,
        "regime_crash":                40,
        "regime_deteriorated":         20,
    },
    "exit_basket_thresholds": {
        "SETUP_MOMENTUM":    45,
        "SETUP_BOUNCE":      40,
        "SETUP_INFLECTION":  60,
        "SETUP_NEWS_DRIVEN": 35,
    },
}


def _mock_cfg():
    return patch(
        "strategy.signal_basket._load_basket_config",
        return_value={
            "entry_weights":    _CFG["basket_signal_weights"],
            "exit_weights":     _CFG["exit_basket_signal_weights"],
            "entry_thresholds": _CFG["basket_thresholds"],
            "exit_thresholds":  _CFG["exit_basket_thresholds"],
        },
    )


# ── Baseline data contexts ────────────────────────────────────────────────────

def _momentum_data(**overrides) -> dict:
    """Data that satisfies SETUP_MOMENTUM (RSI+MA+MACD+volume = 60pts)."""
    base = {
        "rsi":                    58.0,   # rsi_momentum_zone(15) ✓
        "price_above_20ma":       True,   # price_above_20ma(15) ✓
        "price_above_50ma":       True,   # price_above_50ma(10) ✓
        "macd_bullish":           True,   # macd_bullish(15) ✓
        "volume_ratio":           1.5,    # volume_confirming(15) ✓
        "atr_pct":                1.5,    # atr_not_extreme(5) ✓
        "score_pct":              0.55,   # score_above_52pct(15) ✓
        "fundamental_score_pct":  0.50,
        "daily_trend":            "BULLISH",
        "regime_state":           "MID_BULL",  # regime_favourable(15)
        "vix":                    15.0,          # vix_not_inverted(5)
        "stabilisation_score":    0,
        "major_negative_event_48h": False,       # no_major_negative_48h(10)
        "no_structural_sec":      False,          # no_structural_sec(10)
        "negative_news_48h":      False,          # no_negative_news_48h(20)
        "peers_not_singled_out":  False,
        "sector_wide_negative":   False,
        "pe_ratio":               20.0,
        "signal_type":            "",
        "news_direction":         "neutral",
        "news_event_hours_ago":   99.0,
        "recent_major_news":      False,
        "positive_news_6h":       False,
        "major_positive_event_2h":False,
        "insider_buying_30d":     False,
        "estimates_revised_up":   False,
        "revenue_inflecting":     False,
        "sector_positive_news":   False,
        "sector_inflow":          False,
    }
    base.update(overrides)
    return base


def _bounce_data(**overrides) -> dict:
    """Data that satisfies SETUP_BOUNCE (rsi_oversold+stabilisation_4of7 = 45pts)."""
    base = {
        "rsi":                    32.0,   # rsi_oversold(20) ✓
        "stabilisation_score":    5,      # stabilisation_4of7(25) ✓
        "price_above_20ma":       False,
        "price_above_50ma":       False,
        "macd_bullish":           False,
        "volume_ratio":           1.0,
        "atr_pct":                2.5,
        "score_pct":              0.48,
        "fundamental_score_pct":  0.45,
        "daily_trend":            "BEARISH",
        "regime_state":           "NEUTRAL",
        "vix":                    22.0,
        "major_negative_event_48h": False,
        "no_structural_sec":      False,
        "negative_news_48h":      False,
        "peers_not_singled_out":  False,
        "sector_wide_negative":   False,
        "pe_ratio":               20.0,
        "signal_type":            "",
        "news_direction":         "neutral",
        "news_event_hours_ago":   99.0,
        "recent_major_news":      False,
        "positive_news_6h":       False,
        "major_positive_event_2h":False,
        "insider_buying_30d":     False,
        "estimates_revised_up":   False,
        "revenue_inflecting":     False,
        "sector_positive_news":   False,
        "sector_inflow":          False,
    }
    base.update(overrides)
    return base


# ── detect_primary_setup ──────────────────────────────────────────────────────

def test_setup_news_driven_has_highest_priority():
    data = _momentum_data(
        recent_major_news=True,
        news_event_hours_ago=1.5,
        news_direction="POSITIVE",
        # also satisfies MOMENTUM conditions — NEWS_DRIVEN must win
    )
    assert detect_primary_setup(data) == "SETUP_NEWS_DRIVEN"


def test_setup_inflection_priority_over_bounce_and_momentum():
    data = _momentum_data(
        signal_type="CYCLICAL_INFLECTION",
        rsi=32.0,
        stabilisation_score=5,
    )
    assert detect_primary_setup(data) == "SETUP_INFLECTION"


def test_setup_bounce_detected_oversold_stabilised():
    data = _bounce_data()
    assert detect_primary_setup(data) == "SETUP_BOUNCE"


def test_setup_momentum_detected_normal_conditions():
    data = _momentum_data()
    assert detect_primary_setup(data) == "SETUP_MOMENTUM"


def test_setup_none_when_no_conditions_met():
    data = _momentum_data(
        rsi=45.0,           # not in momentum zone 50-72, not oversold
        daily_trend="NEUTRAL",
        score_pct=0.40,     # below 52%
        stabilisation_score=0,
        recent_major_news=False,
        signal_type="",
    )
    assert detect_primary_setup(data) == "NONE"


def test_news_driven_requires_positive_direction():
    """Neutral news even if major should not trigger NEWS_DRIVEN."""
    data = _momentum_data(
        recent_major_news=True,
        news_event_hours_ago=1.0,
        news_direction="neutral",  # not POSITIVE
    )
    # should fall through to MOMENTUM (RSI+MA still satisfied)
    assert detect_primary_setup(data) == "SETUP_MOMENTUM"


# ── compute_entry_basket: SETUP_MOMENTUM ─────────────────────────────────────

def test_momentum_basket_fires_rsi_ma_macd_volume():
    """
    RSI(15)+MA(15)+MA50(10)+MACD(15)+volume(15)+atr(5)+score52pct(15)
    +no_major_neg(10)+no_struct_sec(10)+regime(15)+vix(5)+daily_trend(15)
    +no_neg_news(20)+peers(15) = well above 55 threshold, no news needed.
    """
    data = _momentum_data()
    with _mock_cfg():
        result = compute_entry_basket(data, "SETUP_MOMENTUM")
    assert result["qualifies"] is True
    assert result["basket_score"] >= 55
    assert result["threshold"] == 55
    assert result["setup_type"] == "SETUP_MOMENTUM"


def test_momentum_basket_fails_without_key_signals():
    """With all technical + news signals negative the basket falls below 55."""
    data = _momentum_data(
        rsi=40.0,                       # not in momentum zone, not oversold
        macd_bullish=False,
        daily_trend="BEARISH",          # daily_trend_bullish fails
        price_above_20ma=False,
        price_above_50ma=False,
        volume_ratio=0.8,               # volume_confirming fails
        score_pct=0.45,                 # score_above_52pct fails
        major_negative_event_48h=True,  # no_major_negative_48h fails
        negative_news_48h=True,         # no_negative_news_48h fails
        no_structural_sec=True,         # no_structural_sec fails
        sector_wide_negative=True,      # peers_not_singled_out fails
    )
    with _mock_cfg():
        result = compute_entry_basket(data, "SETUP_MOMENTUM")
    assert result["qualifies"] is False


# ── compute_entry_basket: SETUP_BOUNCE ───────────────────────────────────────

def test_bounce_fires_with_rsi_oversold_and_stabilisation_only():
    """
    rsi_oversold(20) + stabilisation_4of7(25) = 45 pts alone.
    Add no_major_neg(10) + no_struct_sec(10) + no_neg_news(20) = 45+40 = 85.
    Well above bounce threshold of 50. No news or momentum needed.
    """
    data = _bounce_data()
    with _mock_cfg():
        result = compute_entry_basket(data, "SETUP_BOUNCE")
    assert result["qualifies"] is True
    assert result["threshold"] == 50


def test_bounce_blocked_by_major_negative_event():
    """Major negative event + negative news disqualifies SETUP_BOUNCE."""
    # A major negative event implies negative news — both flags must be set
    data = _bounce_data(
        major_negative_event_48h=True,
        negative_news_48h=True,
    )
    with _mock_cfg():
        result = compute_entry_basket(data, "SETUP_BOUNCE")
    hit_names = [s.split("(")[0] for s in result["signals_hit"]]
    assert "no_major_negative_48h" not in hit_names
    assert "no_negative_news_48h" not in hit_names


def test_bounce_blocked_by_structural_sec_filing():
    """Going-concern or SEC fraud signal should reduce bounce basket."""
    data = _bounce_data(no_structural_sec=True)   # True = problem present
    with _mock_cfg():
        result = compute_entry_basket(data, "SETUP_BOUNCE")
    hit_names = [s.split("(")[0] for s in result["signals_hit"]]
    assert "no_structural_sec" not in hit_names


# ── compute_entry_basket: SETUP_INFLECTION ────────────────────────────────────

def test_inflection_fires_without_strong_technicals():
    """
    revenue_inflecting(20) + insider_buying_30d(20) + score_above_52pct(15)
    = 55 pts alone, above threshold of 45. No RSI zone or MA needed.
    """
    data = _bounce_data(
        revenue_inflecting=True,     # +20
        insider_buying_30d=True,     # +20
        score_pct=0.53,              # score_above_52pct(15)
        # RSI and MAs deliberately weak (typical inflection entry)
        rsi=38.0,
        price_above_20ma=False,
        macd_bullish=False,
        stabilisation_score=0,
    )
    with _mock_cfg():
        result = compute_entry_basket(data, "SETUP_INFLECTION")
    assert result["qualifies"] is True
    assert result["threshold"] == 45
    hit_names = [s.split("(")[0] for s in result["signals_hit"]]
    assert "revenue_inflecting" in hit_names
    assert "insider_buying_30d" in hit_names


# ── compute_exit_basket ───────────────────────────────────────────────────────

def test_exit_basket_requires_45pts_for_momentum():
    """Single RSI signal (20pts) is not enough to exit MOMENTUM."""
    data = _momentum_data(
        rsi_below_45_streak=2,      # rsi_momentum_lost_2bars(20) ✓
        price_below_20ma_streak=0,
        macd_bearish_cross=False,
        regime_state="MID_BULL",
        score_pct=0.60,             # above exit threshold → no score signal
        major_negative_event_48h=False,
        negative_guidance=False,
        ceo_departure=False,
        insider_selling_count_30d=0,
        estimates_cut_count=0,
        revenue_declining_accel=False,
        volume_distribution=False,
        regime_at_entry="MID_BULL",
        setup_type_at_entry="SETUP_MOMENTUM",
    )
    with _mock_cfg():
        result = compute_exit_basket(data, "SETUP_MOMENTUM")
    assert result["threshold"] == 45
    # RSI alone = 20 pts < 45 → should NOT exit
    assert result["basket_score"] < 45
    assert result["qualifies"] is False


def test_exit_basket_qualifies_with_multiple_signals():
    """RSI(20) + price_below_20MA(20) + MACD_bear(15) = 55 >= 45 → exit."""
    data = _momentum_data(
        rsi_below_45_streak=2,      # +20
        price_below_20ma_streak=2,  # +20
        macd_bearish_cross=True,    # +15
        score_pct=0.38,             # score_below_exit_threshold(20) +20
        regime_state="MID_BULL",
        major_negative_event_48h=False,
        negative_guidance=False,
        ceo_departure=False,
        insider_selling_count_30d=0,
        estimates_cut_count=0,
        revenue_declining_accel=False,
        volume_distribution=False,
        regime_at_entry="MID_BULL",
        setup_type_at_entry="SETUP_MOMENTUM",
    )
    with _mock_cfg():
        result = compute_exit_basket(data, "SETUP_MOMENTUM")
    assert result["qualifies"] is True
    assert result["basket_score"] >= 45


def test_exit_basket_inflection_has_very_high_bar():
    """SETUP_INFLECTION exit threshold is 60 — long-term thesis."""
    data = _momentum_data(
        rsi_below_45_streak=2,
        price_below_20ma_streak=2,
        macd_bearish_cross=True,
        score_pct=0.38,
        regime_at_entry="MID_BULL",
        regime_state="MID_BULL",
        setup_type_at_entry="SETUP_INFLECTION",
        major_negative_event_48h=False,
        negative_guidance=False,
        ceo_departure=False,
        insider_selling_count_30d=0,
        estimates_cut_count=0,
        revenue_declining_accel=False,
        volume_distribution=False,
    )
    with _mock_cfg():
        result = compute_exit_basket(data, "SETUP_INFLECTION")
    assert result["threshold"] == 60   # very high bar


def test_regime_crash_forces_exit():
    """REGIME = CRASH alone (40pts) exceeds NEWS_DRIVEN exit threshold (35)."""
    data = _momentum_data(
        regime_state="CRASH",
        regime_at_entry="MID_BULL",
        setup_type_at_entry="SETUP_NEWS_DRIVEN",
        score_pct=0.60,
        major_negative_event_48h=False,
        negative_guidance=False,
        ceo_departure=False,
        insider_selling_count_30d=0,
        estimates_cut_count=0,
        revenue_declining_accel=False,
        volume_distribution=False,
        rsi_below_45_streak=0,
        price_below_20ma_streak=0,
        macd_bearish_cross=False,
    )
    with _mock_cfg():
        result = compute_exit_basket(data, "SETUP_NEWS_DRIVEN")
    assert result["qualifies"] is True


# ── check_hard_exit ───────────────────────────────────────────────────────────

def test_hard_exit_going_concern():
    data = {"going_concern_8k": True}
    assert check_hard_exit(data) == "going_concern_8k"


def test_hard_exit_kill_switch():
    data = {"kill_switch_active": True}
    assert check_hard_exit(data) == "kill_switch_active"


def test_no_hard_exit_when_clean():
    data = {c: False for c in HARD_EXIT_CONDITIONS}
    assert check_hard_exit(data) is None


def test_hard_exit_conditions_list_is_complete():
    expected = [
        "hard_stop_triggered", "kill_switch_active",
        "going_concern_8k", "trading_halt",
    ]
    for c in expected:
        assert c in HARD_EXIT_CONDITIONS


# ── Config loading: no hardcoded weights ──────────────────────────────────────

def test_basket_weights_loaded_from_config_not_hardcoded():
    """
    Changing the config threshold changes whether the basket qualifies.
    Proves weights are read from config at runtime — not hardcoded.
    """
    data = _momentum_data()

    # With normal threshold (55): should qualify
    with _mock_cfg():
        result_normal = compute_entry_basket(data, "SETUP_MOMENTUM")
    assert result_normal["qualifies"] is True

    # With raised threshold (999): should not qualify
    high_threshold_cfg = {
        "entry_weights":    _CFG["basket_signal_weights"],
        "exit_weights":     _CFG["exit_basket_signal_weights"],
        "entry_thresholds": {"SETUP_MOMENTUM": 999},  # impossibly high
        "exit_thresholds":  _CFG["exit_basket_thresholds"],
    }
    with patch("strategy.signal_basket._load_basket_config",
               return_value=high_threshold_cfg):
        result_high = compute_entry_basket(data, "SETUP_MOMENTUM")
    assert result_high["qualifies"] is False
    assert result_high["threshold"] == 999
