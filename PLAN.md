# PLAN.md — Session Restart Guide

> **READ THIS FIRST IN ANY NEW CHAT SESSION.**
> This document is the single source of truth for continuing work on trading-system-v2.
> It is self-contained — a new session needs only this file + the codebase to resume.

---

## What This Project Is (3 sentences)

`trading-system-v2` is a greenfield rebuild of `C:\Users\ASUS\TradingSystem`.
It is an event-driven, ML-augmented US equity paper/live trading system on Alpaca.
Key improvements: 3-layer event-driven universe, LightGBM meta-model, 39-source news
pipeline, properly wired signal basket, and real-time discovery via WebSocket + RSS.

---

## Current Build Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Foundation | ✅ Complete | Merged to main via PR |
| Phase 2 — YOLOv8 + Feature Vector | ✅ Complete | Branch: feat/phase-2-yolov8-patterns |
| Phase 3 — LightGBM Meta-Model | 🔨 In Progress | Branch: feat/phase-3-lightgbm |
| Phase 4 — Options expansion | ⏳ Not started | Needs Phase 3 |

### Files Status (update this as work progresses)

#### DONE — Phase 1 Foundation (branch: feat/phase-1-foundation, committed bc7715f)

**New files written:**
- ✅ `README.md`, `PLAN.md`, `requirements.txt`, `config.json`, `.env.example`, `.gitignore`
- ✅ `data/hot_universe.py` — TTL cache for event-driven symbol discovery
- ✅ `data/news_sources.py` — 39-source tier registry
- ✅ `data/news_stream.py` — Alpaca NewsDataStream("*") WebSocket consumer
- ✅ `data/edgar_monitor.py` — SEC EDGAR 8-K + Form 4 RSS monitor
- ✅ `data/tv_screener.py` — TradingView Screener (all US stocks, free)
- ✅ `data/news_aggregator.py` — async 39-source RSS aggregator with NLP
- ✅ `strategy/threat_classifier.py` — word-list NLP: CRITICAL/HIGH/MEDIUM/LOW
- ✅ `strategy/technical_analysis.py` — pandas-ta indicators + 60 CDL patterns
- ✅ `strategy/news_sentiment.py` — FinBERT (yiyanghkust/finbert-tone) wrapper
- ✅ `strategy/position_manager.py` — entry/exit with signal basket PROPERLY WIRED
- ✅ `server/alpaca_trader.py` — alpaca-py with fill verification polling
- ✅ `server/scheduler.py` — APScheduler orchestrator (replaces schedule lib)
- ✅ `run_server.py` — entry point with full dependency injection
- ✅ `tests/conftest.py`, `test_hot_universe.py`, `test_threat_classifier.py`, `test_technical_analysis.py`

**Copied verbatim from TradingSystem v12.15:**
- ✅ `strategy/`: signal_basket, signal_types, regime_detector, trade_ledger,
  position_aging, correlation_risk, holding_tier, trade_state, trade_labels,
  config_loader, screener_guardrails
- ✅ `server/`: api.py, kill_switch.py, order_manager.py, telegram_alerts.py
- ✅ `database.py`
- ✅ Tests ported: 11 test files from TradingSystem

#### DONE — Phase 1 Complete (all files committed on feat/phase-1-foundation)

Additional files added to complete Phase 1:
- ✅ `data/fetcher.py` — yfinance primary + alpaca-py fallback, 5-min cache
- ✅ `data/news_relevance_engine.py` — SpaCy NER + 5000+ company name→ticker map
- ✅ `data/price_stream.py` — Alpaca StockDataStream WebSocket (1-min bars)
- ✅ `strategy/macro_analyser.py` — VIX, yield spread, SPY trend, optional FRED
- ✅ `run_server.py` — fully wired with all dependencies injected

**Phase 1 is complete. The system can paper trade.**

---

#### DONE — Phase 2 (branch: feat/phase-2-yolov8-patterns, merged to main)

- ✅ `strategy/chart_pattern_detector.py` — YOLOv8 visual chart pattern recognition (16 patterns)
  Graceful degradation: returns all-zero dict until `models/chart_patterns_yolov8n.pt` is trained.
- ✅ `strategy/scorer.py` — FeatureVector builder (~125 features) + async PostgreSQL logger
- ✅ `data/fundamentals_fetcher.py` — SEC EDGAR XBRL primary, FMP fallback, weekly cache
- ✅ `server/api.py` — added `GET /api/ledger/trade/{trade_id}` (was missing)
- ✅ Tests: `test_chart_pattern_detector.py` (11), `test_scorer.py` (16)

---

#### IN PROGRESS — Phase 3 (branch: feat/phase-3-lightgbm)

**Files written:**
- ✅ `strategy/triple_barrier.py` — TP/SL/time barrier labeling (López de Prado)
- ✅ `strategy/meta_model.py` — LightGBM multi-class wrapper (train, predict_proba, save/load, WF-validate)
- ✅ `strategy/training_pipeline.py` — CLI orchestrator for historical bootstrap + live retrain
- ✅ Tests: `test_triple_barrier.py` (22), `test_meta_model.py` (18)

**TODO — Phase 3 remaining:**
1. Wire `MetaModel` into `run_server.py` (load at startup if `models/meta_model_lgbm.pkl` exists)
2. Wire `Scorer.get_position_size_multiplier()` → `position_manager.py` (use multiplier on `calculate_position_size()`)
3. Add `trade_features` table to `database.py` (stores feature vectors per trade for live labeling)
4. Nightly Sunday job in `scheduler.py` → calls `TrainingPipeline.run_live_update()` when enough samples
5. Alembic migration for `trade_features` table
6. Update `config.json` with all Phase 3 model params
7. Commit + push + PR

**How to resume Phase 3 in a new session:**
> "We are building trading-system-v2 at `C:\Users\ASUS\git_repos\trading-system-v2`. Phases 1 and 2 are complete and merged. Phase 3 is in progress on branch `feat/phase-3-lightgbm`. Read PLAN.md for context. The next step is wiring MetaModel into run_server.py and position_manager.py, adding the trade_features DB table, and wiring the Sunday retrain job in scheduler.py."

---

## How to Resume in a New Session

```
Tell the AI:
"We are building trading-system-v2 at C:\Users\ASUS\git_repos\trading-system-v2.
Read PLAN.md for full context. The source repo to copy files from is at
C:\Users\ASUS\TradingSystem. Continue from the Files Status section in PLAN.md —
pick the first TODO item and build it on a new branch off main."
```

---

## Architecture — Key Decisions (Immutable)

### Signal Flow (Meta-Labeling Pattern)
```
PRIMARY MODEL:  signal_basket.py  →  BUY/SELL direction  (rule-based, auditable)
META MODEL:     LightGBM          →  P(win)              →  position size multiplier
```
LightGBM does NOT override the signal basket direction. It only controls bet size.
In Phase 1 (no LightGBM yet), default position size = config `base_size_pct`.

### Universe — 3 Layers
```
Layer 1: Static Watchlist    → config.json watchlist → scored every 60 minutes
Layer 2: HOT_UNIVERSE        → event-triggered TTL cache → scored immediately on entry
Layer 3: TradingView Screener → proactive batch scan → feeds HOT_UNIVERSE every 5 min
```

### HOT_UNIVERSE Triggers
| Source | Trigger | TTL |
|--------|---------|-----|
| Alpaca NewsDataStream("*") | FinBERT > 0.70 AND threat NOT CRITICAL/HIGH | 4h |
| SEC EDGAR 8-K RSS | Material event (item 1.01, 5.02, 7.01) | 6h |
| SEC EDGAR Form 4 RSS | Insider buying (direction = "Buy") | 6h |
| TradingView Screener | volume_ratio > 3.0 AND price_change > 1.5% | 2h |

### SETUP_NEWS_DRIVEN Policy
Never auto-execute. Always send Telegram alert and wait for human reply ("y" = execute, anything else = skip). TTL = 30 minutes for the confirm window.

### Exit Priority (highest to lowest)
1. Hard exits: `kill_switch_active`, `trading_halt`, `going_concern_8k`, `CRITICAL` threat
2. Regime deterioration: EARLY_BEAR → exit non-INFLECTION; MID_BEAR/CRASH → exit all
3. Exit basket: `compute_exit_basket()` score exceeds setup-specific threshold
4. AGE-1 partial profit: held N days + unrealized gain > X% → sell 50% (Telegram confirm)
5. Trailing stop: ATR-based, activates after +20% gain

### Regime States (8-state detector)
```
EARLY_BULL → MID_BULL → LATE_BULL → EUPHORIA
                   ↕
               NEUTRAL
                   ↕
EARLY_BEAR → MID_BEAR → CRASH
```
Entry allowed in: EARLY_BULL, MID_BULL, NEUTRAL (reduced size)
No new longs in: LATE_BULL, EUPHORIA, any BEAR state

### Feature Vector (Phase 3+ LightGBM input)
All values are RAW NUMERICS — no pre-weighting. LightGBM learns the weights.
```
Technical:     RSI, MACD histogram, price_vs_20ma, price_vs_50ma, price_vs_200ma,
               bb_position, bb_width, volume_ratio, ATR_pct, ADX, VFI
Binary signals: rsi_oversold, rsi_overbought, macd_bullish_cross, price_above_20ma,
               bb_squeeze, volume_surge, macd_bearish_cross
Candlestick:   ~60 pandas-ta patterns (binary 0/1)
Chart patterns: ~15 YOLOv8 visual patterns (binary 0/1) — Phase 2+
OHLC geometry: body_size, body_direction, upper_wick, lower_wick, wick_ratio,
               candle_range_vs_atr, gap_up, gap_down
Fundamental:   pe_ratio, roe, revenue_growth_yoy, earnings_revision_3m, debt_equity
News/NLP:      finbert_score, news_volume_7d, threat_level (0-4)
Macro:         regime_encoded (0-7), vix, yield_spread_10y2y, fear_greed
Context:       sector_encoded, setup_type_encoded, days_since_earnings, mktcap_bucket
```

### Training Labels (Triple Barrier — Phase 3)
```
Upper barrier = ATR × 3.0   → label +1 (take profit hit first)
Lower barrier = ATR × 1.5   → label -1 (stop loss hit first)
Vertical barrier = 10 days  → label  0 (time expired)
```

---

## Key Libraries and Why Each Was Chosen

| Library | Replaces | Why |
|---------|----------|-----|
| `pandas-ta` | `ta` library | 130+ indicators, MIT, pure Python, no C dependency |
| `APScheduler` | `schedule` lib | PostgreSQL job store, asyncio, persistent, no single-threaded blocking |
| `quantstats` | custom `performance_calculator.py` | 50+ metrics, HTML tearsheets, 10 lines vs 245 |
| `alpaca-py` | `alpaca-trade-api` (archived Jun 2026) | Official SDK, options support, maintained |
| `sec-edgar-api` | FMP (quota 250/day) | Free, unlimited, direct from SEC XBRL |
| `tradingview-screener` | FMP fundamental screener | Scans 9000+ US stocks in 1 call, MIT, free |
| `lightgbm` | fixed weighted scorecard | Learns dynamic weights, factor interactions, regime-aware |
| `pybroker` | custom backtester | ML + walk-forward built in, Apache 2.0 |
| `feedparser` | requests-based RSS | Proper RFC 4287 parsing, entry dedup, encoding handling |
| `yiyanghkust/finbert-tone` | `ProsusAI/finbert` | Same labels, modern transformers API, maintained |

---

## Free Tier Compliance

All free. No paid APIs required for core functionality.

| Source | Method | Cost |
|--------|--------|------|
| Alpaca NewsDataStream("*") | WebSocket (1 connection) | $0 — pushed to you |
| SEC EDGAR RSS | REST 1/minute | $0 — 60 calls/hour |
| TradingView Screener | REST 1/5min | $0 — 12 calls/hour |
| yfinance OHLCV | REST on-demand | $0 — personal use |
| Alpaca bars (watchlist) | WebSocket (1 connection) | $0 — IEX feed |

Optional paid upgrade: Alpaca SIP real-time tape ($9/month) for true consolidated volume.

---

## Reference Repos on This Machine

| Repo | Path | Use |
|------|------|-----|
| TradingSystem (v12.15) | `C:\Users\ASUS\TradingSystem` | Copy signal_basket, regime_detector, trade_ledger |
| blm_terminal_windy | `C:\Users\ASUS\git_repos\blm_terminal_windy` | Borrow: 39-source news tier system, threat classifier pattern, orchestrator pattern |

---

## Git Workflow Rules

- `main` is protected — never commit directly to main
- Branch naming: `feat/short-description`, `fix/short-description`, `chore/short-description`
- Commit message format: `type(scope): description`
  - Types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`
  - Examples:
    - `feat(data): add EDGAR Form 4 RSS monitor for insider buying signals`
    - `fix(server): add order fill verification after submit_order()`
    - `chore(deps): pin lightgbm to 4.5.0 for reproducibility`
- Linear history: always `git rebase main` before opening PR, never merge commits
- PR title = commit message of the squashed PR commit

---

## Environment Variables Required

```
ALPACA_API_KEY          Alpaca account API key
ALPACA_SECRET_KEY       Alpaca account secret key
ALPACA_PAPER            true (paper trading) or false (live)
DATABASE_URL            postgresql://user:pass@localhost:5432/trading_system_v2
TELEGRAM_BOT_TOKEN      Telegram bot token from @BotFather
TELEGRAM_CHAT_ID        Your Telegram chat/user ID
FMP_API_KEY             (optional) FinancialModelingPrep — fallback fundamentals
OPENBB_PAT              (optional) OpenBB personal access token
```

---

## Phase 3 Bootstrap Plan (LightGBM)

**Problem:** LightGBM needs labeled trade data to train on, but we have no trades yet.

**Solution:**
1. Run Phase 1 in paper trading for 2-3 months → accumulates real trade outcomes in ledger
2. For historical bootstrap: run `triple_barrier.py` on 5 years of yfinance OHLCV for all
   watchlist symbols → generates synthetic labeled dataset retroactively
3. Combine real (Phase 1) + historical (retroactive) labels → train LightGBM
4. Walk-forward validate with PyBroker: train 2020-2023, test 2024, train 2020-2024, test 2025

**Required samples before training:** min 500 labeled trades (configurable in config.json).

---

## Config Structure Summary

All runtime behavior is controlled via `config.json`. No hardcoded values in source code.
Key sections: `watchlist`, `universe`, `signal_basket`, `risk`, `position_sizing`,
`regime`, `news`, `scheduler`, `alpaca`, `model`, `performance`.

See `config.json` for full structure with defaults.
