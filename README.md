# trading-system-v2

An event-driven, ML-augmented US equity trading system built on Alpaca paper/live trading.
Greenfield rebuild of [TradingSystem](../TradingSystem) with a 3-layer universe, LightGBM
meta-model, 39-source news pipeline, and properly wired signal basket.

---

## Architecture Overview

### Full Data & Signal Flow

```mermaid
flowchart TB
    subgraph INGEST["Data Ingestion — Parallel, Always-On"]
        direction LR
        I1["Alpaca NewsDataStream<br/>ALL symbols · WebSocket · free"]
        I2["SEC EDGAR RSS<br/>8-K + Form 4 · every 60s · free"]
        I3["TradingView Screener<br/>All US stocks · every 5min · free"]
        I4["39-Source RSS Feeds<br/>Tier 1/2/3 · every 5min"]
        I5["Alpaca StockDataStream<br/>Watchlist bars · WebSocket · free"]
    end

    subgraph NLP["NLP Pipeline"]
        direction LR
        NLP1["FinBERT<br/>Sentiment −1→+1"]
        NLP2["Threat Classifier<br/>CRITICAL / HIGH / MEDIUM / LOW"]
        NLP3["SpaCy NER<br/>Company → Sector → Macro"]
    end

    subgraph UNIVERSE["Universe Selection — 3 Layers"]
        direction LR
        U1["Static Watchlist<br/>screener_picks + manual + mandatory<br/>Scored every 60 min"]
        U2["HOT_UNIVERSE — TTL Cache<br/>News event: 4h · Volume spike: 2h<br/>EDGAR filing: 6h · Screener: 2h<br/>Scored immediately on entry"]
    end

    subgraph SIGNAL["Signal Generation"]
        direction TB
        S0["8-State Regime Detector<br/>EARLY_BULL → MID_BULL → LATE_BULL → EUPHORIA<br/>NEUTRAL → EARLY_BEAR → MID_BEAR → CRASH"]
        S1["detect_primary_setup()<br/>MOMENTUM · BOUNCE · INFLECTION · NEWS_DRIVEN"]
        S2["compute_entry_basket()<br/>27 weighted signals → score 0–100"]
        S3["LightGBM Meta-Model<br/>Phase 3+ · P(win) → position size multiplier"]
    end

    subgraph RISK["Risk Checks — All Must Pass"]
        direction TB
        R1["Regime Gate · No longs in BEAR / CRASH"]
        R2["Threat Gate · CRITICAL blocks any entry"]
        R3["Sector Concentration · Max 30% per sector"]
        R4["Correlation · ρ > 0.8 → reduce size"]
        R5["Cash Reserve · Min 20% always"]
    end

    subgraph EXEC["Execution"]
        direction LR
        E1["alpaca-py submit_order()"]
        E2["Poll get_order() until FILLED"]
        E3["Trade Ledger · PostgreSQL · Telegram"]
    end

    subgraph MONITOR["Position Monitoring — 5-min cycle"]
        direction TB
        M1["compute_exit_basket() · 13 exit signals"]
        M2["Hard Exits · kill_switch · CRITICAL news"]
        M3["AGE-1 Partial Profits · Telegram confirm"]
        M4["Trailing Stop · ATR-based · activates at +20%"]
    end

    I1 & I4 --> NLP
    NLP --> |"bullish + not CRITICAL"| U2
    I2 --> |"material filing"| U2
    I3 --> |"unusual volume"| U2
    I5 --> S0

    U1 --> |"60-min batch"| SIGNAL
    U2 --> |"immediate trigger"| SIGNAL

    S0 --> S1 --> S2 --> S3
    S3 --> RISK

    RISK --> |"all checks pass<br/>not NEWS_DRIVEN"| EXEC
    RISK --> |"SETUP_NEWS_DRIVEN"| TG["📱 Telegram Confirm<br/>Human approves / rejects"]
    TG --> |"approved"| EXEC

    EXEC --> MONITOR
    MONITOR --> |"exit signal fired"| EXEC
```

### Exit Priority Order

```mermaid
flowchart LR
    P1["Priority 1<br/>Hard Exits<br/>kill_switch · trading_halt<br/>going_concern_8K · CRITICAL news"]
    P2["Priority 2<br/>Regime Exit<br/>drops to EARLY_BEAR → exit non-INFLECTION<br/>MID_BEAR / CRASH → exit all"]
    P3["Priority 3<br/>Exit Basket<br/>compute_exit_basket() score<br/>exceeds setup threshold"]
    P4["Priority 4<br/>AGE-1 Partial Profit<br/>held N days + gain > X%<br/>sell 50%, keep remainder"]
    P5["Priority 5<br/>Trailing Stop<br/>ATR-based · activates at +20%<br/>trails 2× ATR below recent high"]

    P1 --> P2 --> P3 --> P4 --> P5
```

### LightGBM Meta-Model (Phase 3+)

```mermaid
flowchart LR
    FV["Feature Vector<br/>• 24 scorecard factors (raw values)<br/>• 60 candlestick patterns (pandas-ta)<br/>• 15 chart patterns (YOLOv8)<br/>• OHLC geometry features<br/>• FinBERT score + threat level<br/>• Regime + VIX + yield spread<br/>• Sector + setup type + earnings gap"]
    LGB["LightGBM<br/>Meta-Model<br/>trained on triple-barrier labels"]
    OUT["P(win) 0.0→1.0<br/>→ position size multiplier<br/>≥ 0.65 → full size<br/>0.52–0.65 → half size<br/>< 0.52 → skip"]

    FV --> LGB --> OUT
```

---

## Repository Structure

```
trading-system-v2/
│
├── README.md                        ← this file — architecture + flowcharts
├── PLAN.md                          ← SESSION RESTART GUIDE — read this in new chats
├── requirements.txt                 ← all Python dependencies
├── config.json                      ← all runtime config (no hardcoded values in code)
├── .env.example                     ← environment variable template
├── .gitignore
├── database.py                      ← SQLAlchemy models + async engine
├── run_server.py                    ← entry point: starts FastAPI + APScheduler
│
├── data/                            ← all data ingestion and caching
│   ├── __init__.py
│   ├── cache_manager.py             ← 8-layer TTL cache, atomic JSON writes
│   ├── fetcher.py                   ← OHLCV via yfinance + alpaca-py bars
│   ├── fundamentals_fetcher.py      ← sec-edgar-api primary, FMP fallback
│   ├── hot_universe.py              ← TTL cache for event-driven symbol discovery
│   ├── news_sources.py              ← 39-source registry (Tier 1 / 2 / 3)
│   ├── news_aggregator.py           ← RSS fetch + dedup + NLP pipeline trigger
│   ├── news_relevance_engine.py     ← SpaCy NER → sector → macro causal rules
│   ├── news_stream.py               ← Alpaca NewsDataStream("*") WebSocket
│   ├── edgar_monitor.py             ← SEC EDGAR 8-K + Form 4 RSS polling
│   ├── tv_screener.py               ← TradingView Screener (MIT, no API key)
│   └── price_stream.py              ← Alpaca StockDataStream WebSocket
│
├── strategy/                        ← signal generation, risk, ML model
│   ├── __init__.py
│   ├── signal_basket.py             ← [COPIED] entry/exit basket computation
│   ├── signal_types.py              ← [COPIED] SetupType enum + constants
│   ├── regime_detector.py           ← [COPIED] 8-state regime detection
│   ├── trade_ledger.py              ← [COPIED] per-symbol markdown + master log
│   ├── position_aging.py            ← [COPIED] AGE-1 partial profit logic
│   ├── correlation_risk.py          ← [COPIED] portfolio correlation checks
│   ├── holding_tier.py              ← [COPIED] T1/T2/T3 tier classification
│   ├── trade_state.py               ← [COPIED] in-memory position state
│   ├── trade_labels.py              ← [COPIED] trade outcome labeling
│   ├── config_loader.py             ← [COPIED] typed config.json loader
│   ├── technical_analysis.py        ← pandas-ta indicators + candlestick patterns
│   ├── threat_classifier.py         ← word-list NLP: CRITICAL/HIGH/MEDIUM/LOW
│   ├── news_sentiment.py            ← FinBERT wrapper (yiyanghkust/finbert-tone)
│   ├── macro_analyser.py            ← VIX + yield curve + FRED + Fear&Greed
│   ├── position_manager.py          ← entry/exit logic with signal basket WIRED
│   ├── scorer.py                    ← feature vector builder (input to LightGBM)
│   ├── meta_model.py                ← [Phase 3] LightGBM multi-class wrapper, WF-validate, save/load
│   ├── triple_barrier.py            ← [Phase 3] TP/SL/time label generation (López de Prado)
│   ├── training_pipeline.py         ← [Phase 3] bootstrap + live-label retrain CLI
│   ├── screener_guardrails.py       ← [COPIED] fundamental quality filters
│   └── performance_calculator.py   ← quantstats wrapper
│
├── server/                          ← API, scheduling, broker
│   ├── __init__.py
│   ├── api.py                       ← FastAPI routes
│   ├── scheduler.py                 ← APScheduler orchestrator (replaces schedule lib)
│   ├── alpaca_trader.py             ← alpaca-py broker client + fill verification
│   ├── order_manager.py             ← order dedup + retry logic
│   ├── kill_switch.py               ← emergency halt
│   └── telegram_alerts.py          ← Telegram bot notifications
│
├── models/                          ← trained ML model artifacts (Phase 3+)
│   └── .gitkeep
│
├── logs/                            ← trade ledger, backtest results, performance
│   ├── trade_ledger/
│   ├── backtest_ledger/
│   └── performance/
│
└── tests/                           ← pytest test suite
    ├── conftest.py
    ├── test_signal_basket.py        ← [PORTED from TradingSystem]
    ├── test_regime_detector.py      ← [PORTED]
    ├── test_threat_classifier.py    ← new
    ├── test_hot_universe.py         ← new
    └── test_technical_analysis.py  ← new
```

---

## Build Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Working live system: event-driven universe, 39-source news, signal basket wired, paper trading |
| **Phase 2** | ✅ Complete | YOLOv8 visual chart pattern detection (16 patterns), ~125-feature vector builder, fundamentals fetcher |
| **Phase 3** | ✅ Complete | LightGBM meta-model: triple-barrier labeling, walk-forward training, P(win) position sizing |
| **Phase 4** | 🔨 In Progress | Options expansion: long calls, protective puts, covered calls via alpaca-py |

### Phase 3 — LightGBM Bootstrap Strategy

```
Two data paths feed the meta-model:

PATH A (retroactive bootstrap, run once):
  yfinance 5yr OHLCV → TripleBarrierLabeler → feature vectors → ~50k labeled samples

PATH B (live paper trades, ongoing):
  Paper trade closes → triple-barrier label recorded → appended to training set

Train when: total samples ≥ config.json model.retrain_min_samples (default 500)
Validate:   TimeSeriesSplit walk-forward, n_splits=5 (no lookahead bias)
Threshold:  P(win) ≥ 0.65 → full size  |  0.52–0.65 → half size  |  < 0.52 → skip
```

**Phase 3 CLI:**
```bash
# Bootstrap from 5yr historical data for watchlist symbols
python -m strategy.training_pipeline --years 5

# Bootstrap for specific symbols
python -m strategy.training_pipeline --symbols AAPL MSFT NVDA GOOGL --years 5

# Retrain using only live paper trade labels
python -m strategy.training_pipeline --live-only

# Re-validate without re-fetching data
python -m strategy.training_pipeline --validate-only
```

### Phase 4 — Options Strategies

Three strategies controlled by `config.json options` section. All disabled by default (`options.enabled: false`).

| Strategy | When | Contract |
|----------|------|----------|
| **Long call** | Bullish entry + LightGBM P(win) ≥ 0.65 | ATM call, delta 0.40–0.60, DTE 25–50 days |
| **Protective put** | Regime → LATE_BULL or EARLY_BEAR | OTM put 7% below price, DTE 30–60 days |
| **Covered call** | After AGE-1 partial profit on equity | OTM call 7% above price, DTE 14–21 days |

**Exit rules (all positions):**
- Stop loss: mid drops 50% from entry premium
- Take profit: mid doubles from entry premium
- DTE close: ≤ 7 days to expiry (avoid gamma/pin risk)

**Enable options:**
```json
// config.json
"options": {
    "enabled": true,
    "strategies_enabled": ["long_call", "protective_put"]
}
```

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/bangalore-killua/trading-system-v2.git
cd trading-system-v2
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure environment
cp .env.example .env
# Edit .env with your Alpaca API keys, DB URL, Telegram token

# 3. Set up database
createdb trading_system_v2
python -c "from database import create_tables; create_tables()"

# 4. Run (paper trading by default)
python run_server.py
```

---

## Git Workflow

- `main` branch is protected — no direct commits
- All work on feature branches: `feat/`, `fix/`, `chore/`
- Linear history: use rebase, not merge commits
- PR required to merge into main

```bash
# Start new work
git checkout main && git pull --rebase
git checkout -b feat/your-feature-name

# Commit often with descriptive messages
git add -p
git commit -m "feat(data): add EDGAR Form 4 RSS monitor for insider buying"

# Rebase before PR
git rebase main

# Push and open PR
git push -u origin feat/your-feature-name
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Broker | Alpaca | Paper + live, free, supports options |
| Signal direction | Rule-based signal basket | Auditable, backtestable, deterministic |
| Bet sizing | LightGBM meta-model | Learns which conditions actually predict wins |
| Training labels | Triple barrier (TP/SL/time) | Matches real trade outcomes, not naive forward return |
| Universe discovery | WebSocket + RSS (not polling 9000 stocks) | Free tier compatible, event-driven |
| Scheduler | APScheduler | PostgreSQL job store, asyncio, persistent across reboots |
| Technical indicators | pandas-ta | 130+ indicators, MIT, pure Python, no C library |
| News NLP | FinBERT + word-list threat classifier | FinBERT for sentiment, word-list for speed on threat triage |
| SETUP_NEWS_DRIVEN | Never auto-execute | Human Telegram confirm required — too volatile |
