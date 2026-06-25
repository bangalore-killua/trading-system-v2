# PLAN.md — Session Restart Guide

> **READ THIS FIRST IN ANY NEW CHAT SESSION.**
> This document is the single source of truth for continuing work on trading-system-v2.
> It is self-contained — a new session needs only this file + the codebase to resume.

---

## What This Project Is

`trading-system-v2` is a fully automated US equity trading system on Alpaca (paper + live).
It runs on a private VPS (Hetzner CX22, €4.49/mo). AGPL-3.0, personal use only.

**Core philosophy (v13):** signals are SCORES that drive position SIZE, not gates.
Composite score `= Σ(signal_i × IC[signal_i][regime])` → half-Kelly sizing. Only 2 hard gates: CRASH regime + cash < 10%.

---

## Current Build Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Foundation | ✅ Complete | 3-layer universe, 39-source news, signal basket wired |
| Phase 2 — Chart patterns + feature vector | ✅ Complete | YOLOv8 (kept as legacy), scorer.py, fundamentals |
| Phase 3 — LightGBM meta-model | ✅ Complete | Triple-barrier labels, walk-forward CV |
| Phase 4 — Options expansion | ✅ Complete | Long calls, protective puts, covered calls |
| Phase 5 — Wiring + migrations | ✅ Complete | Alembic, consensus fixes, signal wiring |
| **v13 — Full intelligence upgrade** | ✅ Complete | Scores-not-gates · alpha-beta · 3-tier AI · LSTM exit · dip detector · tranche manager · 14 new data sources · intelligent backtester |

**All phases complete. System is ready for backtester bootstrap → model training → VPS deploy.**

---

## Next Steps (in priority order)

### 1. Backtester bootstrap (CRITICAL — unblocks all ML)
```bash
# Generate labeled training dataset with alpha-beta separation
python backtester/intelligent_backtest.py --symbol NVDA AAPL MSFT GOOGL AMZN --years 5
# Validate entry model IC > 0 before building exit model
python -m strategy.training_pipeline --symbols NVDA AAPL MSFT GOOGL AMZN --years 5
```
Entry model must be validated first (IC > 0). Exit model (LSTM) only after.

### 2. VPS deployment (Hetzner CX22 — €4.49/mo)
```bash
# On VPS:
git clone https://github.com/bangalore-killua/trading-system-v2.git
pip install -r requirements.txt
cp .env.example .env && nano .env
alembic upgrade head
python run_server.py    # starts paper trading + dashboard
```
See memory: `project_vps_server.md`

### 3. Telegram setup (required before VPS deploy)
```
/newbot on @BotFather → get token → send any message to bot → curl getUpdates → get chat_id
```
Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env`.

### 4. Phase 14 (deferred — after 6–12 months real paper-trade data)
- FinRL-X: RL framework, continuous action space, A/B test vs AutoGluon
- Temporal Fusion Transformer: mixed-frequency exit model (replace LSTM if TFT outperforms)
- Polygon.io ($29/mo): survivorship-bias-free historical data (includes delisted stocks)

---

## Repo Info

- Repo: `C:\Users\ASUS\git_repos\trading-system-v2`
- GitHub: **private** at `https://github.com/bangalore-killua/trading-system-v2`
- Branch policy: main protected · feature branches → PR → squash merge
- Security: `.env` never committed · no API keys in git history · repo stays private

---

## v13 New Files Reference

### New Data Sources
| File | What |
|------|------|
| `data/sources/tvdatafeed_client.py` | OHLCV all TFs native (replaces yfinance for OHLCV) |
| `data/sources/alpaca_feed.py` | Real-time US stock data (free with Alpaca account) |
| `data/sources/pattern_detector.py` | Algebraic pattern detection (replaces YOLOv8) |
| `data/sources/fred_macro.py` | FRED API — yield curve, credit spread, VIX |
| `data/sources/unusual_whales.py` | Options flow via Playwright + Barchart fallback |
| `data/sources/prnewswire.py` | PRNewswire RSS — credibility 1.0, 22-min edge |
| `data/sources/businesswire.py` | BusinessWire RSS — credibility 1.0, 22-min edge |
| `data/sources/stocktwits.py` | Retail sentiment via free public API |
| `data/sources/earnings_whispers.py` | Whisper numbers — Playwright + RSS fallback |
| `data/sources/seeking_alpha.py` | Headlines — Playwright + Google News fallback |
| `data/sources/base_source.py` | SourceWithFallback base class |
| `data/sources/browser_scraper.py` | Playwright base class |
| `data/notifications/telegram.py` | Telegram bot client, rate-limited |
| `data/notifications/alerts.py` | Alert type constants |
| `data/fincept/agents/research_agent.py` | Claude Haiku + Playwright browser research |
| `data/fincept/congressional_trades.py` | STOCK Act House + Senate, 6h cache |
| `data/fincept/google_news.py` | Google News RSS, credibility 0.60 |

### New Strategy Files
| File | What |
|------|------|
| `backtester/intelligent_backtest.py` | vectorbt · alpha-beta separated · multi-regime oracle |
| `strategy/signal_orchestrator.py` | Composite scoring · IC weighting · meta-stacking · Claude Sonnet gate |
| `strategy/exit_monitor.py` | LSTM binary classifier P(peak in next 5 days) |
| `strategy/dip_detector.py` | AutoGluon dip model + dip_verdict() |
| `strategy/tranche_manager.py` | T1/T2/T3 + dip tranches · add-up vs add-down |

### Modified Files (major changes)
| File | What changed |
|------|------|
| `strategy/consensus.py` | 3-tier AI routing · exit/dip/tranche verdicts · RAG fixes |
| `strategy/position_manager.py` | Half-Kelly sizing · ATR stops · portfolio-level risk controls |
| `strategy/training_pipeline.py` | AutoGluon + mlfinlab CPCV · honest IC ~54% |
| `data/feedback_loop.py` | Alpha-beta separation · 9 label types · regime-matched RAG |
| `data/screener.py` | 8 factor scores (was 5) · all 0–1 normalised (was 0–100 percentile) |
| `data/fincept/yfinance_data.py` | +4 functions: get_analysis, get_earnings_data, get_financials_detailed, get_holdings_data |
| `server/api.py` | +4 endpoints: /api/analysis · /api/earnings · /api/financials · /api/holdings |
| `dashboard/terminal.html` | TradingView widget · 9 TF buttons · tranche canvas · 11 tabs · 2 position panels |

---

## Environment Variables

```bash
# Broker (required)
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # switch to live when ready

# AI (required)
GROQ_API_KEY=        # free at console.groq.com
ANTHROPIC_API_KEY=   # Claude Haiku + Sonnet

# Data (free, optional but recommended)
FRED_API_KEY=        # free at fred.stlouisfed.org

# Notifications (optional but recommended)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Database
DATABASE_URL=postgresql://user:pass@localhost/trading_system_v2
```

---

## How to Resume in a New Session

> "We are building trading-system-v2 at `C:\Users\ASUS\git_repos\trading-system-v2`.
> All phases through v13 are complete on `main`. Read PLAN.md.
> Next: run the intelligent backtester to generate training data, then train the entry model.
> VPS deploy (Hetzner CX22) planned after first model training run."
