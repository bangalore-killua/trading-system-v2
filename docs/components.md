# Component Reference (v14)

## ML / Modeling

| Library | Role | Why |
|---------|------|-----|
| AutoGluon | Entry, dip, tranche, meta-stacking models | Tries 22+ models, ensembles winners, ONNX export |
| LightGBM | Fallback / primary entry model | Fast gradient boosting, handles class imbalance |
| PyTorch | Exit LSTM binary classifier | Sequence model for P(peak in next 5 days) |
| mlfinlab | Purged cross-validation (CPCV) | Prevents financial data leakage; honest IC ~54% |
| sentence-transformers | Cross-article news clustering | all-MiniLM-L6-v2: 80MB, CPU-fast |
| transformers (FinBERT) | News sentiment | yiyanghkust/finbert-tone, financial domain |
| chromadb | Vector store for feedback RAG | Embedded, no external service |
| onnxruntime | Model inference | 3–5× faster than PyTorch/LGBM native |

## Data Sources

| Library | Role |
|---------|------|
| tvdatafeed | OHLCV all TFs native (1m→1M); replaces yfinance for price data |
| yfinance | Fundamentals, earnings, holders, financials (quote data deferred to tvdatafeed) |
| alpaca-py | Real-time bars + news WebSocket + order execution |
| fredapi | FRED macro data: yield curve, credit spread, VIX term structure |
| feedparser | RSS parsing for PRNewswire, BusinessWire, Google News |
| playwright | Browser automation for Earnings Whispers, Seeking Alpha (Unusual Whales removed in v14) |
| requests | SEC EDGAR JSON API, CapitolTrades, Senate EFTS |
| pytrends | Google Trends (retail attention signal) |
| tradingview-screener | Momentum expansion, daily mover fetch (all US stocks, free) |
| networkx | Knowledge graph (supply-chain DiGraph from SEC 10-K) |
| finvizfinance | ETF holdings for theme basket expansion |
| spacy | NER for company name → ticker extraction |
| pandas-ta | 130+ technical indicators + 60 CDL candlestick patterns |
| vectorbt | Vectorized backtester (intelligent_backtest.py) |
| scipy | Spearman correlation for IC tracking |

## LLM / AI (v14 — 2 tiers, Haiku removed)

| Library | Model | Role | Cost |
|---------|-------|------|------|
| groq | llama-3.3-70b-versatile | All routine verdicts: tranche, exit, dip, symbol classification | Free (14,400 req/day) |
| anthropic | claude-sonnet-4-6 | Final synthesis only: anomaly detection, narrative shift | ~$2/mo |

> **v14 change**: Claude Haiku removed — Groq (free) handles all verdict/classification tasks at equivalent quality.

## Structured Logging (v14 — new)

| Component | Role |
|-----------|------|
| `data/logging/trade_logger.py` | SQLite WAL + JSONL backup, queue-based writer, 12 stage types |
| `logs/trade_lifecycle.db` | Primary store: O(1) lookup by trace_id |
| `logs/structured/trades_YYYYMMDD.jsonl` | Human-readable backup (plain-English why field) |
| `data/model_tester.py` | 5-gate deployment guard before any model swap |
| `data/model_health_reporter.py` | Weekly Sunday Telegram digest |
| `data/notifications/telegram_bot.py` | Interactive bot: /trace /status /positions /health |
| `data/parallel_runner.py` | Adaptive workers: min(cpu_count, n_symbols) |

## Backend

| Library | Role |
|---------|------|
| fastapi | REST API server (40+ endpoints) |
| uvicorn | ASGI server |
| sqlalchemy | ORM for PostgreSQL |
| alembic | Database migrations |
| apscheduler | Job orchestrator (PostgreSQL job store) |
| websockets | Alpaca WebSocket streams |

## Frontend (Terminal)

| Asset | CDN | Role |
|-------|-----|------|
| TradingView Widget | s3.tradingview.com/tv.js | Main chart (free embed) — all TFs, dark theme, volume, MA, RSI |
| Canvas API | Native browser | Tranche overlay on top of TradingView iframe |
| Cytoscape.js | CDN | Knowledge graph visualisation |

**v13 note:** TradingView widget replaced the custom Lightweight Charts implementation.
9 timeframe buttons (1m/5m/15m/30m/1H/4H/1D/1W/1M) each update `TradingView.widget({ interval })`.

## Algebraic Pattern Detector (replaces YOLOv8)

`data/sources/pattern_detector.py` — 12 patterns, all algebraic, no model file required.

| Pattern | Method |
|---------|--------|
| Cup and handle | Peak1 / trough / Peak2 geometry + depth 15–35% + recovery > 85% |
| Head and shoulders | 3-peak structure with shoulders below head |
| Bull flag | Tight consolidation after strong move |
| Double bottom | Two troughs at similar price level |
| Breakout | Price exceeds N-period high on volume confirmation |
| VWAP reclaim | Price re-crosses VWAP after pullback |
| RSI divergence | Price lower low but RSI higher low |
| Support bounce | Touch of defined support level |
| Momentum thrust | ADX > 25 + directional movement confirmation |
| Volume surge | Volume > 2× 20-day average |
| Moving average crossover | EMA50 crosses EMA200 |
| Bollinger squeeze | BB width < 20th percentile → expansion signal |

Advantages over YOLOv8:
- Deterministic — same input always gives same output
- No training data, no GPU, no image pipeline
- Microseconds vs milliseconds
- Transparent — you can read exactly why it fired
- Not sensitive to chart rendering style

## SourceWithFallback Base Class

All Playwright scrapers inherit from `data/sources/base_source.py`:

```python
class SourceWithFallback:
    name: str
    fallbacks: list  # ordered, tried in sequence

    def get(self, symbol: str):
        # 1. Try primary (Playwright)
        # 2. Try each fallback in order
        # 3. Telegram alert on degradation (1h cooldown)
        # 4. Return None if all fail → caller uses 0.5 neutral default
```

Fallback hierarchy:

| Source | Primary | Fallback 1 | Fallback 2 | If all down |
|--------|---------|-----------|-----------|-------------|
| unusual_whales | unusualwhales.com | Barchart options | yfinance chain | score = 0.5 |
| earnings_whispers | earningswhispers.com | EW RSS | yfinance estimate | use official |
| seeking_alpha | seekingalpha.com | Google News RSS | skip | skip |
| research_agent | Playwright browse | memory-only + Haiku | skip | use cached |

PRNewswire, BusinessWire, and StockTwits are RSS/API — no Playwright, no fallback needed.

## Terminal — 11 Tabs

| Tab | Data source | API endpoint |
|-----|------------|--------------|
| Overview | yfinance.info (24 fields) | inline in analyseSymbol() |
| Chart | TradingView widget + canvas tranche overlay | /api/ohlcv/{sym} for tranche positions |
| Analysis | yfinance recommendations + Claude Haiku scenarios | /api/analysis/{sym} |
| Earnings | yfinance earnings_history + quarterly_financials | /api/earnings/{sym} |
| Financials | yfinance quarterly/annual statements | /api/financials/{sym}?period=q|a |
| Holdings | yfinance institutional/mutual/insider + congressional | /api/holdings/{sym} |
| Options | Unusual Whales + yfinance chain | /api/options/{sym} |
| Filings | SEC EDGAR sec_data.py | /api/sec/filings/{sym} |
| News | PRNewswire + BusinessWire + Google News merged | /api/news?ticker={sym} |
| Relations | Cytoscape knowledge graph | /api/graph/{sym} |
| COT | CFTC COT data | /api/cot |

## Data Sources — All Free

| Source | API key? | TTL / frequency |
|--------|----------|----------------|
| tvdatafeed | None | On demand |
| Alpaca bars + news | Free account | Real-time WebSocket |
| PRNewswire RSS | None | Every 5 min |
| BusinessWire RSS | None | Every 5 min |
| SEC EDGAR JSON | None | Every 60s (8-K/Form 4) |
| FRED API | Free registration | Hourly |
| Unusual Whales | None (Playwright) | Per request |
| Capitol Trades | None | 6h cache |
| Senate EFTS | None | 6h cache |
| Google News RSS | None | Per request |
| StockTwits | None (public API) | Per request |
| TradingView Screener | None | Every 5 min |
| yfinance | None | Per request |
| Google Trends (pytrends) | None | Per request |
