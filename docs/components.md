# Component Reference

## Open Source Libraries Used

### ML / Modeling

| Library | Version | Role | Why |
|---|---|---|---|
| LightGBM | latest | Core trading signal model | Fast gradient boosting, handles class imbalance, interpretable feature importance |
| sentence-transformers | >=2.7.0 | Cross-article news clustering | all-MiniLM-L6-v2: 80MB, CPU-fast, excellent semantic similarity |
| transformers (FinBERT) | latest | News sentiment scoring | Pre-trained financial domain BERT, most accurate for financial text |
| ultralytics (YOLOv8) | fork v8.3.40 | Chart pattern detection | 16 patterns, custom trained, real-time inference |
| optuna | >=3.6 | LightGBM hyperparameter tuning | Efficient Bayesian optimization with pruning |
| chromadb | >=0.5.0 | Vector store for feedback RAG | Embedded DB, no external service needed |

### Data

| Library | Version | Role |
|---|---|---|
| yfinance | latest | Stock quotes, OHLCV, fundamentals, options, news |
| finvizfinance | >=0.14 | ETF holdings for theme basket expansion |
| networkx | >=3.3 | Knowledge graph (supply-chain DiGraph) |
| pandas-datareader | - | FRED macro data fallback (no API key needed) |
| fredapi | - | FRED macro data (requires free API key) |
| requests | latest | SEC EDGAR JSON API, CFTC CSV download |
| tradingview-screener | - | Momentum expansion + daily mover fetch |
| spacy | >=3.7.0 | NER for company name → ticker extraction |

### LLM

| Library | Version | Role |
|---|---|---|
| groq | >=0.4.0 | Llama 3.3 70B client (free tier: 14,400 req/day) |
| agno | >=1.0.0 | Agent framework for portfolio/risk/signal agents |

### Backend

| Library | Role |
|---|---|
| fastapi | REST API server |
| uvicorn | ASGI server |
| sqlalchemy | ORM for PostgreSQL |
| alembic | Database migrations |
| alpaca-py | Broker API (paper + live) |

### Frontend

| Library | CDN | Role |
|---|---|---|
| Lightweight Charts v5 | unpkg | Candlestick/sparkline charts |
| Cytoscape.js 3.30 | unpkg | Knowledge graph spider-web visualization |

## Model Files (gitignored)

```
models/
  scorecard_meta.json     ← tracked in git (version manifest only)
  scorecard_v1.pkl        ← gitignored
  scorecard_v2.pkl        ← gitignored
  chart_patterns_yolov8n.pt  ← gitignored
  knowledge_graph.json    ← gitignored
```

## API Endpoints Summary

### Trading & Signals
- `GET /api/signals` — current watchlist signals with P(win)
- `GET /api/positions` — current open positions
- `GET /api/performance` — win rate, profit factor, drawdown

### Market Data (via Fincept modules)
- `GET /api/macro/snapshot` — FRED 5-indicator macro snapshot
- `GET /api/macro/fred/{series_id}` — arbitrary FRED series
- `GET /api/global/indices` — SPY/QQQ/GLD/VIX/futures quotes
- `GET /api/ohlcv/{symbol}` — OHLCV in Lightweight Charts format
- `GET /api/stock/{symbol}` — quote + fundamentals + news
- `GET /api/options/{symbol}` — options chain with BS Greeks
- `GET /api/cot` — COT positioning for 10 futures contracts
- `GET /api/sec/filings/{symbol}` — SEC filings + Form 4 insider trades

### Intelligence
- `GET /api/news` — recent articles with FinBERT sentiment
- `GET /api/news/analysis` — sentence-transformers + Groq portfolio analysis
- `GET /api/graph/{symbol}` — Cytoscape.js knowledge graph
- `GET /api/patterns/{symbol}` — YOLOv8 chart pattern detections
- `GET /api/ai/agent/{agent_name}` — Groq AI agent report

### UI
- `GET /dashboard` — main trading dashboard
- `GET /terminal` — Bloomberg-style research terminal
