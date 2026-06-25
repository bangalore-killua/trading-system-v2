# System Architecture (v13)

## Full Signal Pipeline

```mermaid
flowchart TB
    subgraph INGEST["Data Ingest — Always-On"]
        direction LR
        I1["Alpaca NewsDataStream\nALL symbols · WebSocket"]
        I2["SEC EDGAR RSS\n8-K + Form 4 · 60s poll"]
        I3["TradingView Screener\n9000+ US stocks · 5 min"]
        I4["PRNewswire RSS\ncredibility 1.0 · 22-min edge"]
        I5["BusinessWire RSS\ncredibility 1.0 · 22-min edge"]
        I6["tvdatafeed\n1m/5m/15m/30m/1H/4H/1D/1W/1M native"]
        I7["Alpaca StockDataStream\nwatchlist bars · real-time"]
    end

    subgraph ALT["Alternative Data Layer"]
        direction TB
        A1["Unusual Whales\noptions call/put ratio\nPlaywright + Barchart fallback"]
        A2["FRED API\nyield curve (T10Y2Y)\ncredit spread (BAMLH0A0HYM2)\nVIX term structure"]
        A3["Congressional Trades\nCapitolTrades + Senate EFTS\n6h cache · free"]
        A4["Earnings Whispers\nwhisper vs consensus\nPlaywright + RSS fallback"]
        A5["StockTwits\nretail sentiment\nfree public API"]
        A6["Google News RSS\ncredibility 0.60\nheadline velocity signal"]
    end

    subgraph NLP["NLP Pipeline"]
        direction LR
        N1["FinBERT\nyiyanghkust/finbert-tone\nsentiment −1→+1"]
        N2["SpaCy NER\ncompany name → ticker\n5000+ entity map"]
        N3["Headline velocity\nrate-of-change\nvs count"]
        N4["Threat classifier\nword-list NLP\nCRITICAL/HIGH/MEDIUM/LOW"]
    end

    subgraph UNIV["Universe — 3 Layers"]
        direction TB
        U1["Static watchlist\nscreener picks · mandatory · seeds\n60-min score cycle"]
        U2["HOT_UNIVERSE TTL cache\nNews event: 4h · EDGAR 8K: 6h\nForm 4: 6h · Vol spike: 2h\nScored immediately on entry"]
    end

    subgraph SCORES["8 Factor Scores (all 0–1, IC-weighted per regime)"]
        direction LR
        SC1["momentum_score\n12-1m return vs peers"]
        SC2["quality_score\nROE vs sector"]
        SC3["technical_score\nRSI vs own history"]
        SC4["valuation_score\nP/E vs sector (lower=better)"]
        SC5["news_score\nFinBERT 7d + velocity"]
        SC6["options_score\ncall/put ratio + unusual premium"]
        SC7["macro_score\nyield curve + credit + VIX term"]
        SC8["insider_score\nForm 4 + congressional\nrecency-weighted"]
    end

    subgraph ORCH["Signal Orchestrator"]
        direction TB
        O1["composite = Σ signal_i × IC[signal_i][regime]"]
        O2["position_size = half_kelly × composite × regime_scalar"]
        O3["IC tracking: Spearman(signal, alpha) rolling 90d per regime"]
        O4["4H EMA50 > EMA200 feature (paradigm-agnostic Sharpe lift)"]
        O5["1D regime slope = sign(EMA200[-1] − EMA200[-20])"]
    end

    subgraph MODELS["Decision Models (build in order)"]
        direction LR
        M1["Entry — LightGBM\nP(win) · triple-barrier labels\nalpha-adjusted\nvalidate IC > 0 first"]
        M2["Exit — LSTM binary\nP(peak in next 5 days)\n30-day sequence input\nnot a forecaster"]
        M3["Dip — AutoGluon\nP(recovery)\ndrop cause + fundamental"]
        M4["Tranche — LightGBM\nP(continuation)\npyramid up/down"]
        M5["Meta-stacking — AutoGluon\nconflict resolution\n2000+ conflict cases needed"]
    end

    subgraph AI["3-Tier AI Routing"]
        direction TB
        AI1["Tier 1 — Groq Llama 3.3 70B\nFree · 14,400 req/day\ntranche_verdict · routine consensus"]
        AI2["Tier 2 — Claude Haiku\nclaude-haiku-4-5-20251001\n~$0.72/mo\nexit_verdict · dip_verdict\nresearch_agent · scenarios"]
        AI3["Tier 3 — Claude Sonnet\nclaude-sonnet-4-6\n~$2/mo\nfinal synthesis · anomaly detection\n5–10 calls/day"]
        AI4["Research Agent\nHaiku + Playwright\nbrowses EDGAR · news · ratings\npasses real-time context to Sonnet"]
    end

    subgraph HARD["Hard Constraints (non-ML, non-negotiable)"]
        direction LR
        H1["ATR stop → SELL_ALL"]
        H2["Cash < 10% → no buys"]
        H3["Symbol > 22% → no adds"]
        H4["CRASH regime → no entries"]
        H5["VIX > 40 → sizes × 0.5"]
        H6["Drawdown > 12% → pause"]
        H7["5 losses → alert human"]
    end

    subgraph EXEC["Execution"]
        E1["alpaca-py submit_order()\nfill verification polling\nTelegram trade alerts"]
    end

    subgraph FB["Feedback + Self-Improvement"]
        direction TB
        F1["alpha = trade_return − spy_return × beta\nstored at trade close"]
        F2["9 label types\ncorrect_peak_exit · early_exit · late_exit\ncorrect_dip_buy · caught_falling_knife\ncorrect_hold · correct_scale_up · etc"]
        F3["ChromaDB RAG\nregime-matched queries\nwhere regime = current_regime"]
        F4["AutoGluon retrain\nmonthly · mlfinlab CPCV\nIC validation · auto-rollback"]
    end

    I1 & I4 & I5 --> N1
    I1 & I4 & I5 --> N4
    I1 & I5 --> N2
    I2 --> U2
    I3 --> U2
    I6 & I7 --> ORCH
    A1 --> SC6
    A2 --> SC7
    A3 & A4 --> SC8
    A5 & A6 --> N1
    N1 --> SC5
    N2 --> U2
    N3 --> SC5

    U1 --> SCORES
    U2 --> SCORES
    SCORES --> ORCH
    ORCH --> MODELS
    MODELS --> AI
    AI --> HARD
    HARD --> EXEC
    EXEC --> FB
    FB --> |"IC update"| ORCH
```

---

## Tranche Management

```mermaid
flowchart TD
    ENTRY["New signal: composite_score ≥ 0.35"]
    T1["Tranche 1 — 33% of full Kelly\nAlways fires on threshold breach\n'Starter position'"]
    CHECK2["Price confirms:\nprice > entry + 1.5×ATR\nAND score still ≥ 0.35?"]
    T2["Tranche 2 — +33%\n'Price confirmed the thesis'"]
    CHECK3["New high AND\nscore ≥ regime_threshold?\n(EARLY_BULL: 0.58\nMID_BULL: 0.65\nLATE_BULL: 0.74)"]
    T3["Tranche 3 — +34%\nFull position\n'High conviction'"]
    DIP["Dip Tranche — 0.8–1.5× T1\ndip_score ≥ 0.65\nfundamentals intact\nregime bullish\nLarger than T1 — price is lower"]

    EXIT1["E1 partial — 20–50%\nexit_score ≥ 0.65\nor news deterioration"]
    EXIT2["E2 partial — 30–80% of remaining\nexit_score ≥ 0.75"]
    EXIT3["E3 full exit\nexit_score ≥ 0.88\nor ATR stop"]
    CORE["Core hold — 10–20%\nlong-term conviction intact"]

    ENTRY --> T1 --> CHECK2
    CHECK2 -->|Yes| T2 --> CHECK3
    CHECK2 -->|No| HOLD["Hold T1"]
    CHECK3 -->|Yes| T3
    CHECK3 -->|No| HOLD2["Hold T1+T2"]
    ENTRY --> |"Dip event"| DIP

    T3 --> EXIT1 --> EXIT2 --> EXIT3
    EXIT3 --> CORE
```

---

## Intelligent Backtester

```mermaid
flowchart LR
    DATA["yfinance 5yr OHLCV\n+ news timestamps\n+ filing dates"]
    PIT["Point-in-Time Correctness\nearnings: use release date\nForm 4: +2 biz days\n13F: +45 days\nnews: pub timestamp only"]
    TB["Triple-Barrier Labels\nTP = 2.0×ATR (profit)\nSL = 1.5×ATR (loss)\ntime = 20 days (neutral)"]
    AB["Alpha-Beta Separation\nalpha = trade_return\n− spy_return × stock_beta\nstore alpha, NOT raw return"]
    REG["4 Regime Windows\nbull: 2020-04 → 2021-12\ncrash: 2022-01 → 2022-12\nrecovery: 2023-01 → 2024-06\nsideways: 2024-07 → 2025-12"]
    RS["robust_sharpe\n= min(sharpe per window)\nworst-case, not average"]
    MO["Multi-objective oracle\nprofit ≥ 20% AND\navg_position ≥ 5% AND\nnot Pareto-dominated"]
    BAH["Buy-and-hold benchmark\nequal-weight portfolio\nalongside every run"]
    OUT["Labeled dataset\none record per position-entry\nfeatures + alpha label\nentry + exit + dip + tranche labels"]

    DATA --> PIT --> TB
    PIT --> AB
    AB --> OUT
    TB --> OUT
    OUT --> REG --> RS
    OUT --> MO
    OUT --> BAH
```

---

## SourceWithFallback — Graceful Degradation

```mermaid
flowchart TD
    CALL["source.get(symbol)"]
    TRY["Try PRIMARY\nPlaywright / API"]
    OK["Return data\nsource_name, fallback=False"]
    ERR["Primary failed"]
    FB1["Try fallback 1"]
    FB1OK["Return data\nfallback=True\nTelegram WARNING (1h cooldown)"]
    FB1ERR["fallback 1 failed"]
    FB2["Try fallback 2"]
    FB2OK["Return data\nfallback=True"]
    ALLFAIL["ALL sources failed\nReturn None\nTelegram ERROR\nCaller applies neutral default (0.5)"]

    CALL --> TRY
    TRY -->|success| OK
    TRY -->|failure| ERR --> FB1
    FB1 -->|success| FB1OK
    FB1 -->|failure| FB1ERR --> FB2
    FB2 -->|success| FB2OK
    FB2 -->|failure| ALLFAIL
```

---

## Component Map (v13)

| Component | File | Role |
|-----------|------|------|
| Signal orchestrator | `strategy/signal_orchestrator.py` | Composite scoring · IC weighting · meta-stacking · Claude Sonnet |
| Exit monitor | `strategy/exit_monitor.py` | LSTM P(peak) binary classifier |
| Dip detector | `strategy/dip_detector.py` | AutoGluon P(recovery) + dip_verdict() |
| Tranche manager | `strategy/tranche_manager.py` | T1/T2/T3 + dip tranches |
| Consensus engine | `strategy/consensus.py` | 3-tier AI routing + verdict dispatch |
| Position manager | `strategy/position_manager.py` | Half-Kelly sizing · ATR stops · portfolio risk |
| Pattern detector | `data/sources/pattern_detector.py` | Algebraic · 12 patterns · no model file |
| Options flow | `data/sources/unusual_whales.py` | Playwright + Barchart fallback |
| FRED macro | `data/sources/fred_macro.py` | Yield curve · credit spread · VIX |
| Congressional | `data/fincept/congressional_trades.py` | STOCK Act · 6h cache |
| Research agent | `data/fincept/agents/research_agent.py` | Claude Haiku + Playwright browser |
| Telegram | `data/notifications/telegram.py` | Rate-limited · all alert types |
| Backtester | `backtester/intelligent_backtest.py` | vectorbt · alpha-beta · multi-regime |
| Training pipeline | `strategy/training_pipeline.py` | AutoGluon + mlfinlab CPCV |
| Feedback loop | `data/feedback_loop.py` | Alpha labels · regime-matched RAG |
| FastAPI | `server/api.py` | 40+ endpoints incl. analysis/earnings/financials/holdings |
| Terminal | `dashboard/terminal.html` | TradingView widget · 11 tabs · 2 position panels |
