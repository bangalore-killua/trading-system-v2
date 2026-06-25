# Pipeline Flow (v13)

## Weekly Universe Rebuild (Sunday)

```
1. Load seeds from config.json (mandatory + manual_additions)
2. In parallel:
   a. KG expansion:    knowledge_graph.get_all_related(seed, hops=2) — SEC 10-K supply-chain
   b. Theme baskets:   10 sector ETFs → top-50 holdings each via finvizfinance
   c. Momentum:        TradingView screener (>40% 3m return, >$5, >200k vol)
   d. Seeds:           mandatory + manual from config.json
3. Deduplicate → cap at 600 candidates
4. Compute 8 factor scores per candidate (all 0–1, async parallel):
   momentum, quality, technical, valuation, news, options, macro, insider
5. Run entry LightGBM P(win) per candidate (batch inference)
6. rank_candidates() → top 30 by P(win) = new watchlist
7. Telegram: weekly watchlist digest + any new entries / exits
```

## Daily Signal Cycle (market open)

```
1. For each watchlist stock:
   a. Fetch OHLCV from tvdatafeed (1D, 4H native — no resampling)
   b. Compute ~138 features:
      - pandas-ta: 130 indicators + 60 CDL patterns
      - 8 factor scores (screener.compute_factor_scores())
      - Algebraic patterns (pattern_detector.detect_all())
      - 4H EMA50>EMA200 gate feature
      - 1D regime slope feature
   c. Compute composite_score = Σ(score_i × IC[score_i][regime])
   d. Models run in parallel:
      i.  entry LightGBM.predict() → P(win)
      ii. exit_monitor.score(position) → P(peak) [only for open positions]
      iii. dip_detector.score(symbol) → P(recovery) [if dip event detected]
   e. SignalOrchestrator → candidate_action + confidence
   f. ConsensusEngine routes to Groq/Haiku/Sonnet by decision type
   g. Hard constraints check → if all pass: EXEC
2. position_size = half_kelly × composite × regime_scalar
3. Tranche logic:
   - No open position: T1 entry if composite ≥ 0.35
   - T1 open: T2 trigger if price > entry + 1.5×ATR + score still ≥ 0.35
   - T1+T2 open: T3 trigger if new high + score ≥ regime_threshold
   - Dip event: dip_score ≥ 0.65 → dip tranche (0.8–1.5× T1)
4. Alpaca order execution with fill verification
5. Telegram: trade alert for every execution
```

## Continuous News Pipeline (always-on)

```
PRNewswire + BusinessWire RSS  →  every 5 min  (credibility 1.0, primary source)
39 RSS feeds (news_aggregator)  →  every 5 min
Alpaca NewsDataStream           →  real-time WebSocket (all symbols)
SEC EDGAR RSS 8-K + Form 4      →  every 60s

Per article:
  FinBERT             → sentiment score [−1, +1]
  news_relevance_engine → SpaCy NER → extract tickers
  headline_velocity   → rate-of-change (more informative than count)
  threat_classifier   → CRITICAL / HIGH / MEDIUM / LOW

Per ticker event:
  if not on watchlist AND score passes → add to HOT_UNIVERSE with TTL
  if CRITICAL threat → hard exit if in position
  if material 8-K / Form 4 → HOT_UNIVERSE add · insider_score update
  if PRNewswire/BusinessWire source → prepend to /api/news (priority display)
```

## Alternative Data Cycle (hourly + on-demand)

```
Unusual Whales (options flow):
  Playwright → unusualwhales.com/stock/{symbol}/options-flow
  Extracts call/put ratio + unusual premium activity
  → options_score [0–1]
  Fallback: Barchart options data → 0.5 neutral if all fail

FRED macro signals (hourly cache):
  T10Y2Y (yield curve) + BAMLH0A0HYM2 (credit spread) + VIXCLS
  → macro_score [0–1] (0.5 = neutral; curve inverted → lower; spread wide → lower)
  → updated in regime_scalar at signal time

Congressional trades (6h cache):
  CapitolTrades aggregator → Senate EFTS fallback
  Recency weights: <30d=1.0, 30-60d=0.5, 60-90d=0.2
  Amount weight: min(amount_mid / 250,000, 3.0) × 0.2
  → insider_score [0.3, 0.8] (0.5 = neutral)
```

## Exit Monitoring (every 5 min during market hours)

```
For each open position:
  1. Hard exit check:  ATR stop / kill_switch / CRITICAL news?       → SELL_ALL immediately
  2. Regime check:     CRASH → exit all; EARLY_BEAR → exit non-INFLECTION
  3. LSTM exit score:  P(peak) → exit_score
      features: unrealized_pnl, peak_drawdown, days_held, entry_composite,
                finbert_delta_since_entry, analyst_downgrade_since_entry,
                insider_sell_since_entry_usd, regime_changed_since_entry
     exit_score ≥ 0.65  → E1 partial (sell 20–50%)  via Haiku exit_verdict()
     exit_score ≥ 0.75  → E2 partial (sell 30–80% of remaining)
     exit_score ≥ 0.88  → E3 full exit
  4. AGE-1 check:      held N days + gain > X% → sell 50%, Telegram confirm
  5. Tranche exits mirror entries: each tranche has independent monitoring
```

## Daily Feedback Loop (4:30 PM)

```
1. Fetch top-20 movers (TradingView screener → yfinance fallback)
2. For each mover:
   a. Classify outcome:
      true_positive, false_positive, true_negative, missed, neutral
      correct_peak_exit, early_exit, late_exit (v13 additions)
      correct_dip_buy, caught_falling_knife (v13 additions)
   b. Compute alpha = trade_return − (spy_return × stock_beta)
   c. Fetch 3 recent headlines for the symbol
   d. Groq: "Was this move repeatable? What catalyst drove it?"
3. Persist to:
   logs/feedback_loop.jsonl  (audit trail)
   ChromaDB feedback_records (entry cases)
   ChromaDB exit_records     (exit cases, v13)
   ChromaDB dip_records      (dip cases, v13)

RAG queries MUST include regime filter:
  chromadb.query(embedding=composite, where={"regime": {"$eq": current_regime}})
  Same RSI/MACD in MID_BULL vs LATE_BULL → different expected outcomes.
```

## Monthly Model Update (first Sunday of month)

```
1. Load feedback records from ChromaDB
2. Apply 4 noise filters (in order):
   a. Frequency:    catalyst_type must appear ≥ 3 times
   b. Outlier:      exclude moves > 3 sigma (meme/squeeze events)
   c. Regime:       only samples from same HMM regime as today
   d. Fundamental:  Groq confirms repeatable catalyst
3. AutoGluon retrains on filtered + alpha-labeled dataset
4. mlfinlab CPCV: purged folds, honest IC ~54% (not inflated ~57%)
5. Validate: IC must be > 0 (else rollback)
6. Save as models/scorecard_entry_v{N+1}.pkl
7. Auto-rollback if new IC < previous model IC
8. Telegram: "Entry model retrained — IC: 0.142 (+0.008). 847 new samples."
```

## API Endpoints (v13)

```
/api/ohlcv/{symbol}?interval=240      ← tvdatafeed (native 4H)
/api/analysis/{symbol}                ← analyst ratings + Haiku scenarios
/api/earnings/{symbol}                ← 8Q EPS history + filing links
/api/financials/{symbol}?period=q|a   ← income / balance / cash flow
/api/holdings/{symbol}                ← institutional + mutual + insiders + politicians
/api/news?ticker={sym}&hours=24       ← PRNewswire + BusinessWire + Google News merged
/api/ledger/open                      ← open positions with tranche state
/api/screener/scores/{symbol}         ← all 8 factor scores
/api/patterns/{symbol}                ← algebraic pattern detection results
```
