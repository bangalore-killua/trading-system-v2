# Pipeline Flow

## Weekly Universe Rebuild (Sunday)

```
1. Load user seeds from config.json
2. In parallel:
   a. KG expansion: knowledge_graph.get_all_related(seed, hops=2)
   b. Theme baskets: fetch top-50 ETF holdings for all 20 ETFs
   c. Momentum: TradingView screener (>40% 3m return, >$5, >200k vol)
   d. User seeds: mandatory + manual_additions from config
3. Deduplicate → cap at 600 candidates
4. Compute 5 factor scores per candidate (yfinance, async parallel)
5. Run LightGBM P(win) per candidate (batch inference)
6. rank_candidates() → top 30 by P(win) = new watchlist
7. Update hot_universe TTL entries for all watchlist stocks
```

## Daily Signal Cycle (Market open)

```
1. For each watchlist stock:
   a. Fetch OHLCV + fundamentals (yfinance)
   b. Compute 130 existing features (technical_analysis.py)
   c. Add 5 factor scores (screener.compute_factor_scores())
   d. LightGBM.predict() → P(win)
   e. ConsensusEngine.evaluate():
      i.  Determine LGBM verdict (bullish / neutral / bearish)
      ii. Call Groq: "Is this trade bullish/neutral/bearish?"
      iii. Apply decision table → STRONG_BUY / BUY / CONFLICT / SKIP
      iv. Log CONFLICT to logs/consensus_conflicts.jsonl
   f. Write signal to database
2. Signal basket allocates positions (position_manager.py)
3. Alpaca order execution (if regime allows entry)
```

## Continuous News Pipeline

```
Always-on:
  news_aggregator.py → fetches 39 RSS feeds every 5 min
  edgar_monitor.py → polls SEC 8-K + Form 4 every 60s
  news_stream.py → Alpaca WebSocket (all symbols, real-time)

Per article:
  FinBERT → sentiment score [-1, 1]
  news_relevance_engine.py → extract tickers via spaCy + $TICKER regex
  hot_universe.py → if ticker not on watchlist + score passes → add with catalyst TTL
    (2 weeks for earnings/FDA, 6 weeks for partnerships, 3 months for M&A)

On-demand (API call):
  news_intelligence.py → sentence-transformers cluster + Groq portfolio analysis
```

## Daily Feedback Loop (4:30 PM)

```
1. Fetch top-20 movers (TradingView → yfinance fallback)
2. Load today's watchlist + positions + signals
3. Classify each mover (TP / FP / TN / missed / neutral)
4. Groq per mover: catalyst? repeatable? lesson?
5. Persist to logs/feedback_loop.jsonl + ChromaDB
```

## Monthly Model Update (1st Sunday)

```
1. Load all feedback_loop.jsonl records
2. Filter 1: catalyst_type frequency >= 3
3. Filter 2: outlier removal (> 3 sigma moves excluded)
4. Filter 3: regime match (current HMM regime)
5. Filter 4: Groq fundamental confirmation (batch, one call)
6. Retrain LightGBM on validated samples
7. Save scorecard_v{N+1}.pkl, update scorecard_meta.json
8. If new wf_accuracy < previous: auto-rollback to previous version
9. Write human-readable report to logs/model_update_reports/
```

## Position Lifecycle

```
Entry:
  STRONG_BUY → full base_position_size × 1.0
  BUY         → base_position_size × 0.75
  CONFLICT    → base_position_size × 0.25 (risk-reduced)
  SKIP        → no trade

Exit (any of):
  - Profit barrier hit (triple-barrier upper)
  - Loss barrier hit (triple-barrier lower)
  - Time barrier hit (holding period expires)
  - Regime changes to CRASH / EUPHORIA
  - Kill switch activated
```
