# Universe Screener

## Overview

`data/screener.py` builds the candidate universe using 4 parallel discovery methods, then scores each candidate with 5 percentile-based factor scores that become LightGBM input features.

The screener does NOT eliminate stocks — it scores them. LightGBM learns which combinations of scores lead to wins.

## Universe Building (4 methods, run in parallel)

### Method 1: Knowledge Graph Expansion
- Takes user seeds → expands 1-2 hops through the supply-chain graph
- Discovers APLD/ALAB because NVIDIA mentions them in its 10-K
- Source: `data/knowledge_graph.py` (NetworkX, SEC 10-K regex extraction)

### Method 2: Theme Baskets
- 10 pre-defined baskets (AI infrastructure, cloud, crypto, financials, energy, consumer, healthcare, defence, staples, metals)
- Each basket: seed tickers + ETF tickers → pulls top-50 ETF holdings via finvizfinance
- Ensures cross-sector diversification

### Method 3: Momentum Expansion
- Any US stock with >40% 3-month return, >$5 price, >200k avg volume
- Source: TradingView screener (falls back to yfinance if unavailable)
- Catches breakout names before mainstream coverage

### Method 4: User Seeds
- Manually specified in `config.json` under `watchlist.mandatory` and `watchlist.manual_additions`

## Factor Scoring (5 scores → LightGBM features)

All scores are 0–100 percentile rankings. 50 = neutral/missing data.

| Score | What it measures | Benchmark |
|---|---|---|
| `momentum_score` | 12-1 month price return | vs same theme basket peers |
| `quality_score` | Return on equity (ROE) | vs same theme basket peers |
| `technical_score` | RSI | vs stock's own 1-year RSI history |
| `valuation_score` | Trailing P/E (lower = better) | vs same theme basket peers |
| `news_score` | FinBERT aggregate last 7 days | linear mapping [-1,1] → [0,100] |

## Watchlist Selection

After scoring, `rank_candidates()` sorts by:
1. LightGBM P(win) if scorecard is available
2. Factor composite score (equal-weight average) if no scorecard yet

Top 30 candidates (configurable in config.json `watchlist.max_size`) become the weekly watchlist.

## Configuration

```json
{
  "screener": {
    "max_universe_size": 600
  },
  "watchlist": {
    "max_size": 30,
    "mandatory": ["SPY"],
    "manual_additions": []
  }
}
```
