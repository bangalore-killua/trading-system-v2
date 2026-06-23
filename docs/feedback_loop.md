# Feedback Loop + Model Self-Improvement

## Overview

The feedback system has two parts:
1. **Daily collection** (`data/feedback_loop.py`) — runs at 4:30 PM every trading day
2. **Monthly retrain** (`data/model_updater.py`) — runs first Sunday of each month

## Daily Feedback Collection

### What it does

1. Fetches top-20 movers of the day (TradingView screener → yfinance fallback)
2. For each mover, classifies the outcome:
   - `true_positive`: we held it and it went up
   - `false_positive`: we signalled BUY but it went down
   - `true_negative`: we passed, it went down (correct skip)
   - `missed`: we missed a >5% mover not in our watchlist
   - `neutral`: mover within normal variance
3. Calls Groq per mover: "Was this move repeatable? What catalyst drove it?"
4. Persists to:
   - `logs/feedback_loop.jsonl` — human-readable audit trail
   - ChromaDB `feedback_records` collection — vector store for RAG

### Running manually

```bash
python -m data.feedback_loop         # run immediately
python -m data.feedback_loop --dry   # print records without saving
```

## Monthly Model Update

### 4-filter noise removal (applied in order)

All records must pass all 4 filters before use in retraining:

| Filter | Rule | Reason |
|---|---|---|
| 1. Frequency | catalyst_type must appear >= 3 times | Removes one-off anomalies |
| 2. Outlier | Exclude moves > 3 sigma | Removes squeezes and meme events |
| 3. Regime | Only samples from same HMM regime | Prevents regime-mismatch contamination |
| 4. Fundamental | Groq confirms repeatable catalyst | Removes noise-driven moves |

### After filtering

- Retrains LightGBM scorecard (via `TrainingPipeline`)
- New model saved as `scorecard_v{N+1}.pkl` via versioning system
- Auto-rollback if new walk-forward accuracy < previous
- Human-readable JSON report saved to `logs/model_update_reports/`

### Running manually

```bash
python -m data.model_updater              # full update
python -m data.model_updater --dry        # print plan without retraining
python -m data.model_updater --force      # skip Groq fundamental check (faster)
```

## RAG for Groq

ChromaDB stores feedback records as text embeddings. When the consensus engine calls Groq, it can retrieve similar past cases:

```python
from data.feedback_loop import query_similar_setups
past_cases = query_similar_setups("NVDA", outcome_filter="true_positive", n_results=5)
# → 5 most similar past NVDA setups that ended as true positives
```

This gives Groq system-specific memory without any fine-tuning.

## Invariants

- Prompt updates are **manual only** — never automatic
- The human-readable report is always reviewed before any deployment
- False positives are tracked, not just winners (avoids survivorship bias)
- Conflict signals are separately logged in `logs/consensus_conflicts.jsonl`
