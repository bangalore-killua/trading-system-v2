# LightGBM Scorecard

## Overview

The scorecard is the core ML model. It takes 135 features per stock and outputs P(win) — the probability that a trade on this stock will reach the profit barrier before the loss barrier within the holding window.

## Training

- **Labels**: Triple-barrier (TB) labeling — +1 (hit profit barrier), -1 (hit loss barrier), 0 (timeout)
- **Class imbalance**: bearish ~5500 samples >> bullish ~2300 samples → `class_weight='balanced'`
- **Validation**: Walk-forward cross-validation (5 folds, 252-day training window)
- **Current accuracy**: Train 94%, walk-forward ~57% (overfitting reduced with `min_child_samples=50`)

## Features (135 total)

### Original 130 features (from signal basket)
- Price momentum (various periods)
- Volume ratios and anomalies
- RSI, MACD, Bollinger Band position
- ADX (trend strength)
- FinBERT sentiment (7-day aggregate)
- News volume
- Regime encoding (HMM state)
- Threat level encoding
- Options IV (when available)
- Sector/market relative returns

### 5 Factor Score features (Phase 2)
Added by `data/screener.py`, computed per stock at scoring time:
- `momentum_score` — 12-1 month return percentile vs sector peers
- `quality_score` — ROE percentile vs sector
- `technical_score` — RSI percentile vs stock's own 1-year history
- `valuation_score` — P/E sector percentile (lower P/E = higher score)
- `news_score` — FinBERT aggregate last 7 days, mapped to 0–100

All scores use `data/percentile_utils.py` — no fixed thresholds, always relative.

## Inference

```python
from strategy.meta_model import LightGBMMetaModel
model = LightGBMMetaModel(model_path="models/scorecard_meta.json")
model.load()
p_win = model.predict(feature_vector)  # returns P(win) in [0, 1]
```

## Consensus thresholds

| P(win) | LightGBM verdict | Default consensus |
|---|---|---|
| >= 0.65 | bullish | STRONG_BUY (with Groq bullish) |
| 0.52 – 0.65 | neutral | BUY (with Groq neutral) |
| < 0.52 | bearish | SKIP |

## Versioning

Active version stored in `models/scorecard_meta.json`. See `strategy/meta_model.py` for versioning logic.
Auto-rollback if new walk-forward accuracy < active version's accuracy.
