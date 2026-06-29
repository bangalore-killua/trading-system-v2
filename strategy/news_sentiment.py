"""
news_sentiment.py — FinBERT Sentiment Scorer

Wraps `yiyanghkust/finbert-tone` (HuggingFace) for financial sentiment analysis.

Why finbert-tone over ProsusAI/finbert:
  - Same label scheme (positive/neutral/negative)
  - Trained on more recent financial text
  - Works with current `transformers` API without deprecated kwargs

Output: a float in [-1.0, +1.0]
  +1.0 = strongly positive (Positive P=1.0)
  -1.0 = strongly negative (Negative P=1.0)
   0.0 = neutral

The model is loaded once and cached. First call triggers the ~400MB download
from HuggingFace hub (or loads from local cache on subsequent runs).
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "yiyanghkust/finbert-tone"


class FinBERTSentiment:
    """Thread-safe FinBERT wrapper with lazy model loading.

    Call `score(text)` to get a sentiment float. The model is loaded on the
    first call and reused for all subsequent calls (no per-call load overhead).

    For batch scoring efficiency, use `score_batch(texts)` which feeds the
    full list to the HuggingFace pipeline in one forward pass.
    """

    def __init__(self, device: str = "cpu", max_length: int = 512):
        self._device = device
        self._max_length = max_length
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        """Lazy load — only called on first score request."""
        if self._pipeline is not None:
            return

        logger.info("Loading FinBERT model (%s) — this may take a moment...", MODEL_NAME)
        from transformers import pipeline  # type: ignore

        self._pipeline = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=self._device,
            # Return all label scores so we can compute a weighted average
            top_k=None,
        )
        logger.info("FinBERT loaded on device=%s", self._device)

    def score(self, text: str) -> float:
        """Score a single text string. Returns float in [-1.0, +1.0]."""
        if not text or not text.strip():
            return 0.0

        self._ensure_loaded()

        # Truncate to avoid tokenizer overflow warnings
        text = text[: self._max_length * 5]

        try:
            results = self._pipeline(text, truncation=True, max_length=self._max_length)
            return self._results_to_score(results)
        except Exception:
            logger.exception("FinBERT score error for: %s...", text[:60])
            return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a list of texts in one batched forward pass (more GPU-efficient)."""
        if not texts:
            return []

        self._ensure_loaded()

        # Filter empties and remember positions so we can restore them
        non_empty = [(i, t[:self._max_length * 5]) for i, t in enumerate(texts) if t and t.strip()]

        if not non_empty:
            return [0.0] * len(texts)

        indices, clean_texts = zip(*non_empty)

        try:
            batch_results = self._pipeline(
                list(clean_texts),
                truncation=True,
                max_length=self._max_length,
                batch_size=16,
            )
        except Exception:
            logger.exception("FinBERT batch score error")
            return [0.0] * len(texts)

        scores = [0.0] * len(texts)
        for idx, result in zip(indices, batch_results):
            scores[idx] = self._results_to_score(result)

        return scores

    @staticmethod
    def _results_to_score(results) -> float:
        """Convert pipeline output (list of {label, score} dicts) to [-1, +1] float.

        finbert-tone labels: 'Positive', 'Negative', 'Neutral'
        Formula: weighted_sum = P(Positive) - P(Negative)
        This preserves the magnitude of confidence, not just the sign.
        """
        if not results:
            return 0.0

        # pipeline with top_k=None returns a list of dicts directly
        label_scores = {}
        for item in results:
            label_scores[item["label"].lower()] = item["score"]

        positive = label_scores.get("positive", 0.0)
        negative = label_scores.get("negative", 0.0)

        return round(positive - negative, 4)

    def is_bullish(self, text: str, threshold: float = 0.70) -> bool:
        return self.score(text) >= threshold

    def is_bearish(self, text: str, threshold: float = -0.50) -> bool:
        return self.score(text) <= threshold


class MiniCPMSentiment:
    """Financial sentiment via local MiniCPM-1B through Ollama (no API cost).

    Scores text in [-1.0, +1.0] using a structured JSON prompt.
    Activate in config.json: news.sentiment_model = "minicpm"

    Install:
      winget install Ollama.Ollama
      ollama pull minicpm          # MiniCPM-1B (smallest, fastest)

    Fine-tune for better accuracy:
      python scripts/finetune_minicpm_sentiment.py
      ollama create minicpm-finetune -f models/minicpm_sentiment_lora/Modelfile
      config: news.sentiment_model = "minicpm-finetune"
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"

    def __init__(self, model: str = "minicpm", timeout: float = 10.0):
        self._model   = model
        self._timeout = timeout
        self._warned  = False

    def score(self, text: str) -> float:
        """Score a single text. Returns float in [-1.0, +1.0], 0.0 on failure."""
        if not text or not text.strip():
            return 0.0
        prompt = (
            "You are a financial news sentiment analyser. "
            'Reply with JSON only: {"sentiment": "positive|neutral|negative", "score": <float -1 to 1>}\n\n'
            f"Text: {text[:800]}"
        )
        try:
            import json as _json
            import httpx
            resp = httpx.post(
                self.OLLAMA_URL,
                json={"model": self._model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.0, "num_predict": 60}},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw    = resp.json().get("response", "").strip()
            raw    = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = _json.loads(raw)
            val    = float(parsed.get("score", 0.0))
            return max(-1.0, min(1.0, val))
        except Exception as e:
            if not self._warned:
                logger.warning("MiniCPM sentiment unavailable (%s) -- returning 0.0", e)
                self._warned = True
            return 0.0

    def score_batch(self, texts: list) -> list:
        """Score a list of texts via per-item calls (Ollama local, no rate limit)."""
        return [self.score(t) for t in texts]

    def is_bullish(self, text: str, threshold: float = 0.40) -> bool:
        return self.score(text) >= threshold

    def is_bearish(self, text: str, threshold: float = -0.30) -> bool:
        return self.score(text) <= threshold


def get_sentiment_model(config: dict):
    """Factory: return FinBERTSentiment or MiniCPMSentiment based on config.

    config key: news.sentiment_model = "finbert" (default) | "minicpm" | "minicpm-finetune"
    """
    model_name = config.get("news", {}).get("sentiment_model", "finbert")
    if model_name.startswith("minicpm"):
        return MiniCPMSentiment(model=model_name)
    return FinBERTSentiment()
