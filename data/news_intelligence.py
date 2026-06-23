"""
news_intelligence.py — Multi-Layer News Analysis Pipeline

Stacks three capabilities on top of the existing FinBERT sentiment scores:

  1. sentence-transformers (all-MiniLM-L6-v2, 80MB, CPU-fast)
     Groups articles by semantic similarity → correlated clusters.
     No API call needed — runs locally.

  2. NewsRelevanceEngine (data/news_relevance_engine.py)
     Maps company mentions to tickers using spaCy NER + $TICKER regex
     + pre-built company→ticker lookup. Already implemented.

  3. Groq (Llama 3.3 70B, free tier)
     Portfolio-aware analysis: given correlated article clusters and the
     current portfolio holdings, explains impact per holding + cross-sector
     correlations. Returns a structured JSON with impact, correlations,
     and a 3-bullet market snapshot.

Usage:
  engine = NewsIntelligenceEngine()
  result = await engine.analyse(articles, portfolio_holdings)
  # result: {clusters, impact, cross_correlations, summary}

The /api/news/analysis endpoint calls this module.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# sentence-transformers model — 80MB download on first use, CPU-fast after
ST_MODEL_NAME = "all-MiniLM-L6-v2"

# Similarity threshold for grouping articles into clusters (cosine similarity)
CLUSTER_THRESHOLD = 0.65

# Groq model for portfolio analysis
GROQ_MODEL = "llama-3.3-70b-versatile"

# Max articles passed to Groq to stay within context window
MAX_ARTICLES_FOR_GROQ = 15


class NewsIntelligenceEngine:
    """Multi-layer news analysis: cluster → link tickers → Groq portfolio analysis.

    All three layers are optional — the engine degrades gracefully when
    sentence-transformers is not installed or Groq key is missing.
    """

    def __init__(self):
        self._st_model = None        # lazy-loaded sentence-transformers model
        self._st_warned = False
        self._groq_client = None     # lazy-loaded Groq client
        self._groq_warned = False

        # NewsRelevanceEngine for ticker extraction (already implemented)
        from data.news_relevance_engine import NewsRelevanceEngine
        self._relevance = NewsRelevanceEngine()

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyse(
        self,
        articles: List[dict],
        portfolio_holdings: Optional[List[str]] = None,
    ) -> dict:
        """Full pipeline: cluster → link tickers → Groq portfolio analysis.

        Parameters
        ----------
        articles : list of dicts, each with at minimum:
            {"title": str, "summary": str, "finbert_score": float, "url": str}
        portfolio_holdings : list of ticker strings currently held

        Returns
        -------
        dict with keys:
            clusters        — list of article groups (each group = related articles)
            ticker_map      — {article_index: [tickers mentioned]}
            impact          — [{ticker, sentiment, reason, action_hint}]  (Groq)
            cross_correlations — [str]  (Groq)
            summary         — str, 3-bullet market snapshot  (Groq)
        """
        if not articles:
            return self._empty_result()

        # Step 1: Cluster articles by semantic similarity
        clusters = self._cluster_articles(articles)

        # Step 2: Link each article to tickers
        ticker_map = self._extract_tickers_from_articles(articles)

        # Step 3: Groq portfolio-aware analysis (async, may be skipped if no API key)
        groq_result = await self._groq_portfolio_analysis(
            articles=articles[:MAX_ARTICLES_FOR_GROQ],
            portfolio_holdings=portfolio_holdings or [],
            clusters=clusters,
        )

        return {
            "clusters": clusters,
            "ticker_map": ticker_map,
            "impact": groq_result.get("impact", []),
            "cross_correlations": groq_result.get("cross_correlations", []),
            "summary": groq_result.get("summary", ""),
        }

    def cluster_articles(self, articles: List[dict]) -> List[List[int]]:
        """Public convenience: return cluster groups (list of article-index lists)."""
        return self._cluster_articles(articles)

    def extract_tickers(self, articles: List[dict]) -> Dict[int, List[str]]:
        """Public convenience: return {article_index: [tickers]} mapping."""
        return self._extract_tickers_from_articles(articles)

    # ── Step 1: Semantic Clustering ───────────────────────────────────────────

    def _cluster_articles(self, articles: List[dict]) -> List[List[int]]:
        """Group articles by cosine similarity of their sentence embeddings.

        Returns a list of clusters; each cluster is a list of article indices.
        Articles without any match form singleton clusters.
        """
        if len(articles) < 2:
            return [[i] for i in range(len(articles))]

        model = self._get_st_model()
        if model is None:
            # Fallback: each article is its own cluster
            return [[i] for i in range(len(articles))]

        texts = [
            (a.get("title", "") + " " + a.get("summary", ""))[:512]
            for a in articles
        ]
        try:
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
            # Cosine similarity matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normed = embeddings / norms
            sim_matrix = normed @ normed.T  # (N, N) cosine similarities

            # Greedy clustering: assign each article to the first cluster where
            # it has similarity ≥ CLUSTER_THRESHOLD with the cluster centroid
            assigned = [-1] * len(articles)
            clusters: List[List[int]] = []

            for i in range(len(articles)):
                best_cluster = -1
                best_sim = CLUSTER_THRESHOLD - 0.01

                for c_idx, cluster in enumerate(clusters):
                    # Compare to mean embedding of existing cluster members
                    cluster_emb = embeddings[cluster].mean(axis=0)
                    cluster_emb_norm = cluster_emb / (np.linalg.norm(cluster_emb) + 1e-10)
                    sim = float(normed[i] @ cluster_emb_norm)
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = c_idx

                if best_cluster >= 0:
                    clusters[best_cluster].append(i)
                    assigned[i] = best_cluster
                else:
                    assigned[i] = len(clusters)
                    clusters.append([i])

            logger.debug(
                "Clustered %d articles into %d groups (threshold=%.2f)",
                len(articles), len(clusters), CLUSTER_THRESHOLD,
            )
            return clusters

        except Exception:
            logger.exception("Article clustering failed — returning singletons")
            return [[i] for i in range(len(articles))]

    def _get_st_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._st_model is not None:
            return self._st_model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(ST_MODEL_NAME)
            logger.info("sentence-transformers model loaded: %s", ST_MODEL_NAME)
            return self._st_model
        except ImportError:
            if not self._st_warned:
                logger.warning(
                    "sentence-transformers not installed — article clustering disabled. "
                    "Run: pip install sentence-transformers>=2.7.0"
                )
                self._st_warned = True
            return None
        except Exception:
            logger.exception("Failed to load sentence-transformers model")
            return None

    # ── Step 2: Ticker Extraction ─────────────────────────────────────────────

    def _extract_tickers_from_articles(self, articles: List[dict]) -> Dict[int, List[str]]:
        """Map each article index to a list of mentioned US equity tickers."""
        ticker_map: Dict[int, List[str]] = {}
        for i, article in enumerate(articles):
            text = article.get("title", "") + " " + article.get("summary", "")
            tickers = self._relevance.extract_tickers(text)
            if tickers:
                ticker_map[i] = tickers
        return ticker_map

    # ── Step 3: Groq Portfolio Analysis ──────────────────────────────────────

    async def _groq_portfolio_analysis(
        self,
        articles: List[dict],
        portfolio_holdings: List[str],
        clusters: List[List[int]],
    ) -> dict:
        """Call Groq to analyse news impact on portfolio holdings.

        Returns dict with 'impact', 'cross_correlations', 'summary' keys.
        Returns empty dict if Groq is unavailable.
        """
        client = self._get_groq_client()
        if client is None:
            return {}

        # Build a compact article summary for the prompt
        article_lines = []
        for i, a in enumerate(articles):
            score = a.get("finbert_score", 0.0)
            sentiment = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
            article_lines.append(
                f"[{i}] ({sentiment} {score:+.2f}) {a.get('title', '')[:120]}"
            )

        # Cluster summary
        cluster_summary = []
        for c_idx, cluster in enumerate(clusters):
            if len(cluster) > 1:
                titles = [articles[i].get("title", "")[:60] for i in cluster[:3]]
                cluster_summary.append(f"Cluster {c_idx+1}: {', '.join(titles)}")

        prompt = f"""You are a senior portfolio manager analysing market news.

Portfolio holdings: {', '.join(portfolio_holdings) if portfolio_holdings else 'None provided'}

Recent news articles (with FinBERT sentiment):
{chr(10).join(article_lines)}

{f'Related article clusters:{chr(10)}{chr(10).join(cluster_summary)}' if cluster_summary else ''}

Analyse the news and return a JSON object with exactly these keys:
{{
  "impact": [
    {{"ticker": "AAPL", "sentiment": "bullish/bearish/neutral", "reason": "one sentence", "action_hint": "HOLD/WATCH/REVIEW"}}
  ],
  "cross_correlations": [
    "Energy news + airline sector = cost pressure risk",
    "AI chip demand surge = positive for NVDA ecosystem"
  ],
  "summary": "• Bullet 1\\n• Bullet 2\\n• Bullet 3"
}}

Only include tickers from the portfolio holdings in 'impact'.
Return valid JSON only, no markdown fences."""

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                ),
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.warning("Groq returned non-JSON news analysis response")
            return {}
        except Exception:
            logger.exception("Groq news analysis failed")
            return {}

    def _get_groq_client(self):
        """Lazy-load the Groq client."""
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            if not self._groq_warned:
                logger.info(
                    "GROQ_API_KEY not set — portfolio-aware news analysis disabled. "
                    "Add GROQ_API_KEY to .env to enable Groq features."
                )
                self._groq_warned = True
            return None
        try:
            from groq import Groq  # type: ignore
            self._groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized (model: %s)", GROQ_MODEL)
            return self._groq_client
        except ImportError:
            if not self._groq_warned:
                logger.warning("groq package not installed — run: pip install groq>=0.4.0")
                self._groq_warned = True
            return None
        except Exception:
            logger.exception("Failed to initialize Groq client")
            return None

    def _empty_result(self) -> dict:
        return {
            "clusters": [],
            "ticker_map": {},
            "impact": [],
            "cross_correlations": [],
            "summary": "",
        }


# Module-level singleton
_engine: Optional[NewsIntelligenceEngine] = None


def get_news_intelligence_engine() -> NewsIntelligenceEngine:
    global _engine
    if _engine is None:
        _engine = NewsIntelligenceEngine()
    return _engine
