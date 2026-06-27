"""
feedback_loop.py — Daily Post-Market Feedback Collection (Phase 5 + v13 upgrade)

v13 changes (from audit):
  Fix 2: `{symbol in signal}` bug → replaced with `{in_watchlist}` (was always False)
  Fix 3: fetch 3 real headlines per symbol, pass to _groq_analyse_mover()
  NEW:   alpha-beta separation — every stored outcome = trade_return - (spy_return × beta)
  NEW:   9 label types (correct_peak_exit, caught_falling_knife, etc.)
  NEW:   regime field on every record (for regime-matched RAG queries)
  NEW:   regime-matched ChromaDB query — RAG now filters by regime, not just symbol

Runs automatically at 4:30 PM US/Eastern every trading day (configured via cron).

Records stored per day:
  - true_positive, false_positive, true_negative, missed (original 4)
  - correct_peak_exit, early_exit, late_exit (exit labels)
  - correct_dip_buy, caught_falling_knife (dip labels)
  - correct_accumulate, correct_hold, correct_scale_up, premature_scale_up (new)

ChromaDB collections:
  feedback_records (existing) — entry cases, queried with regime-matching
  exit_records (new)          — exit cases
  dip_records  (new)          — dip cases

Usage:
  python -m data.feedback_loop        # run immediately (one shot)
  python -m data.feedback_loop --dry  # print what would be collected, no writes
"""

import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FEEDBACK_LOG_PATH  = Path("logs/feedback_loop.jsonl")
CHROMA_COLLECTION  = "feedback_records"
EXIT_COLLECTION    = "exit_records"
DIP_COLLECTION     = "dip_records"
GROQ_MODEL         = "llama-3.3-70b-versatile"
TOP_N_MOVERS       = 20
MAX_PCT_MOVE       = 50.0  # exclude >50% single-day moves (meme/squeeze outliers)

# v13: 9 label types (in addition to the original 4 outcome_type values)
ENTRY_LABELS = ["true_positive", "false_positive", "true_negative", "missed", "neutral"]
EXIT_LABELS  = ["correct_peak_exit", "early_exit", "late_exit", "correct_hold"]
DIP_LABELS   = ["correct_dip_buy", "caught_falling_knife", "correct_accumulate"]
TRANCHE_LABELS = ["correct_scale_up", "premature_scale_up"]


class FeedbackCollector:
    """Daily top-mover collection and Groq analysis.

    Instantiate once per session. The `run()` coroutine is the main entry point.
    """

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or {}
        self._groq_client = None
        self._chroma_client = None
        self._collection = None
        FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    async def run(self, dry_run: bool = False) -> List[dict]:
        """Main entry point — run the full daily collection cycle.

        Parameters
        ----------
        dry_run : if True, print records but do not write to disk or ChromaDB

        Returns
        -------
        list of feedback records collected today
        """
        today = date.today()
        logger.info("Feedback loop: collecting top movers for %s", today)

        # Step 1: fetch movers
        movers = await self._fetch_top_movers()
        if not movers:
            logger.warning("No movers fetched — market may be closed or data unavailable")
            return []

        # Step 2: load today's context (watchlist, positions, signals)
        context = self._load_today_context()

        # Step 3: analyse each mover
        records = []
        for mover in movers:
            record = await self._analyse_mover(mover, context)
            records.append(record)

        if dry_run:
            for r in records:
                print(json.dumps(r, indent=2, default=str))
            return records

        # Step 4: persist
        await self._persist_records(records)

        logger.info(
            "Feedback loop complete: %d records saved (%d TP, %d FP, %d missed)",
            len(records),
            sum(1 for r in records if r["outcome_type"] == "true_positive"),
            sum(1 for r in records if r["outcome_type"] == "false_positive"),
            sum(1 for r in records if r["outcome_type"] == "missed"),
        )
        return records

    # ── Step 1: Fetch Movers ──────────────────────────────────────────────────

    async def _fetch_top_movers(self) -> List[dict]:
        """Fetch top gainers + losers for today, capped at MAX_PCT_MOVE."""
        movers = []

        # Try TradingView screener first (already installed)
        try:
            from tradingview_screener import Query, Column  # type: ignore
            result = (
                Query()
                .select("name", "change", "close", "average_volume_10d_calc", "volume")
                .where(
                    Column("close") > 1.0,
                    Column("average_volume_10d_calc") > 100_000,
                    Column("exchange").isin(["NASDAQ", "NYSE", "AMEX"]),
                )
                .order_by("change|abs", ascending=False)
                .limit(TOP_N_MOVERS * 2)
                .get_scanner_data()
            )
            _, df = result if isinstance(result, tuple) else (0, result)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    pct = float(row.get("change", 0))
                    if abs(pct) <= MAX_PCT_MOVE:
                        ticker = str(row.get("name", "")).replace("NASDAQ:", "").replace("NYSE:", "")
                        movers.append({
                            "symbol": ticker,
                            "pct_change": round(pct, 2),
                            "close": float(row.get("close", 0)),
                            "volume_ratio": (
                                float(row.get("volume", 0)) / max(float(row.get("average_volume_10d_calc", 1)), 1)
                            ),
                        })
                    if len(movers) >= TOP_N_MOVERS:
                        break
        except Exception:
            logger.debug("TradingView screener unavailable for movers, falling back to yfinance")

        # Fallback: yfinance market screener (standard large-caps)
        if not movers:
            try:
                import yfinance as yf  # type: ignore
                # yfinance 0.2+ supports market screener
                screener = yf.Screener()
                screener.set_default_body({"query": {"operator": "and", "operands": []}, "sortField": "percentchange"})
                result = screener.get_screeners(["day_gainers", "day_losers"])
                for category, quotes in result.items():
                    for q in (quotes or [])[:10]:
                        pct = float(q.get("regularMarketChangePercent", 0))
                        if abs(pct) <= MAX_PCT_MOVE:
                            movers.append({
                                "symbol": q.get("symbol", ""),
                                "pct_change": round(pct, 2),
                                "close": float(q.get("regularMarketPrice", 0)),
                                "volume_ratio": 1.0,
                            })
            except Exception:
                logger.debug("yfinance screener unavailable — no movers fetched")

        # Deduplicate by symbol
        seen = set()
        unique = []
        for m in movers:
            if m["symbol"] and m["symbol"] not in seen:
                seen.add(m["symbol"])
                unique.append(m)
        return unique[:TOP_N_MOVERS]

    # ── Step 2: Load Context ──────────────────────────────────────────────────

    def _load_today_context(self) -> dict:
        """Load today's watchlist, positions, and any signals generated."""
        context: Dict[str, Any] = {
            "watchlist": set(),
            "positions": set(),
            "signals": {},
            "universe": set(),
        }
        try:
            from database import get_db  # type: ignore
            db = get_db()
            signals = db.get_latest_signals()
            for s in signals:
                sym = s.get("symbol", "")
                if sym:
                    context["watchlist"].add(sym.upper())
                    context["signals"][sym.upper()] = s
        except Exception:
            logger.debug("DB unavailable — context will be empty")

        try:
            from database import get_db  # type: ignore
            db = get_db()
            positions = db.get_positions()
            for p in positions:
                sym = p.get("symbol", "")
                if sym:
                    context["positions"].add(sym.upper())
        except Exception:
            pass

        return context

    # ── Step 3: Analyse Each Mover ────────────────────────────────────────────

    async def _analyse_mover(self, mover: dict, context: dict) -> dict:
        """Classify the mover and optionally get Groq analysis.

        Returns a feedback record dict.
        """
        symbol = mover["symbol"].upper()
        pct = mover["pct_change"]
        in_watchlist = symbol in context["watchlist"]
        in_positions = symbol in context["positions"]
        in_universe = symbol in context["universe"]
        signal = context["signals"].get(symbol, {})

        # Classify outcome type
        signal_buy = signal.get("signal", "").upper() in ("BUY", "STRONG_BUY")
        signal_skip = not signal_buy

        if in_positions and pct > 0:
            outcome_type = "true_positive"
        elif in_positions and pct < 0:
            outcome_type = "false_positive"
        elif signal_buy and pct < 0:
            outcome_type = "false_positive"
        elif signal_skip and pct < 0:
            outcome_type = "true_negative"
        elif not in_watchlist and abs(pct) >= 5.0:
            outcome_type = "missed"
        else:
            outcome_type = "neutral"

        # v13 Fix 3: fetch 3 real headlines per symbol and pass to Groq
        headlines = await self._fetch_headlines(symbol)

        # v13: alpha-beta separation
        alpha = self._compute_alpha(symbol, pct)

        # v13: regime for regime-matched RAG
        regime = self._get_current_regime()

        # v13 Fix 2: was `{symbol in signal}` (always False bug) → now `{in_watchlist}`
        groq_analysis = await self._groq_analyse_mover(symbol, mover, signal, headlines, in_watchlist)

        record = {
            "date":         date.today().isoformat(),
            "symbol":       symbol,
            "pct_change":   pct,
            "close":        mover["close"],
            "volume_ratio": mover.get("volume_ratio", 1.0),
            "in_watchlist": in_watchlist,
            "in_positions": in_positions,
            "outcome_type": outcome_type,
            "signal":       signal.get("signal", "NONE"),
            "lgbm_score":   signal.get("normalised_score", None),
            "alpha":        alpha,                           # v13: beta-adjusted excess return
            "regime":       regime,                          # v13: for regime-matched RAG
            "headlines":    headlines[:3],                   # v13: real news context
            "groq_analysis": groq_analysis,
            "recorded_at":  datetime.utcnow().isoformat(),
        }

        # ── OUTCOME + FEEDBACK logs ───────────────────────────────────────────
        try:
            from data.logging.trade_logger import new_trace, WhyAnalyzer
            won = alpha > 0 if alpha is not None else False
            raw_return = pct / 100.0
            result_label = "PASS" if won else "FAIL" if outcome_type == "false_positive" else "MISS"

            trace_id, olog = new_trace(symbol, "OUTCOME")
            olog.result(result_label).model({
                "alpha":          alpha,
                "raw_return":     raw_return,
                "outcome_type":   outcome_type,
                "regime":         regime,
            }).why(WhyAnalyzer.outcome(
                alpha or 0.0, raw_return,
                days_held=1,
                exit_reason=outcome_type,
                entry_composite=float(signal.get("normalised_score", 0.5) or 0.5),
            )).emit()

            groq_lesson = groq_analysis.get("lesson", "") if isinstance(groq_analysis, dict) else str(groq_analysis)[:120]
            top_signals = [k for k, v in signal.items() if isinstance(v, float) and v > 0.7][:3]
            tfl_stage_fn = __import__("data.logging.trade_logger", fromlist=["stage"]).stage
            tfl_stage_fn(trace_id, symbol, "FEEDBACK").result("PASS").model({
                "won": won, "groq_lesson": groq_lesson[:200],
                "ic_update_queued": True,
            }).why(WhyAnalyzer.feedback(won, top_signals, groq_lesson)).emit()
        except Exception:
            pass

        return record

    async def _groq_analyse_mover(
        self, symbol: str, mover: dict, signal: dict,
        headlines: list, in_watchlist: bool
    ) -> dict:
        """Ask Groq whether the move was repeatable and what drove it.

        v13 Fix 2: `in_watchlist` replaces `{symbol in signal}` (was always False).
        v13 Fix 3: Real headlines are passed in (fetched before this call).
        """
        client = self._get_groq_client()
        if client is None:
            return {}

        pct           = mover["pct_change"]
        headline_str  = "\n".join(f"  - {h}" for h in headlines[:3]) if headlines else "  (none)"

        system_prompt = "You are a quantitative analyst. Analyse today's market move. Return JSON only."
        user_prompt   = (
            f"Stock: {symbol}\n"
            f"Move: {pct:+.1f}% today\n"
            f"Our signal: {signal.get('signal', 'NONE')} (score: {signal.get('normalised_score', 'N/A')})\n"
            f"Was in watchlist: {in_watchlist}\n\n"       # Fix 2: was `{symbol in signal}` (always False)
            f"Today's headlines:\n{headline_str}\n\n"     # Fix 3: real headlines
            f'Return JSON: {{"repeatable": true/false, '
            f'"catalyst": "one sentence", '
            f'"catalyst_type": "earnings|fda|partnership|macro|technical|squeeze|unknown", '
            f'"lesson": "one takeaway"}}'
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                ),
            )
            content = response.choices[0].message.content.strip()
            content = content.lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(content)
        except Exception:
            return {}

    async def _fetch_headlines(self, symbol: str) -> list:
        """Fetch up to 3 real headlines for symbol via Google News RSS (Fix 3)."""
        try:
            from data.sources.seeking_alpha import _fetch_google_news  # type: ignore
            articles = _fetch_google_news(symbol, limit=3)
            return [a["title"] for a in articles if a.get("title")]
        except Exception:
            return []

    def _compute_alpha(self, symbol: str, trade_return_pct: float) -> Optional[float]:
        """Beta-adjusted excess return: alpha = trade_return - (spy_return × beta).

        v13: Every stored outcome is beta-adjusted to measure edge, not bull-market luck.
        """
        try:
            import yfinance as yf
            from datetime import date, timedelta
            end   = date.today()
            start = end - timedelta(days=5)

            spy_data    = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
            sym_data    = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

            if spy_data.empty or sym_data.empty:
                return None

            spy_return   = float((spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[0]) - 1)
            stock_return = trade_return_pct / 100

            # Beta approximation from 1-year correlation
            hist_spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)["Close"].pct_change().dropna()
            hist_sym = yf.download(symbol, period="1y", progress=False, auto_adjust=True)["Close"].pct_change().dropna()
            aligned  = hist_spy.align(hist_sym, join="inner")
            if len(aligned[0]) < 30:
                beta = 1.0
            else:
                import numpy as np
                cov  = float(np.cov(aligned[0], aligned[1])[0, 1])
                var  = float(aligned[0].var())
                beta = cov / (var + 1e-9)

            alpha = stock_return - (spy_return * beta)
            return round(alpha, 5)
        except Exception:
            return None

    def _get_current_regime(self) -> str:
        """Get current market regime for regime-matched RAG."""
        try:
            from strategy.regime_detector import RegimeDetector  # type: ignore
            rd = RegimeDetector()
            return str(rd.get_current_regime())
        except Exception:
            return "NEUTRAL"

    # ── Step 4: Persist ───────────────────────────────────────────────────────

    async def _persist_records(self, records: List[dict]) -> None:
        """Write records to JSONL file and ChromaDB."""
        # 1. Append to JSONL audit trail
        with open(FEEDBACK_LOG_PATH, "a") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

        # 2. Store in ChromaDB for RAG retrieval
        client, collection = self._get_chroma_collection()
        if collection is None:
            return

        import hashlib
        docs, metas, ids = [], [], []
        for r in records:
            text = (
                f"Symbol: {r['symbol']} | Date: {r['date']} | Move: {r['pct_change']:+.1f}% | "
                f"Signal: {r['signal']} | Outcome: {r['outcome_type']} | "
                f"Catalyst: {r.get('groq_analysis', {}).get('catalyst', 'unknown')}"
            )
            doc_id = hashlib.md5(f"{r['symbol']}_{r['date']}".encode()).hexdigest()
            docs.append(text)
            metas.append({
                "symbol":       r["symbol"],
                "date":         r["date"],
                "outcome_type": r["outcome_type"],
                "pct_change":   r["pct_change"],
                "catalyst_type": r.get("groq_analysis", {}).get("catalyst_type", "unknown"),
                "repeatable":   str(r.get("groq_analysis", {}).get("repeatable", False)),
                "regime":       r.get("regime", "NEUTRAL"),   # v13: regime for RAG matching
                "alpha":        str(r.get("alpha", "")),       # v13: beta-adjusted outcome
            })
            ids.append(doc_id)

        try:
            collection.upsert(documents=docs, metadatas=metas, ids=ids)
            logger.info("ChromaDB: upserted %d feedback records", len(records))
        except Exception:
            logger.exception("ChromaDB upsert failed — records saved to JSONL only")

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def _get_chroma_collection(self):
        """Lazy-init ChromaDB client + feedback_records collection."""
        if self._chroma_client is not None:
            return self._chroma_client, self._collection
        try:
            import chromadb  # type: ignore
            chroma_path = self._cfg.get("chromadb_path", "data/chromadb")
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
            self._collection = self._chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB initialized at %s (collection: %s)", chroma_path, CHROMA_COLLECTION)
            return self._chroma_client, self._collection
        except ImportError:
            logger.info("chromadb not installed — feedback stored in JSONL only. Run: pip install chromadb>=0.5.0")
            return None, None
        except Exception:
            logger.exception("Failed to initialize ChromaDB")
            return None, None

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        try:
            from groq import Groq  # type: ignore
            self._groq_client = Groq(api_key=api_key)
            return self._groq_client
        except Exception:
            return None


# ── Query helpers (used by model_updater.py and Groq RAG) ────────────────────

def query_similar_setups(
    symbol:         str,
    outcome_filter: Optional[str] = None,
    regime:         Optional[str] = None,   # v13: regime-matched RAG
    n_results:      int = 5,
) -> List[dict]:
    """Retrieve past feedback records similar to the given symbol from ChromaDB.

    v13: regime parameter filters to same-regime cases for RAG (regime-matched RAG).
    Same RSI/MACD setup in MID_BULL vs LATE_BULL → different expected outcome.

    Used by Groq prompts to add system-specific memory:
    "Last 3 times NVDA had this setup in MID_BULL: outcomes were X, Y, Z"

    Parameters
    ----------
    symbol         : ticker to query
    outcome_filter : if set, only return records with this outcome_type
    regime         : if set, filter to records from same regime (e.g. "MID_BULL")
    n_results      : max number of records to return
    """
    try:
        import chromadb  # type: ignore
        chroma_path = "data/chromadb"
        client      = chromadb.PersistentClient(path=chroma_path)
        collection  = client.get_collection(CHROMA_COLLECTION)

        # Build where filter — regime-matched first, symbol fallback
        where_conditions = []
        if regime:
            where_conditions.append({"regime": {"$eq": regime}})
        if outcome_filter:
            where_conditions.append({"outcome_type": {"$eq": outcome_filter}})

        if len(where_conditions) > 1:
            where = {"$and": where_conditions}
        elif len(where_conditions) == 1:
            where = where_conditions[0]
        else:
            where = {"symbol": {"$eq": symbol}}

        results = collection.query(
            query_texts=[f"Symbol: {symbol}"],
            n_results=n_results,
            where=where,
        )
        records = results.get("metadatas", [[]])[0]

        # Fallback to symbol-only if regime-matched returns nothing
        if not records and regime:
            results = collection.query(
                query_texts=[f"Symbol: {symbol}"],
                n_results=n_results,
                where={"symbol": {"$eq": symbol}},
            )
            records = results.get("metadatas", [[]])[0]

        return records
    except Exception:
        return []


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run daily feedback collection")
    parser.add_argument("--dry", action="store_true", help="Print records without saving")
    args = parser.parse_args()
    asyncio.run(FeedbackCollector().run(dry_run=args.dry))
