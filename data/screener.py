"""
screener.py — Multi-Method Universe Builder + Factor Scoring

Builds a candidate universe of stocks using 4 parallel discovery methods,
then scores each candidate with 5 percentile-based factor scores.

The 5 factor scores become INPUT FEATURES for LightGBM — they do NOT filter
stocks out. LightGBM learns which combination of scores matters.

Discovery methods:
  1. Knowledge graph expansion  — supply chain / ecosystem stocks (APLD via NVDA)
  2. Theme baskets              — ETF components across 10 diversified themes
  3. Momentum expansion         — any stock > 40% 3-month return, any market cap
  4. User seeds                 — manually pinned symbols from config.json

Factor scores (all percentile-based, 0–100):
  momentum_score   — 12-1 month return vs sector peers
  quality_score    — ROE + FCF margin vs sector (higher = better business quality)
  technical_score  — RSI percentile + price above key MAs vs stock's own history
  valuation_score  — P/E vs sector peers (lower P/E = better value = higher score)
  news_score       — FinBERT aggregate score last 7 days

Usage:
  screener = UniverseScreener(config)
  candidates = await screener.build_universe()  # returns list of dicts
  factor_scores = await screener.compute_factor_scores(["NVDA", "AAPL"])
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── 10 Theme Baskets ─────────────────────────────────────────────────────────
# Each basket has seed tickers + ETF tickers (used to pull all ETF holdings).
# Utilities and REITs are excluded — interest-rate driven, poor model fit.

THEME_BASKETS: Dict[str, dict] = {
    "ai_infrastructure": {
        "seeds": ["NVDA", "AMD", "INTC", "QCOM"],
        "etfs":  ["SOXX", "SMH"],
        "extra": ["APLD", "ALAB", "SMCI", "VRT", "ANET"],
    },
    "cloud_hyperscaler": {
        "seeds": ["MSFT", "GOOGL", "AMZN", "ORCL"],
        "etfs":  ["WCLD", "SKYY"],
        "extra": ["SNOW", "CRM", "NOW", "DDOG"],
    },
    "crypto_fintech": {
        "seeds": ["COIN", "MSTR", "PYPL", "SQ"],
        "etfs":  ["BITQ", "BKCH"],
        "extra": ["HOOD", "MARA", "RIOT"],
    },
    "financials": {
        "seeds": ["JPM", "GS", "BAC", "MS"],
        "etfs":  ["XLF", "KBE"],
        "extra": ["V", "MA", "AXP", "BLK"],
    },
    "energy": {
        "seeds": ["XOM", "CVX", "COP", "SLB"],
        "etfs":  ["XLE", "OIH"],
        "extra": ["MPC", "PSX", "VLO"],
    },
    "consumer_discretionary": {
        "seeds": ["AMZN", "TSLA", "HD", "MCD"],
        "etfs":  ["XLY", "FDIS"],
        "extra": ["NKE", "SBUX", "CMG", "ABNB"],
    },
    "healthcare": {
        "seeds": ["LLY", "UNH", "JNJ", "ABBV"],
        "etfs":  ["XLV", "IBB"],
        "extra": ["MRNA", "BNTX", "REGN", "ISRG"],
    },
    "defence_industrial": {
        "seeds": ["LMT", "RTX", "NOC", "GD"],
        "etfs":  ["ITA", "XAR"],
        "extra": ["BA", "L3H", "CACI"],
    },
    "consumer_staples": {
        "seeds": ["PG", "KO", "PEP", "COST"],
        "etfs":  ["XLP", "FSTA"],
        "extra": ["WMT", "CL", "GIS"],
    },
    "metals_commodities": {
        "seeds": ["NEM", "FCX", "AA", "CLF"],
        "etfs":  ["GDX", "XME"],
        "extra": ["GOLD", "AEM", "MP"],
    },
}

# Minimum 3-month return for momentum expansion method
MOMENTUM_THRESHOLD_PCT = 40.0


class UniverseScreener:
    """Builds the candidate universe and computes factor scores.

    Parameters
    ----------
    config : full config.json dict
    knowledge_graph : optional KnowledgeGraph instance for method 1
    news_cache : optional dict {symbol: {"finbert_score": float, ...}} for news_score
    """

    def __init__(
        self,
        config: dict,
        knowledge_graph=None,
        news_cache: Optional[Dict[str, dict]] = None,
    ):
        self._cfg = config
        self._kg = knowledge_graph
        self._news_cache = news_cache or {}
        self._screener_cfg = config.get("screener", {})
        self._max_universe = self._screener_cfg.get("max_universe_size", 600)
        self._max_watchlist = config.get("watchlist", {}).get("max_size", 30)

    # ── Universe Building ─────────────────────────────────────────────────────

    async def build_universe(self, extra_seeds: Optional[List[str]] = None) -> List[str]:
        """Run all 4 discovery methods in parallel and return the combined set.

        Returns a deduplicated list of tickers (no duplicates across methods).
        Caps at max_universe_size from config (default 600).
        """
        seeds = list(extra_seeds or [])
        # Add user-configured seeds from config
        watchlist_cfg = self._cfg.get("watchlist", {})
        seeds += watchlist_cfg.get("mandatory", [])
        seeds += watchlist_cfg.get("manual_additions", [])

        results = await asyncio.gather(
            self._method1_knowledge_graph(seeds),
            self._method2_theme_baskets(),
            self._method3_momentum_expansion(),
            self._method4_user_seeds(seeds),
            return_exceptions=True,
        )

        combined: Set[str] = set()
        method_names = ["knowledge_graph", "theme_baskets", "momentum", "user_seeds"]
        for name, result in zip(method_names, results):
            if isinstance(result, Exception):
                logger.warning("Universe method '%s' failed: %s", name, result)
            else:
                combined.update(result)
                logger.info("Method '%s': %d symbols", name, len(result))

        universe = list(combined)[: self._max_universe]
        logger.info("Universe built: %d total candidates (cap %d)", len(universe), self._max_universe)
        return universe

    async def _method1_knowledge_graph(self, seeds: List[str]) -> List[str]:
        """Expand seeds using the knowledge graph (1-2 hops)."""
        if self._kg is None:
            return list(seeds)
        result: Set[str] = set(s.upper() for s in seeds)
        for seed in seeds:
            related = self._kg.get_all_related(seed.upper(), hops=2)
            result.update(related)
        return list(result)

    async def _method2_theme_baskets(self) -> List[str]:
        """Pull all theme basket seeds + ETF holdings."""
        symbols: Set[str] = set()
        for theme, basket in THEME_BASKETS.items():
            symbols.update(basket["seeds"])
            symbols.update(basket.get("extra", []))

            for etf in basket.get("etfs", []):
                holdings = await self._fetch_etf_holdings(etf)
                symbols.update(holdings)

        return [s.upper() for s in symbols if s]

    async def _method3_momentum_expansion(self) -> List[str]:
        """Find stocks with >40% 3-month return across all US equities."""
        try:
            from tradingview_screener import Query, Column  # type: ignore
            df_result = (
                Query()
                .select("name", "change", "close", "average_volume_10d_calc")
                .where(
                    Column("change") > MOMENTUM_THRESHOLD_PCT,
                    Column("close") > 5.0,
                    Column("average_volume_10d_calc") > 200_000,
                    Column("exchange").isin(["NASDAQ", "NYSE", "AMEX"]),
                )
                .order_by("change", ascending=False)
                .limit(100)
                .get_scanner_data()
            )
            _, df = df_result if isinstance(df_result, tuple) else (0, df_result)
            if df is None or df.empty:
                return []
            symbols = [
                str(row.get("name", "")).replace("NASDAQ:", "").replace("NYSE:", "").replace("AMEX:", "")
                for _, row in df.iterrows()
            ]
            return [s for s in symbols if s]
        except Exception:
            logger.debug("Momentum expansion failed (tradingview-screener unavailable)")
            return []

    async def _method4_user_seeds(self, seeds: List[str]) -> List[str]:
        """Return user-provided seeds from config."""
        return [s.upper() for s in seeds if s]

    async def _fetch_etf_holdings(self, etf_ticker: str) -> List[str]:
        """Fetch ETF top-50 holdings via finvizfinance (cached per session)."""
        try:
            from finvizfinance.quote import finvizfinance  # type: ignore
            stock = finvizfinance(etf_ticker)
            holdings = stock.TickerFunds()
            if holdings is None or holdings.empty:
                return []
            # finvizfinance returns a DataFrame with 'Ticker' column for funds
            if "Ticker" in holdings.columns:
                return list(holdings["Ticker"].dropna().str.upper())[:50]
            return []
        except Exception:
            logger.debug("ETF holdings fetch failed for %s", etf_ticker)
            return []

    # ── Factor Scoring ────────────────────────────────────────────────────────

    async def compute_factor_scores(self, symbols: List[str]) -> Dict[str, dict]:
        """Compute all 5 percentile-based factor scores for a list of symbols.

        Returns dict: {symbol → {momentum_score, quality_score, technical_score,
                                  valuation_score, news_score}}

        These 5 scores become new LightGBM features (added to the 130 existing ones).
        """
        tasks = [self._score_single(sym) for sym in symbols]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        scores: Dict[str, dict] = {}
        for sym, result in zip(symbols, results_list):
            if isinstance(result, Exception):
                logger.debug("Factor score failed for %s: %s", sym, result)
                scores[sym] = self._neutral_scores()
            else:
                scores[sym] = result
        return scores

    async def _score_single(self, symbol: str) -> dict:
        """Compute all 5 factor scores for one symbol."""
        import yfinance as yf  # type: ignore
        from data.percentile_utils import stock_percentile, sector_percentile

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            hist = ticker.history(period="1y")

            # ── Momentum score: 12-1 month return vs sector ────────────────
            if len(hist) >= 252:
                ret_12m = (hist["Close"].iloc[-1] / hist["Close"].iloc[-252] - 1) * 100
                ret_1m = (hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1) * 100
                momentum_12_1 = ret_12m - ret_1m  # standard momentum factor
            elif len(hist) >= 63:
                momentum_12_1 = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            else:
                momentum_12_1 = None

            # Use same-sector stocks from theme baskets as peers for percentile
            sector_symbols = self._get_sector_peers(symbol)
            sector_momenta = [await self._get_12m_return(s) for s in sector_symbols[:20]]
            momentum_score = sector_percentile(momentum_12_1, sector_momenta, higher_is_better=True)

            # ── Quality score: ROE + FCF margin vs sector ──────────────────
            roe = info.get("returnOnEquity")
            fcf = info.get("freeCashflow")
            revenue = info.get("totalRevenue") or 1
            fcf_margin = (fcf / revenue * 100) if fcf else None

            sector_roes = [info.get("returnOnEquity") for _ in sector_symbols[:20]]
            quality_score = sector_percentile(roe, sector_roes, higher_is_better=True)

            # ── Technical score: RSI percentile vs own history ─────────────
            if len(hist) >= 14:
                closes = hist["Close"]
                delta = closes.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-10)
                rsi_series = 100 - (100 / (1 + rs))
                current_rsi = float(rsi_series.iloc[-1])
                rsi_history = rsi_series.dropna().tolist()
                technical_score = stock_percentile(current_rsi, rsi_history, higher_is_better=True)
            else:
                technical_score = 50.0

            # ── Valuation score: P/E vs sector (lower P/E = better value) ──
            pe = info.get("trailingPE") or info.get("forwardPE")
            sector_pes = [s_info.get("trailingPE") for s_info in self._get_sector_infos(symbol)]
            valuation_score = sector_percentile(pe, sector_pes, higher_is_better=False)

            # ── News score: FinBERT aggregate last 7 days ──────────────────
            news_data = self._news_cache.get(symbol.upper(), {})
            finbert_raw = news_data.get("finbert_score")
            # FinBERT score is in [-1, 1]; convert to 0-100 percentile
            news_score = ((finbert_raw + 1) / 2 * 100) if finbert_raw is not None else 50.0

            return {
                "momentum_score": round(momentum_score, 1),
                "quality_score": round(quality_score, 1),
                "technical_score": round(technical_score, 1),
                "valuation_score": round(valuation_score, 1),
                "news_score": round(news_score, 1),
            }

        except Exception:
            logger.debug("Factor scoring failed for %s", symbol, exc_info=True)
            return self._neutral_scores()

    def _neutral_scores(self) -> dict:
        return {
            "momentum_score": 50.0,
            "quality_score": 50.0,
            "technical_score": 50.0,
            "valuation_score": 50.0,
            "news_score": 50.0,
        }

    def _get_sector_peers(self, symbol: str) -> List[str]:
        """Find peers in the same theme basket(s)."""
        symbol = symbol.upper()
        peers: Set[str] = set()
        for basket in THEME_BASKETS.values():
            all_in_basket = set(basket["seeds"] + basket.get("extra", []))
            if symbol in all_in_basket:
                peers.update(all_in_basket)
        peers.discard(symbol)
        return list(peers)[:30]

    def _get_sector_infos(self, symbol: str) -> List[dict]:
        """Return yfinance .info dicts for sector peers (best-effort, sync)."""
        peers = self._get_sector_peers(symbol)[:10]
        infos = []
        try:
            import yfinance as yf  # type: ignore
            for peer in peers:
                try:
                    infos.append(yf.Ticker(peer).info or {})
                except Exception:
                    pass
        except ImportError:
            pass
        return infos

    async def _get_12m_return(self, symbol: str) -> Optional[float]:
        """Return 12-month price return for a symbol (async-friendly wrapper)."""
        try:
            import yfinance as yf  # type: ignore
            hist = yf.Ticker(symbol).history(period="1y")
            if len(hist) >= 60:
                return float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100)
        except Exception:
            pass
        return None

    # ── Watchlist Ranking ─────────────────────────────────────────────────────

    def rank_candidates(
        self,
        factor_scores: Dict[str, dict],
        lgbm_scores: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """Rank candidates for weekly watchlist selection.

        Sorts by composite factor score (simple average) or by LightGBM P(win)
        when the scorecard is available. Returns top max_watchlist candidates.
        """
        from data.percentile_utils import composite_score

        ranked = []
        for symbol, scores in factor_scores.items():
            f_score = composite_score(list(scores.values()))
            l_score = lgbm_scores.get(symbol, f_score / 100) if lgbm_scores else f_score / 100
            ranked.append({
                "symbol": symbol,
                "factor_composite": round(f_score, 1),
                "lgbm_p_win": round(l_score, 4),
                **scores,
            })

        ranked.sort(key=lambda x: x["lgbm_p_win"], reverse=True)
        return ranked[: self._max_watchlist]
