"""
knowledge_graph.py — Financial Knowledge Graph (Bloomberg SPLC equivalent)

Maps confirmed structural relationships between companies: suppliers, customers,
ecosystem players, competitors, and JV partners. Built from public SEC filings,
not news — these are permanent structural relationships, not event-driven signals.

Sources:
  1. SEC EDGAR full-text search on 10-K filings (free, no API key)
  2. Reverse traversal: who mentions our seed companies in THEIR 10-K

Storage: NetworkX DiGraph (no database needed for <2000 companies)
Update cadence: quarterly (when new 10-K filings drop on EDGAR)

Relationship types:
  supplier    → stock A buys critical inputs from stock B
  customer    → stock A sells to stock B
  ecosystem   → stock A depends on stock B's platform/infrastructure
  competitor  → direct market competitors
  jv_partner  → confirmed joint ventures or strategic alliances

Usage:
  kg = KnowledgeGraph()
  kg.build(seeds=["NVDA", "MSFT"], hops=2)   # async, takes ~5 min on first run
  related = kg.get_related("NVDA", rel_type="ecosystem")  # → ["APLD", "ALAB", ...]
  kg.save("models/knowledge_graph.json")
  kg.load("models/knowledge_graph.json")
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

# EDGAR full-text search endpoint (free, no auth)
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms=10-K"
EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=5&search_text=&output=atom"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Relationship extraction patterns for 10-K text
RELATIONSHIP_PATTERNS = {
    "supplier": [
        r"(?:we|our company) (?:purchase|procure|source|obtain|rely on|depend on) .{0,60} from ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies|Systems|Holdings)\.?)",
        r"([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies|Systems|Holdings)\.?) (?:supplies|provides|manufactures|produces) .{0,60} (?:to us|for us|to our)",
        r"sole (?:supplier|source|vendor) .{0,60} ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co)\.?)",
    ],
    "customer": [
        r"(?:our|significant) (?:customers?|clients?) (?:include|such as) ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies)\.?)",
        r"([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies)\.?) (?:accounted for|represented) .{0,30} (?:of our|of total) (?:revenue|sales|net revenue)",
    ],
    "ecosystem": [
        r"(?:built on|powered by|running on|uses?|utilizes?|deployed on) ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Cloud|Platform|AI)\.?)",
        r"(?:partnership|collaboration|integration) with ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies)\.?)",
    ],
    "competitor": [
        r"(?:compete with|competition from|competitors? (?:include|such as)) ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies)\.?)",
    ],
    "jv_partner": [
        r"joint venture with ([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies)\.?)",
        r"([A-Z][A-Za-z\s,\.]+(?:Inc|Corp|Ltd|LLC|Co|Technologies)\.?) .{0,30} joint venture",
    ],
}

# Known ticker → company name mapping for matching extracted names to tickers
KNOWN_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META", "tesla": "TSLA",
    "taiwan semiconductor": "TSM", "tsmc": "TSM", "asml": "ASML",
    "broadcom": "AVGO", "qualcomm": "QCOM", "intel": "INTC", "amd": "AMD",
    "synopsys": "SNPS", "cadence": "CDNS", "applied materials": "AMAT",
    "lam research": "LRCX", "kla": "KLAC", "marvell": "MRVL",
    "supermicro": "SMCI", "arista": "ANET", "vertiv": "VRT",
    "applied digital": "APLD", "astera labs": "ALAB",
    "salesforce": "CRM", "oracle": "ORCL", "sap": "SAP",
    "coinbase": "COIN", "microstrategy": "MSTR",
}

REL_TYPES = list(RELATIONSHIP_PATTERNS.keys())


class KnowledgeGraph:
    """Financial supply-chain and ecosystem relationship graph.

    The graph is a NetworkX DiGraph where each directed edge (A → B) carries
    a 'rel_type' attribute: supplier / customer / ecosystem / competitor / jv_partner.

    Reverse edges are NOT added automatically — A→B (supplier) means A supplies B,
    NOT that B has A as a customer. Use get_related(B, rel_type='customer') to
    find who buys FROM B (edges pointing to B).
    """

    GRAPH_FILE = "models/knowledge_graph.json"

    def __init__(self, graph_path: str = GRAPH_FILE):
        try:
            import networkx as nx
            self._G = nx.DiGraph()
        except ImportError:
            raise RuntimeError("networkx not installed — run: pip install networkx>=3.3")

        self._path = Path(graph_path)
        self._ticker_to_name: Dict[str, str] = {}
        self._name_to_ticker: Dict[str, str] = dict(KNOWN_TICKERS)

        if self._path.exists():
            self.load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_related(
        self,
        symbol: str,
        rel_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[str]:
        """Return related symbols.

        Parameters
        ----------
        symbol    : the anchor ticker (e.g. "NVDA")
        rel_type  : one of supplier/customer/ecosystem/competitor/jv_partner or None (all)
        direction : "outgoing" = stocks that symbol points TO (e.g. NVDA→TSM supplier)
                    "incoming" = stocks that point TO symbol (e.g. who has NVDA as customer)
        """
        symbol = symbol.upper()
        if symbol not in self._G:
            return []

        if direction == "outgoing":
            edges = self._G.out_edges(symbol, data=True)
        else:
            edges = self._G.in_edges(symbol, data=True)

        result = []
        for src, dst, data in edges:
            if rel_type is None or data.get("rel_type") == rel_type:
                result.append(dst if direction == "outgoing" else src)
        return list(set(result))

    def get_all_related(self, symbol: str, hops: int = 2) -> Set[str]:
        """Return all symbols reachable from *symbol* within *hops* steps."""
        try:
            import networkx as nx
        except ImportError:
            return set()

        symbol = symbol.upper()
        if symbol not in self._G:
            return set()

        reachable = set()
        for hop in range(1, hops + 1):
            for target in nx.single_source_shortest_path_length(self._G, symbol, cutoff=hop):
                if target != symbol:
                    reachable.add(target)
        return reachable

    def add_edge(self, src: str, dst: str, rel_type: str, confidence: float = 1.0) -> None:
        """Add a directed relationship edge."""
        if rel_type not in REL_TYPES:
            logger.warning("Unknown rel_type '%s' — skipping edge %s→%s", rel_type, src, dst)
            return
        src, dst = src.upper(), dst.upper()
        if self._G.has_edge(src, dst):
            existing = self._G[src][dst].get("rel_type", "")
            if existing == rel_type:
                return  # Already recorded
        self._G.add_edge(src, dst, rel_type=rel_type, confidence=confidence)
        logger.debug("Edge added: %s -[%s]→ %s", src, rel_type, dst)

    def node_count(self) -> int:
        return self._G.number_of_nodes()

    def edge_count(self) -> int:
        return self._G.number_of_edges()

    def summary(self) -> str:
        lines = [f"KnowledgeGraph: {self.node_count()} nodes, {self.edge_count()} edges"]
        for rt in REL_TYPES:
            count = sum(1 for _, _, d in self._G.edges(data=True) if d.get("rel_type") == rt)
            lines.append(f"  {rt:12s}: {count} edges")
        return "\n".join(lines)

    # ── Build from SEC EDGAR ──────────────────────────────────────────────────

    async def build(
        self,
        seeds: List[str],
        hops: int = 2,
        years_back: int = 2,
    ) -> None:
        """Build the knowledge graph from SEC 10-K filings.

        Steps:
          1. Load CIK→ticker mapping from SEC EDGAR
          2. For each seed + their N-hop expanded universe: fetch latest 10-K
          3. Extract relationships using regex patterns
          4. Map company names → tickers
          5. Save graph to disk

        This is slow (~5 min for 50 companies) — run quarterly or on demand.
        Rate-limited to 5 req/s per SEC EDGAR guidelines.
        """
        logger.info("KnowledgeGraph.build() — seeds: %s, hops: %d", seeds, hops)
        await self._load_sec_ticker_mapping()

        # Expand universe via graph hops (start with just seeds on first run)
        to_process: Set[str] = set(s.upper() for s in seeds)
        processed: Set[str] = set()

        for hop in range(hops):
            logger.info("Hop %d/%d: processing %d symbols", hop + 1, hops, len(to_process))
            new_symbols: Set[str] = set()

            for symbol in list(to_process - processed):
                extracted = await self._process_symbol(symbol, years_back)
                new_symbols.update(extracted)
                processed.add(symbol)
                time.sleep(0.2)  # SEC rate limit: max 5 req/s

            to_process = new_symbols - processed

        self.save()
        logger.info("KnowledgeGraph built: %s", self.summary())

    async def _process_symbol(self, symbol: str, years_back: int) -> Set[str]:
        """Fetch and parse the latest 10-K for symbol. Returns newly found tickers."""
        cik = self._ticker_to_cik(symbol)
        if not cik:
            logger.debug("No CIK found for %s", symbol)
            return set()

        text = await self._fetch_10k_text(cik, years_back)
        if not text:
            return set()

        found_tickers: Set[str] = set()
        for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group(1).strip().lower()
                    ticker = self._name_to_ticker.get(name)
                    if not ticker:
                        for known_name, known_ticker in self._name_to_ticker.items():
                            if known_name in name or name in known_name:
                                ticker = known_ticker
                                break
                    if ticker and ticker != symbol:
                        self.add_edge(symbol, ticker, rel_type)
                        found_tickers.add(ticker)

        logger.info("Processed %s — found %d relationships, %d new tickers",
                    symbol, self.edge_count(), len(found_tickers))
        return found_tickers

    async def _fetch_10k_text(self, cik: str, years_back: int) -> str:
        """Fetch the text of the most recent 10-K for a CIK."""
        try:
            async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "trading-system-v2/1.0"}) as client:
                # Get filing index
                url = EDGAR_FILING_URL.format(cik=cik)
                resp = await client.get(url)
                if resp.status_code != 200:
                    return ""

                # Extract first 10-K filing link from Atom feed
                import feedparser
                parsed = feedparser.parse(resp.text)
                entries = parsed.get("entries", [])
                if not entries:
                    return ""

                filing_url = entries[0].get("link", "")
                if not filing_url:
                    return ""

                # Fetch filing index page to find the actual 10-K document
                idx_resp = await client.get(filing_url)
                if idx_resp.status_code != 200:
                    return ""

                # Find the primary document link (usually the largest .htm file)
                htm_links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', idx_resp.text)
                if not htm_links:
                    return ""

                doc_url = f"https://www.sec.gov{htm_links[0]}"
                doc_resp = await client.get(doc_url)
                if doc_resp.status_code != 200:
                    return ""

                # Strip HTML tags for text extraction
                text = re.sub(r"<[^>]+>", " ", doc_resp.text)
                text = re.sub(r"\s+", " ", text)
                return text[:500_000]  # limit to first 500K chars

        except Exception:
            logger.debug("Failed to fetch 10-K for CIK %s", cik)
            return ""

    async def _load_sec_ticker_mapping(self) -> None:
        """Load CIK→ticker and name→ticker from SEC's company_tickers.json."""
        try:
            async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "trading-system-v2/1.0"}) as client:
                resp = await client.get(COMPANY_TICKERS_URL)
                data = resp.json()
            for entry in data.values():
                cik = str(entry.get("cik_str", "")).zfill(10)
                ticker = entry.get("ticker", "").upper()
                name = entry.get("title", "").lower().strip()
                if ticker and cik:
                    self._ticker_to_name[ticker] = cik
                if name and ticker:
                    self._name_to_ticker[name] = ticker
            logger.info("Loaded %d SEC CIK mappings", len(self._ticker_to_name))
        except Exception:
            logger.exception("Failed to load SEC ticker mapping")

    def _ticker_to_cik(self, symbol: str) -> Optional[str]:
        return self._ticker_to_name.get(symbol.upper())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """Serialize graph to JSON."""
        out_path = Path(path) if path else self._path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": list(self._G.nodes()),
            "edges": [
                {"src": src, "dst": dst, **attr}
                for src, dst, attr in self._G.edges(data=True)
            ],
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("KnowledgeGraph saved → %s (%d nodes, %d edges)",
                    out_path, self.node_count(), self.edge_count())

    def load(self, path: Optional[str] = None) -> None:
        """Deserialize graph from JSON."""
        in_path = Path(path) if path else self._path
        if not in_path.exists():
            logger.warning("Knowledge graph file not found at %s", in_path)
            return
        try:
            with open(in_path) as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                self._G.add_node(node)
            for edge in data.get("edges", []):
                self._G.add_edge(
                    edge["src"], edge["dst"],
                    rel_type=edge.get("rel_type", "unknown"),
                    confidence=edge.get("confidence", 1.0),
                )
            logger.info("KnowledgeGraph loaded from %s — %s", in_path, self.summary())
        except Exception:
            logger.exception("Failed to load KnowledgeGraph from %s", in_path)

    def get_graph_json(self, symbol: Optional[str] = None, hops: int = 1) -> dict:
        """Return Cytoscape.js-compatible node/edge JSON for the spider-web UI.

        If *symbol* is given, returns only the subgraph reachable within *hops*.
        Otherwise returns the full graph (may be large).
        """
        if symbol:
            symbol = symbol.upper()
            relevant = self.get_all_related(symbol, hops=hops) | {symbol}
            nodes = [n for n in self._G.nodes() if n in relevant]
            edges = [
                (src, dst, d)
                for src, dst, d in self._G.edges(data=True)
                if src in relevant and dst in relevant
            ]
        else:
            nodes = list(self._G.nodes())
            edges = list(self._G.edges(data=True))

        return {
            "nodes": [{"data": {"id": n, "label": n}} for n in nodes],
            "edges": [
                {
                    "data": {
                        "id": f"{src}-{dst}",
                        "source": src,
                        "target": dst,
                        "rel_type": d.get("rel_type", "unknown"),
                        "confidence": d.get("confidence", 1.0),
                    }
                }
                for src, dst, d in edges
            ],
        }


# Module-level singleton
_kg_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Return the module-level singleton KnowledgeGraph."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance
