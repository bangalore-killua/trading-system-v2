"""
seed_watchlist.py — Build initial 30-stock diversified watchlist.

Run once on first setup before starting the server. After this, the weekly
universe rebuild (UniverseScreener) takes over automatically.

What it does:
  1. Picks 3 stocks per theme basket (best recent momentum from each basket)
  2. Fetches live price + basic fundamentals via yfinance
  3. Writes a signal row per stock to the database (signal=HOLD, score=50)
  4. Logs each addition to watchlist_history
  5. Adds all stocks to the hot_universe cache (TTL = 4 weeks / default catalyst)

Usage:
  python scripts/seed_watchlist.py
  python scripts/seed_watchlist.py --dry   # print picks without writing to DB
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seed_watchlist")

# ── Seed basket: 3 stocks per theme, prioritised by typical liquidity ─────────
# These are starting seeds; the screener will refine the list weekly.

SEED_STOCKS = [
    # AI Infrastructure
    {"symbol": "NVDA",  "bundle_id": "ai_infrastructure",   "bundle_name": "AI Infrastructure",   "bundle_emoji": "🤖"},
    {"symbol": "AMD",   "bundle_id": "ai_infrastructure",   "bundle_name": "AI Infrastructure",   "bundle_emoji": "🤖"},
    {"symbol": "ANET",  "bundle_id": "ai_infrastructure",   "bundle_name": "AI Infrastructure",   "bundle_emoji": "🤖"},
    # Cloud Hyperscaler
    {"symbol": "MSFT",  "bundle_id": "cloud_hyperscaler",   "bundle_name": "Cloud Hyperscaler",   "bundle_emoji": "☁️"},
    {"symbol": "GOOGL", "bundle_id": "cloud_hyperscaler",   "bundle_name": "Cloud Hyperscaler",   "bundle_emoji": "☁️"},
    {"symbol": "AMZN",  "bundle_id": "cloud_hyperscaler",   "bundle_name": "Cloud Hyperscaler",   "bundle_emoji": "☁️"},
    # Crypto / Fintech
    {"symbol": "COIN",  "bundle_id": "crypto_fintech",      "bundle_name": "Crypto / Fintech",    "bundle_emoji": "₿"},
    {"symbol": "MSTR",  "bundle_id": "crypto_fintech",      "bundle_name": "Crypto / Fintech",    "bundle_emoji": "₿"},
    {"symbol": "HOOD",  "bundle_id": "crypto_fintech",      "bundle_name": "Crypto / Fintech",    "bundle_emoji": "₿"},
    # Financials
    {"symbol": "JPM",   "bundle_id": "financials",          "bundle_name": "Financials",          "bundle_emoji": "🏦"},
    {"symbol": "GS",    "bundle_id": "financials",          "bundle_name": "Financials",          "bundle_emoji": "🏦"},
    {"symbol": "V",     "bundle_id": "financials",          "bundle_name": "Financials",          "bundle_emoji": "🏦"},
    # Energy
    {"symbol": "XOM",   "bundle_id": "energy",              "bundle_name": "Energy",              "bundle_emoji": "⚡"},
    {"symbol": "CVX",   "bundle_id": "energy",              "bundle_name": "Energy",              "bundle_emoji": "⚡"},
    {"symbol": "COP",   "bundle_id": "energy",              "bundle_name": "Energy",              "bundle_emoji": "⚡"},
    # Consumer Discretionary
    {"symbol": "TSLA",  "bundle_id": "consumer_disc",       "bundle_name": "Consumer Disc.",      "bundle_emoji": "🛍️"},
    {"symbol": "AMZN",  "bundle_id": "consumer_disc",       "bundle_name": "Consumer Disc.",      "bundle_emoji": "🛍️"},
    {"symbol": "NKE",   "bundle_id": "consumer_disc",       "bundle_name": "Consumer Disc.",      "bundle_emoji": "🛍️"},
    # Healthcare
    {"symbol": "LLY",   "bundle_id": "healthcare",          "bundle_name": "Healthcare",          "bundle_emoji": "🏥"},
    {"symbol": "UNH",   "bundle_id": "healthcare",          "bundle_name": "Healthcare",          "bundle_emoji": "🏥"},
    {"symbol": "ISRG",  "bundle_id": "healthcare",          "bundle_name": "Healthcare",          "bundle_emoji": "🏥"},
    # Defence / Industrial
    {"symbol": "LMT",   "bundle_id": "defence",             "bundle_name": "Defence",             "bundle_emoji": "🛡️"},
    {"symbol": "RTX",   "bundle_id": "defence",             "bundle_name": "Defence",             "bundle_emoji": "🛡️"},
    {"symbol": "NOC",   "bundle_id": "defence",             "bundle_name": "Defence",             "bundle_emoji": "🛡️"},
    # Consumer Staples
    {"symbol": "COST",  "bundle_id": "consumer_staples",    "bundle_name": "Consumer Staples",    "bundle_emoji": "🛒"},
    {"symbol": "PG",    "bundle_id": "consumer_staples",    "bundle_name": "Consumer Staples",    "bundle_emoji": "🛒"},
    {"symbol": "WMT",   "bundle_id": "consumer_staples",    "bundle_name": "Consumer Staples",    "bundle_emoji": "🛒"},
    # Metals / Commodities
    {"symbol": "NEM",   "bundle_id": "metals",              "bundle_name": "Metals & Commodities","bundle_emoji": "⛏️"},
    {"symbol": "FCX",   "bundle_id": "metals",              "bundle_name": "Metals & Commodities","bundle_emoji": "⛏️"},
    {"symbol": "GLD",   "bundle_id": "metals",              "bundle_name": "Metals & Commodities","bundle_emoji": "⛏️"},
]


def fetch_price(symbol: str) -> float:
    """Fetch current price via yfinance. Returns 0.0 on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        return float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
    except Exception:
        return 0.0


def build_signal_row(stock: dict, price: float) -> dict:
    """Build the scorecard dict expected by db.save_signal()."""
    return {
        "symbol":           stock["symbol"],
        "signal":           "HOLD",
        "signal_label":     "HOLD",
        "normalised_score": 50,
        "overall_display":  "Seeded",
        "conviction":       "SEEDED",
        "bundle_id":        stock["bundle_id"],
        "bundle_name":      stock["bundle_name"],
        "bundle_emoji":     stock["bundle_emoji"],
        "tier_id":          "T2",
        "tier_name":        "Core",
        "position_pct":     0,
        "price":            price,
        "trend_intact":     True,
        "trend_filter_applied": False,
        "trade_id":         None,
    }


def main(dry_run: bool = False) -> None:
    # Deduplicate (AMZN appears in cloud + consumer — keep first)
    seen: set = set()
    stocks = []
    for s in SEED_STOCKS:
        if s["symbol"] not in seen:
            seen.add(s["symbol"])
            stocks.append(s)

    log.info("Seeding %d stocks (dry_run=%s)", len(stocks), dry_run)

    if not dry_run:
        from database import get_db
        db = get_db()

        from data.hot_universe import get_hot_universe, TriggerSource
        hu = get_hot_universe()

    written = 0
    failed  = 0

    for stock in stocks:
        sym = stock["symbol"]
        log.info("  Fetching %s ...", sym)
        price = fetch_price(sym)

        if dry_run:
            print(f"  {sym:6s}  ${price:8.2f}  {stock['bundle_name']}")
            continue

        # Write signal row to DB
        try:
            signal_row = build_signal_row(stock, price)
            db.save_signal(sym, signal_row)
            written += 1
        except Exception as e:
            log.warning("  save_signal failed for %s: %s", sym, e)
            failed += 1
            continue

        # Log watchlist addition
        try:
            db.log_watchlist_change(
                symbol    = sym,
                action    = "added",
                bucket    = stock["bundle_id"],
                reason    = "Initial seed — scripts/seed_watchlist.py",
                bundle_id = stock["bundle_id"],
            )
        except Exception as e:
            log.debug("  log_watchlist_change failed for %s: %s", sym, e)

        # Add to hot_universe cache (4-week default TTL)
        try:
            hu.add(
                symbol       = sym,
                source       = TriggerSource.NEWS_EVENT,
                headline     = f"Seed: {stock['bundle_name']}",
                catalyst_type= "default",
            )
        except Exception as e:
            log.debug("  hot_universe.add failed for %s: %s", sym, e)

    if not dry_run:
        log.info("Done. Written: %d  Failed: %d", written, failed)
        log.info("Start the server now: python run_server.py")
    else:
        print(f"\nTotal: {len(stocks)} stocks across 10 theme baskets (dry run — nothing written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed initial watchlist")
    parser.add_argument("--dry", action="store_true", help="Print picks without writing to DB")
    args = parser.parse_args()
    main(dry_run=args.dry)
