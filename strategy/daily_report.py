"""
daily_report.py — End-of-day opportunity audit.

At 4:30 PM after market close, scan the day's top % gainers from the active
universe and answer: "Was this picked?  If not, why not?"

Output:  logs/daily_report_YYYY-MM-DD.txt
Trigger: called by TradingScheduler._generate_performance_report() alongside
         the quantstats HTML tearsheet.

Why this exists:
  The Telegram /today command shows every symbol that was *evaluated*, but it
  doesn't show symbols that gained +5% but were never even looked at (outside
  the 60-min cycle, not hot-universe seeded).  This report bridges that gap.
"""

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH   = Path("logs/trade_lifecycle.db")
REPORT_DIR = Path("logs")


def _fetch_today_evals() -> dict:
    """Return {symbol: list[dict]} of trade_logs rows for today's SIGNAL/MODELS stages."""
    if not _DB_PATH.exists():
        return {}
    cutoff = date.today().isoformat()
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_logs WHERE ts >= ? AND stage IN ('SIGNAL','MODELS','ENTRY','CONSTRAINTS') "
            "ORDER BY symbol, id",
            (cutoff,),
        ).fetchall()
        conn.close()
    except Exception:
        return {}

    result: dict = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in result:
            result[sym] = []
        result[sym].append(dict(r))
    return result


def _fetch_ohlcv_today(symbols: list) -> dict:
    """Return {symbol: pct_change_today} using yfinance for a batch of symbols."""
    if not symbols:
        return {}
    try:
        import yfinance as yf
        data = yf.download(
            tickers=" ".join(symbols),
            period="2d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
        if closes.empty:
            return {}
        pct = closes.pct_change().iloc[-1]
        return {sym: float(pct.get(sym, 0.0) or 0.0) for sym in symbols}
    except Exception as exc:
        logger.debug("OHLCV fetch for daily report failed: %s", exc)
        return {}


def _load_watchlist(config_path: Path = Path("config.json")) -> list:
    import json
    try:
        cfg = json.loads(config_path.read_text())
        wl = cfg.get("watchlist", {})
        return (
            wl.get("mandatory", [])
            + wl.get("screener_picks", [])
            + wl.get("manual_additions", [])
        )
    except Exception:
        return []


def _load_open_positions(config_path: Path = Path("config.json")) -> list:
    """Return symbols currently in open positions from trade_logs ENTRY stage."""
    if not _DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        # Symbols that have an ENTRY-PASS but no OUTCOME yet
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM trade_logs WHERE stage='ENTRY' AND result='PASS'"
        ).fetchall()
        entered = {r["symbol"] for r in rows}
        exited = {
            r["symbol"]
            for r in conn.execute(
                "SELECT DISTINCT symbol FROM trade_logs WHERE stage='OUTCOME'"
            ).fetchall()
        }
        conn.close()
        return list(entered - exited)
    except Exception:
        return []


def generate(state: Optional[dict] = None) -> Path:
    """Build the daily report and write it to logs/daily_report_YYYY-MM-DD.txt.

    Parameters
    ----------
    state : injected live state dict from TradingScheduler (optional).
            Falls back to loading config.json and trade_logs directly.

    Returns
    -------
    Path to the written report file.
    """
    today_str = date.today().isoformat()
    report_path = REPORT_DIR / f"daily_report_{today_str}.txt"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build universe to scan
    watchlist = _load_watchlist()
    open_positions = _load_open_positions()
    universe = list(dict.fromkeys(watchlist + open_positions))  # dedup, preserve order

    if not universe:
        report_path.write_text(f"[{today_str}] No universe symbols found — report skipped.\n")
        return report_path

    # 2. Fetch today's % change for all universe symbols
    pct_map = _fetch_ohlcv_today(universe)

    # 3. Rank by absolute gain (top 20 movers, both up and down)
    ranked = sorted(pct_map.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    # 4. Load today's evaluation log
    evals = _fetch_today_evals()

    # 5. Compose report
    lines = [
        "=" * 64,
        f"  DAILY OPPORTUNITY AUDIT — {today_str}",
        f"  Universe: {len(universe)} symbols  |  Generated: {datetime.now().strftime('%H:%M')}",
        "=" * 64,
        "",
        f"{'Symbol':8s}  {'Chg%':>7s}  {'Status':15s}  Notes",
        "-" * 64,
    ]

    for sym, pct in ranked:
        pct_str = f"{pct:+.1%}"
        in_position = sym in open_positions
        was_evaluated = sym in evals

        if in_position:
            status = "HOLDING"
            notes  = "position open"
        elif was_evaluated:
            stages  = evals[sym]
            entry_s = next((s for s in stages if s["stage"] == "ENTRY"), None)
            if entry_s and entry_s.get("result") == "PASS":
                status = "ENTERED"
                notes  = entry_s.get("why", "")[:55]
            else:
                # Find the first FAIL/SKIP stage to explain why
                fail = next((s for s in stages if s.get("result") in ("FAIL", "SKIP")), None)
                status = "EVALUATED/SKIP"
                notes  = (fail.get("why", "") if fail else "")[:55]
        else:
            # Symbol was in universe but never reached evaluation today
            status = "NOT EVALUATED"
            notes  = "outside 60-min scoring window or HOT_UNIVERSE TTL expired"

        lines.append(f"{sym:8s}  {pct_str:>7s}  {status:15s}  {notes}")

    # 6. Summary stats
    entered_today = [s for s in evals if any(
        r["stage"] == "ENTRY" and r.get("result") == "PASS"
        for r in evals[s]
    )]
    top_gainer   = ranked[0] if ranked else ("N/A", 0.0)
    missed_gainers = [
        (sym, pct) for sym, pct in ranked
        if pct > 0.03 and sym not in entered_today and sym not in open_positions
    ]

    lines += [
        "",
        "-" * 64,
        f"  Entered today      : {len(entered_today)} trade(s) — {', '.join(entered_today[:8])}",
        f"  Currently holding  : {len(open_positions)} position(s)",
        f"  Top mover today    : {top_gainer[0]} {top_gainer[1]:+.1%}",
    ]
    if missed_gainers:
        lines.append(f"  Missed >3% gainers : {', '.join(f'{s} {p:+.1%}' for s, p in missed_gainers[:5])}")
    else:
        lines.append("  Missed >3% gainers : none — good coverage")
    lines += ["", "=" * 64]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Daily report written: %s (%d symbols audited)", report_path, len(ranked))
    return report_path
