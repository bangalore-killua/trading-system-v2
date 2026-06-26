"""
trade_ledger.py — LEDGER-1
Per-symbol markdown ledger + master chronological log + DB record.

Two complementary outputs per trade:
  logs/trade_ledger/{SYMBOL}.md  — all trades for that symbol, newest at bottom
  logs/trade_log.md              — every trade from all symbols, newest first

Public API:
  open_trade(...)    → trade_id str
  update_trade(...)  → None   (only writes if score changed >= 5 pts)
  record_partial(...) → None
  close_trade(...)   → None
"""

import os
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Directory setup ───────────────────────────────────────────────────────────

_REPO_ROOT    = os.path.join(os.path.dirname(__file__), "..")
_LEDGER_DIR   = os.path.join(_REPO_ROOT, "logs", "trade_ledger")
_MASTER_LOG   = os.path.join(_REPO_ROOT, "logs", "trade_log.md")

_SEPARATOR    = "═" * 44
_SUB_SEP      = "─" * 44


def _ensure_dirs() -> None:
    os.makedirs(_LEDGER_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(_MASTER_LOG), exist_ok=True)


def _ledger_path(symbol: str) -> str:
    safe = symbol.replace("/", "-").replace(":", "-")
    return os.path.join(_LEDGER_DIR, f"{safe}.md")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M EST")


# ── Trade ID ──────────────────────────────────────────────────────────────────

def _make_trade_id(symbol: str, entry_date: datetime) -> str:
    return f"{symbol}_{entry_date.strftime('%Y%m%dT%H%M')}"


# ── Trade counter per symbol ──────────────────────────────────────────────────

def _get_trade_number(symbol: str) -> int:
    """Returns the next trade number by counting existing TRADE # headers."""
    path = _ledger_path(symbol)
    if not os.path.exists(path):
        return 1
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return content.count("TRADE #") + 1
    except Exception:
        return 1


# ── Master log helpers ────────────────────────────────────────────────────────

def _append_master_log(line: str) -> None:
    """Prepends one line to trade_log.md (newest first)."""
    try:
        _ensure_dirs()
        existing = ""
        if os.path.exists(_MASTER_LOG):
            with open(_MASTER_LOG, encoding="utf-8") as f:
                existing = f.read()
        header = (
            "MASTER TRADE LOG — all symbols, descending time order\n"
            f"Last updated: {_now_str()}\n"
            f"{'═' * 64}\n\n"
        )
        # Strip old header lines (first 3 lines)
        lines = existing.splitlines()
        body  = "\n".join(lines[4:]) if len(lines) > 4 else existing
        with open(_MASTER_LOG, "w", encoding="utf-8") as f:
            f.write(header + line + "\n" + body)
    except Exception as e:
        log.warning(f"_append_master_log failed: {e}")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_write(trade_id: str, **fields) -> None:
    try:
        from database import get_db, TradeLedger
        from sqlalchemy.orm import Session
        db = get_db()
        with db._get_session() as sess:
            existing = sess.get(TradeLedger, trade_id)
            if existing:
                for k, v in fields.items():
                    setattr(existing, k, v)
                existing.updated_at = datetime.utcnow()
            else:
                rec = TradeLedger(trade_id=trade_id, **fields)
                sess.add(rec)
            sess.commit()
    except Exception as e:
        log.debug(f"_db_write({trade_id}): {e}")


# ── open_trade ────────────────────────────────────────────────────────────────

def open_trade(symbol: str,
               tier: str,
               bundle: str,
               entry_price: float,
               shares: float,
               setup_type: str,
               basket: dict,
               scorecard_snapshot: dict,
               regime: str,
               key_news: list,
               stop_price: float,
               exit_criteria: list) -> str:
    """
    Called immediately after a buy executes.
    Creates or appends to logs/trade_ledger/{SYMBOL}.md.
    Appends one-line summary to logs/trade_log.md.
    Writes DB record.
    Returns trade_id.
    """
    _ensure_dirs()
    now         = datetime.now()
    trade_id    = _make_trade_id(symbol, now)
    trade_num   = _get_trade_number(symbol)
    path        = _ledger_path(symbol)
    basket_score = basket.get("basket_score", 0)
    threshold    = basket.get("threshold", 0)
    signals_hit  = basket.get("signals_hit", [])
    sc_total     = scorecard_snapshot.get("overall_score",
                   scorecard_snapshot.get("total_score", 0))

    # ── Build basket signals block ────────────────────────────
    signals_block = "\n".join(
        f"  ✓ {s}" for s in (signals_hit[:8] if signals_hit else ["(none)"])
    )

    # ── Build scorecard at entry block ────────────────────────
    tech  = scorecard_snapshot.get("technical",   {})
    fund  = scorecard_snapshot.get("fundamental", {})
    macro = scorecard_snapshot.get("macro",       {})
    sc_block = (
        f"  Technical:      {tech.get('display', '?')}\n"
        f"  Fundamental:    {fund.get('display', '?')}\n"
        f"  Macro:          {macro.get('display', '?')}"
    )

    # ── Key news block ────────────────────────────────────────
    news_block = "\n".join(
        f"  {n[:100]}" for n in (key_news[:3] if key_news else ["(none)"])
    )

    # ── Exit criteria block ───────────────────────────────────
    exit_block = "\n".join(
        f"  • {c}" for c in (exit_criteria[:6] if exit_criteria else ["(standard)"])
    )

    # ── Write symbol ledger ───────────────────────────────────
    entry_block = (
        f"\n{_SEPARATOR}\n"
        f"TRADE #{trade_num}\n"
        f"{_SEPARATOR}\n"
        f"DATE:         {now.strftime('%Y-%m-%d %H:%M EST')}\n"
        f"ACTION:       BUY\n"
        f"PRICE:        ${entry_price:.2f}\n"
        f"SHARES:       {shares:.2f}\n"
        f"SETUP:        {setup_type}\n"
        f"REGIME:       {regime}\n"
        f"TIER:         {tier} | BUNDLE: {bundle}\n"
        f"TRADE_ID:     {trade_id}\n"
        f"\n"
        f"ENTRY SIGNALS FIRED (basket: {basket_score} / threshold: {threshold}):\n"
        f"{signals_block}\n"
        f"\n"
        f"SCORECARD AT ENTRY:\n"
        f"{sc_block}\n"
        f"\n"
        f"KEY NEWS (last 24h at entry):\n"
        f"{news_block}\n"
        f"\n"
        f"STOP LOSS:    ${stop_price:.2f}\n"
        f"EXIT CRITERIA FOR THIS TRADE:\n"
        f"{exit_block}\n"
    )

    # Header only for new file
    header = ""
    if not os.path.exists(path):
        header = (
            f"TICKER: {symbol}\n"
            f"TIER:   {tier} | BUNDLE: {bundle}\n"
            f"FILE CREATED: {now.strftime('%Y-%m-%d')}\n\n"
        )

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(header + entry_block)
    except Exception as e:
        log.warning(f"open_trade file write failed: {e}")

    # ── Master log (one line) ─────────────────────────────────
    master_line = (
        f"{now.strftime('%Y-%m-%d %H:%M')}  BUY   {symbol:<10s}"
        f" @ ${entry_price:<8.2f}  {setup_type:<18s}"
        f" Basket:{basket_score}/{threshold}"
    )
    _append_master_log(master_line)

    # ── DB record ─────────────────────────────────────────────
    _db_write(
        trade_id,
        symbol             = symbol,
        entry_date         = now,
        entry_price        = entry_price,
        shares             = shares,
        setup_type         = setup_type,
        entry_thesis       = setup_type,
        basket_score_entry = basket_score,
        scorecard_total_entry = int(sc_total),
        regime_entry       = regime,
        tier               = tier,
        bundle             = bundle,
        stop_price         = stop_price,
        full_context       = {
            "basket":     basket,
            "scorecard":  scorecard_snapshot,
            "key_news":   key_news,
            "exit_criteria": exit_criteria,
        },
        created_at = now,
        updated_at = now,
    )

    log.info(f"Trade opened: {trade_id} @ ${entry_price:.2f} [{setup_type}]")
    return trade_id


# ── update_trade ──────────────────────────────────────────────────────────────

def update_trade(trade_id: str,
                 new_score: float,
                 old_score: float,
                 reason: str,
                 material_event: Optional[dict] = None) -> None:
    """
    Appends UPDATE block to symbol ledger ONLY when:
      - score changed >= 5 pts, OR
      - material_event is not None
    """
    score_delta = abs(new_score - old_score)
    if score_delta < 5 and material_event is None:
        return

    symbol = trade_id.split("_")[0]
    path   = _ledger_path(symbol)
    if not os.path.exists(path):
        return

    event_line = ""
    if material_event:
        event_line = (
            f"  Event: {material_event.get('headline', '')[:80]}\n"
        )

    block = (
        f"{_SUB_SEP}\n"
        f"UPDATE  {_now_str()}\n"
        f"  Score: {old_score:.1f} → {new_score:.1f}  "
        f"({new_score - old_score:+.1f} pts)\n"
        f"  Reason: {reason}\n"
        f"{event_line}"
        f"{_SUB_SEP}\n"
    )
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception as e:
        log.debug(f"update_trade write failed: {e}")

    _db_write(trade_id, updated_at=datetime.utcnow(),
              full_context={"last_update": reason, "score": new_score})


# ── record_partial ────────────────────────────────────────────────────────────

def record_partial(trade_id: str,
                   pct_sold: float,
                   price: float,
                   pnl_pct: float) -> None:
    """Appends partial profit line to symbol ledger file."""
    symbol = trade_id.split("_")[0]
    path   = _ledger_path(symbol)

    line = (
        f"{_SUB_SEP}\n"
        f"🎯 PARTIAL  {_now_str()}\n"
        f"  Sold {pct_sold:.0%} @ ${price:.2f}  |  P&L: {pnl_pct:+.1f}%\n"
        f"{_SUB_SEP}\n"
    )
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        log.debug(f"record_partial write failed: {e}")

    _db_write(trade_id, updated_at=datetime.utcnow())


# ── close_trade ───────────────────────────────────────────────────────────────

def close_trade(trade_id: str,
                exit_price: float,
                exit_reason: str,
                exit_basket: dict,
                exit_news_trigger: Optional[str] = None) -> None:
    """
    Appends CLOSED block to symbol ledger.
    Appends one-line summary to trade_log.md.
    Updates DB record with exit fields.
    """
    symbol = trade_id.split("_")[0]
    path   = _ledger_path(symbol)
    now    = datetime.now()

    # Read entry data from DB to compute days held and return
    entry_price = 0.0
    entry_date  = now
    try:
        from database import get_db, TradeLedger
        db  = get_db()
        rec = db._get_session().__enter__().get(TradeLedger, trade_id)
        if rec:
            entry_price = float(rec.entry_price or 0)
            entry_date  = rec.entry_date if isinstance(
                rec.entry_date, datetime
            ) else now
    except Exception:
        pass

    days_held  = max(0, (now - entry_date).days)
    return_pct = (
        round((exit_price - entry_price) / entry_price * 100, 2)
        if entry_price > 0 else 0.0
    )

    exit_signals = exit_basket.get("signals_hit", [])
    basket_score = exit_basket.get("basket_score", 0)
    threshold    = exit_basket.get("threshold", 0)
    signals_block = "\n".join(
        f"  ✓ {s}" for s in (exit_signals[:6] if exit_signals else ["(hard exit)"])
    )
    news_line = (
        f"EXIT NEWS:    {exit_news_trigger[:100]}\n"
        if exit_news_trigger else ""
    )

    block = (
        f"\n{_SEPARATOR}\n"
        f"TRADE #{_get_trade_number(symbol) - 1} CLOSED\n"
        f"{_SEPARATOR}\n"
        f"DATE:         {now.strftime('%Y-%m-%d %H:%M EST')}\n"
        f"ACTION:       SELL\n"
        f"PRICE:        ${exit_price:.2f}\n"
        f"EXIT REASON:  {exit_reason}\n"
        f"EXIT SIGNALS (basket: {basket_score}/{threshold}):\n"
        f"{signals_block}\n"
        f"{news_line}"
        f"\n"
        f"DAYS HELD:    {days_held} days\n"
        f"RETURN:       {return_pct:+.2f}%\n"
        f"{_SUB_SEP}\n"
    )
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception as e:
        log.warning(f"close_trade file write failed: {e}")

    # ── Master log ────────────────────────────────────────────
    pnl_str    = f"P&L:{return_pct:+.1f}%"
    days_str   = f"{days_held}d"
    master_line = (
        f"{now.strftime('%Y-%m-%d %H:%M')}  SELL  {symbol:<10s}"
        f" @ ${exit_price:<8.2f}  {exit_reason:<18s}"
        f" {pnl_str}  {days_str}"
    )
    _append_master_log(master_line)

    # ── DB update ─────────────────────────────────────────────
    _db_write(
        trade_id,
        exit_date      = now,
        exit_price     = exit_price,
        exit_reason    = exit_reason,
        return_pct     = return_pct,
        days_held      = days_held,
        thesis_outcome = (
            "POSITIVE" if return_pct > 0 else "NEGATIVE"
        ),
        updated_at = now,
    )
    log.info(
        f"Trade closed: {trade_id} @ ${exit_price:.2f} "
        f"P&L={return_pct:+.1f}% [{exit_reason}]"
    )


# ── Class interface for run_server.py / PositionManager ──────────────────────

class TradeLedger:
    """Class wrapper around module-level trade_ledger functions."""

    def __init__(self, cfg: dict = None):
        self._cfg = cfg or {}

    def log_entry(self, symbol: str, qty=None, price=None, **kwargs):
        try:
            open_trade(symbol=symbol, qty=qty, price=price, **kwargs)
        except Exception as e:
            log.warning("TradeLedger.log_entry failed for %s: %s", symbol, e)

    def log_exit(self, symbol: str, qty=None, price=None, **kwargs):
        try:
            close_trade(symbol=symbol, exit_price=price or 0, **kwargs)
        except Exception as e:
            log.warning("TradeLedger.log_exit failed for %s: %s", symbol, e)
