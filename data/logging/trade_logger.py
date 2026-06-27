"""
trade_logger.py — Structured Trade Lifecycle Logger

One log entry per major pipeline stage. All entries for one symbol evaluation
share a 10-char trace_id so any trade can be reconstructed end-to-end.

Two ID types:
  cycle_id  — system-level (DATA_INGEST → NLP_SCORING → UNIVERSE), symbol="__CYCLE__"
  trace_id  — per-symbol  (FACTOR_SCORES → ... → FEEDBACK)

Storage:
  Primary : logs/trade_lifecycle.db  (SQLite, WAL mode, O(1) lookup by trace_id)
  Backup  : logs/structured/trades_YYYYMMDD.jsonl  (one line per entry, why field readable)

Thread safety: all threads push TradeLog objects to a queue.Queue; one writer
thread drains to SQLite — no concurrent write conflicts.

Usage:
    cycle_id, log = new_cycle()
    log.result("PASS").signals({"sources_ok": 8}).why("all sources up").emit()

    trace_id, log = new_trace("NVDA", "FACTOR_SCORES")
    log.result("PASS").signals({"composite": 0.74}).why("composite=0.74 | insider=0.91").emit()

    log2 = stage(trace_id, "NVDA", "SIGNAL")
    log2.result("PASS").model({"composite_score": 0.74}).why("top: alternative=0.91").emit()

    rows = lookup_trace(trace_id)   # all stages for this symbol evaluation
    rows = query(result="FAIL", since_days=7)
"""

import json
import logging
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH   = Path("logs/trade_lifecycle.db")
JSONL_DIR = Path("logs/structured")

_DDL = """
CREATE TABLE IF NOT EXISTS trade_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL,
    cycle_id    TEXT,
    stage       TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    ts          TEXT NOT NULL,
    result      TEXT NOT NULL,
    reason      TEXT,
    signals     TEXT,
    model_out   TEXT,
    decision    TEXT,
    why         TEXT,
    error       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_stage  ON trade_logs(trace_id, stage);
CREATE INDEX       IF NOT EXISTS idx_trace         ON trade_logs(trace_id);
CREATE INDEX       IF NOT EXISTS idx_symbol_stage  ON trade_logs(symbol, stage, ts);
CREATE INDEX       IF NOT EXISTS idx_result_stage  ON trade_logs(result, stage);
CREATE INDEX       IF NOT EXISTS idx_ts            ON trade_logs(ts);
"""

_write_queue: queue.Queue = queue.Queue()
_writer_started = False
_writer_lock = threading.Lock()


def _ensure_writer() -> None:
    global _writer_started
    with _writer_lock:
        if _writer_started:
            return
        _writer_started = True
        t = threading.Thread(target=_writer_loop, daemon=True, name="trade-log-writer")
        t.start()


def _writer_loop() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL)
    conn.commit()

    while True:
        try:
            row: dict = _write_queue.get(timeout=5)
        except queue.Empty:
            continue

        try:
            conn.execute(
                """INSERT OR REPLACE INTO trade_logs
                   (trace_id,cycle_id,stage,symbol,ts,result,reason,signals,model_out,decision,why,error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["trace_id"], row.get("cycle_id"), row["stage"], row["symbol"],
                    row["ts"], row["result"], row.get("reason"),
                    json.dumps(row["signals"])   if row.get("signals")   else None,
                    json.dumps(row["model_out"]) if row.get("model_out") else None,
                    row.get("decision"), row.get("why"), row.get("error"),
                ),
            )
            conn.commit()
        except Exception:
            logger.exception("SQLite write failed for trace=%s stage=%s", row.get("trace_id"), row.get("stage"))

        # JSONL backup — always write even if SQLite fails
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            with open(JSONL_DIR / f"trades_{date_str}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            logger.debug("JSONL backup write failed")

        _write_queue.task_done()


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:10].upper()


class TradeLog:
    """Fluent builder for one log entry. Call .emit() to persist."""

    def __init__(self, trace_id: str, symbol: str, stage_name: str, cycle_id: Optional[str] = None):
        self._row: Dict[str, Any] = {
            "trace_id": trace_id,
            "cycle_id": cycle_id,
            "stage":    stage_name,
            "symbol":   symbol,
            "ts":       datetime.now(timezone.utc).isoformat(),
            "result":   "PASS",
        }

    def result(self, value: str) -> "TradeLog":
        self._row["result"] = value
        return self

    def reason(self, text: str) -> "TradeLog":
        self._row["reason"] = text
        return self

    def signals(self, data: dict) -> "TradeLog":
        self._row["signals"] = data
        return self

    def model(self, data: dict) -> "TradeLog":
        self._row["model_out"] = data
        return self

    def decision(self, text: str) -> "TradeLog":
        self._row["decision"] = text
        return self

    def why(self, text: str) -> "TradeLog":
        self._row["why"] = text
        return self

    def error(self, text: str) -> "TradeLog":
        self._row["result"] = "ERROR"
        self._row["error"]  = text
        return self

    def emit(self) -> str:
        _ensure_writer()
        _write_queue.put(dict(self._row))
        return self._row["trace_id"]


# ── Public factory functions ──────────────────────────────────────────────────

def new_cycle() -> tuple:
    """Generate a new cycle_id for system-level stages (DATA_INGEST, NLP, UNIVERSE)."""
    cycle_id = generate_trace_id()
    log = TradeLog(trace_id=cycle_id, symbol="__CYCLE__", stage_name="CYCLE_START", cycle_id=cycle_id)
    return cycle_id, log


def new_trace(symbol: str, stage_name: str, cycle_id: Optional[str] = None) -> tuple:
    """Generate a new trace_id for per-symbol evaluation."""
    trace_id = generate_trace_id()
    log = TradeLog(trace_id=trace_id, symbol=symbol, stage_name=stage_name, cycle_id=cycle_id)
    return trace_id, log


def stage(trace_id: str, symbol: str, stage_name: str, cycle_id: Optional[str] = None) -> TradeLog:
    """Continue an existing trace at a new stage."""
    return TradeLog(trace_id=trace_id, symbol=symbol, stage_name=stage_name, cycle_id=cycle_id)


def model_eval_log() -> TradeLog:
    """Standalone log for ML auto-tester (no trace chain)."""
    return TradeLog(trace_id=generate_trace_id(), symbol="__MODEL__", stage_name="MODEL_EVAL")


# ── Query functions ───────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def lookup_trace(trace_id: str) -> List[dict]:
    """Return all log entries for a trace_id, ordered by id."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM trade_logs WHERE trace_id=? ORDER BY id", (trace_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def query(
    symbol: Optional[str] = None,
    stage_name: Optional[str] = None,
    result: Optional[str] = None,
    since_days: int = 7,
) -> List[dict]:
    """Flexible filtered query across trade_logs."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    clauses = ["ts >= ?"]
    params: list = [cutoff]
    if symbol:
        clauses.append("symbol = ?"); params.append(symbol)
    if stage_name:
        clauses.append("stage = ?");  params.append(stage_name)
    if result:
        clauses.append("result = ?"); params.append(result)
    sql = "SELECT * FROM trade_logs WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 500"
    try:
        conn = _get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def recent_failures(since_days: int = 1) -> List[dict]:
    return query(result="FAIL", since_days=since_days) + query(result="ERROR", since_days=since_days)


# ── WhyAnalyzer — plain-English why fields per stage ─────────────────────────

class WhyAnalyzer:
    """Static methods that produce the 'why' string for each pipeline stage."""

    @staticmethod
    def data_ingest(sources_ok: list, sources_failed: list, articles: int) -> str:
        ok_str   = ",".join(sources_ok[:6])   + ("..." if len(sources_ok)   > 6 else "")
        fail_str = ",".join(sources_failed[:4]) + ("..." if len(sources_failed) > 4 else "")
        base = f"ok=[{ok_str}] | articles={articles}"
        return base + f" | FAILED=[{fail_str}]" if sources_failed else base

    @staticmethod
    def nlp_scoring(total: int, bullish_pct: float, bearish_pct: float,
                    threats: int, top_symbols: list) -> str:
        tops = ",".join(f"{s[0]}×{s[1]}" for s in top_symbols[:4])
        return (f"{total} articles: {bullish_pct:.0%} bull {bearish_pct:.0%} bear | "
                f"threats={threats} | top={tops}")

    @staticmethod
    def universe(l1: int, l2: int, l3: int, added: list, evicted: list) -> str:
        return (f"L1={l1} watchlist | L2={l2} hot | L3={l3} feedback | "
                f"added={len(added)} evicted={len(evicted)}")

    @staticmethod
    def factor_scores(composite: float, scores: dict, rank: int, total: int) -> str:
        top = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = " | ".join(f"{k}={v:.2f}" for k, v in top)
        return f"composite={composite:.2f} | {top_str} | rank #{rank}/{total}"

    @staticmethod
    def signal(composite: float, signal_scores: dict, consumed_cycle_id: str) -> str:
        top = sorted(signal_scores.items(), key=lambda x: -x[1])[:3]
        top_str = " | ".join(f"{k}={v:.2f}" for k, v in top)
        return f"composite={composite:.2f} | top: {top_str} | cycle={consumed_cycle_id}"

    @staticmethod
    def models(entry_p: float, exit_p: float, dip_p: Optional[float],
               tranche_p: Optional[float]) -> str:
        dip_str     = f"{dip_p:.2f}"     if dip_p     is not None else "N/A(RSI>40)"
        tranche_str = f"{tranche_p:.2f}" if tranche_p is not None else "N/A(no_pos)"
        return (f"entry_P={entry_p:.2f} | exit_P={exit_p:.2f}(hold) | "
                f"dip_P={dip_str} | tranche_P={tranche_str}")

    @staticmethod
    def constraints(passed: bool, gate_reason: Optional[str],
                    cash_pct: float, exposure_pct: float) -> str:
        if passed:
            return f"all gates clear | cash={cash_pct:.0%} | exposure={exposure_pct:.1f}%"
        return f"BLOCKED → {gate_reason} | cash={cash_pct:.0%} | exposure={exposure_pct:.1f}%"

    @staticmethod
    def entry(action: str, composite: float, size_pct: float,
              rsi: Optional[float], vix_scale: float) -> str:
        rsi_str = f"RSI={rsi:.0f}" if rsi is not None else "RSI=n/a"
        return f"{action} | composite={composite:.2f} | size={size_pct:.1f}% | {rsi_str} | vix_scale={vix_scale:.1f}"

    @staticmethod
    def execution(side: str, qty: int, symbol: str, price: float,
                  filled_price: float, attempts: int) -> str:
        slippage = abs(filled_price - price) / price * 100
        return (f"{side} {qty} {symbol} @ ${price:.2f} | "
                f"filled=${filled_price:.2f} | slippage={slippage:.2f}% | attempts={attempts}")

    @staticmethod
    def outcome(alpha: float, raw_return: float, days_held: int,
                exit_reason: str, entry_composite: float) -> str:
        verdict = "strong_win" if alpha > 0.05 else "win" if alpha > 0 else "loss" if alpha > -0.05 else "strong_loss"
        return (f"alpha={alpha:+.1%}({verdict}) | raw={raw_return:+.1%} | "
                f"days={days_held} | exit={exit_reason} | entry_composite={entry_composite:.2f}")

    @staticmethod
    def feedback(won: bool, top_signals: list, groq_lesson: str) -> str:
        verdict = "WON" if won else "LOST"
        tops = ",".join(top_signals[:3])
        return f"POST-MORTEM [{verdict}]: top_signals={tops} | {groq_lesson[:120]}"

    @staticmethod
    def model_eval(ic: float, brier: float, psi: float, gates_passed: int,
                   rollback: bool) -> str:
        status = "ROLLED_BACK" if rollback else "DEPLOYED"
        return (f"{status} | IC={ic:.3f} | brier={brier:.3f} | PSI={psi:.3f} | "
                f"gates_passed={gates_passed}/5")
