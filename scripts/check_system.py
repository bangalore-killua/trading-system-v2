"""
check_system.py — System Health Checker

Follows the same PASS/FAIL pattern as check_dip_labels.py.
Run this after any code change to verify the system is intact.

Checks:
  1. Core module imports (trade_logger, screener, signal_orchestrator, ...)
  2. New data sources (options_flow, finra_dark_pool)
  3. Required env vars (ALPACA_API_KEY, GROQ_API_KEY, optional TELEGRAM_*)
  4. Model files (AutoGluon entry, exit, dip, tranche)
  5. API connectivity (yfinance, FINRA ATS, optional FRED)
  6. SQLite trade_lifecycle.db schema (if it exists)
  7. Parallel runner adaptive worker count

Usage:
    python scripts/check_system.py
    python scripts/check_system.py --quick   # skip network checks
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence noisy library loggers
import logging
logging.basicConfig(level=logging.WARNING)


def section(name: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print('='*50)


def check(label: str, fn) -> bool:
    try:
        result = fn()
        if result is True or result is None:
            print(f"  [PASS] {label}")
            return True
        else:
            print(f"  [FAIL] {label}: {result}")
            return False
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        return False


def main(quick: bool = False) -> int:
    failures = 0

    # ── 1. Core imports ───────────────────────────────────────────────────────
    section("1. Core Module Imports")

    def _import(mod):
        __import__(mod)
        return True

    for mod in [
        "data.logging.trade_logger",
        "data.parallel_runner",
        "data.screener",
        "data.hot_universe",
        "data.news_aggregator",
        "data.model_tester",
        "data.model_health_reporter",
        "data.sources.options_flow",
        "data.sources.finra_dark_pool",
        "data.notifications.telegram",
        "data.notifications.telegram_bot",
        "strategy.signal_orchestrator",
        "strategy.position_manager",
        "backtester.system_backtest",
        "backtester.signal_audit",
    ]:
        if not check(mod, lambda m=mod: _import(m)):
            failures += 1

    # ── 2. Trade logger basic operation ──────────────────────────────────────
    section("2. Trade Logger (SQLite + JSONL)")

    def _logger_write_read():
        from data.logging.trade_logger import new_trace, lookup_trace
        trace_id, log = new_trace("__TEST__", "TEST_STAGE")
        log.result("PASS").why("system_check").emit()
        import time; time.sleep(0.2)  # let writer thread flush
        rows = lookup_trace(trace_id)
        assert len(rows) >= 1, f"expected 1 row, got {len(rows)}"
        return True

    if not check("write + read back trace", _logger_write_read):
        failures += 1

    def _jsonl_exists():
        from datetime import datetime
        path = Path("logs/structured") / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        assert path.exists(), f"JSONL not found: {path}"
        return True

    if not check("JSONL backup file created", _jsonl_exists):
        failures += 1

    # ── 3. Env vars ───────────────────────────────────────────────────────────
    section("3. Environment Variables")

    required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY"]
    optional_vars = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "FRED_API_KEY",
                     "ANTHROPIC_API_KEY", "ALPACA_BASE_URL"]

    for var in required_vars:
        if not check(f"{var} (required)", lambda v=var: os.getenv(v) is not None or f"NOT SET — required for trading"):
            failures += 1

    for var in optional_vars:
        val = os.getenv(var)
        print(f"  [INFO] {var}: {'SET' if val else 'not set (optional)'}")

    # ── 4. Model files ────────────────────────────────────────────────────────
    section("4. AutoGluon Model Files")

    model_paths = {
        "Entry model":   "models/autogluon_entry",
        "Exit model":    "models/autogluon_exit",
        "Dip model":     "models/autogluon_dip",
        "Tranche model": "models/autogluon_tranche",
    }
    for name, path in model_paths.items():
        p = Path(path)
        if p.exists():
            print(f"  [PASS] {name}: {path}")
        else:
            print(f"  [WARN] {name}: {path} not found (train with scripts/train_models.sh)")

    # ── 5. API connectivity ───────────────────────────────────────────────────
    if not quick:
        section("5. API Connectivity")

        def _yfinance():
            import yfinance as yf
            t = yf.Ticker("AAPL")
            info = t.info
            assert info.get("symbol") or info.get("shortName"), "empty yfinance response"
            return True

        if not check("yfinance (AAPL info)", _yfinance):
            failures += 1

        def _finra_dark_pool():
            from data.sources.finra_dark_pool import refresh_dark_pool_cache
            ok = refresh_dark_pool_cache(force=True)
            assert ok, "FINRA ATS API returned no data"
            return True

        if not check("FINRA ATS dark pool API", _finra_dark_pool):
            failures += 1

        def _options_flow():
            from data.sources.options_flow import get_options_score
            score, detail = get_options_score("AAPL")
            assert 0 <= score <= 1, f"options_score out of range: {score}"
            return True

        if not check("yfinance options_flow (AAPL)", _options_flow):
            failures += 1

    # ── 6. Parallel runner ────────────────────────────────────────────────────
    section("6. Parallel Runner")

    def _adaptive_workers():
        from data.parallel_runner import adaptive_workers
        w = adaptive_workers(100)
        cpu = os.cpu_count() or 2
        assert 1 <= w <= cpu, f"expected 1-{cpu}, got {w}"
        return True

    if not check(f"adaptive_workers (CPU={os.cpu_count()})", _adaptive_workers):
        failures += 1

    def _run_cpu_bound():
        from data.parallel_runner import run_cpu_bound
        results = run_cpu_bound(lambda x: x * 2, [1, 2, 3, 4])
        assert results == [2, 4, 6, 8], f"expected [2,4,6,8], got {results}"
        return True

    if not check("run_cpu_bound (4 items)", _run_cpu_bound):
        failures += 1

    # ── 7. SQLite schema (if DB exists) ──────────────────────────────────────
    section("7. SQLite Schema")

    db_path = Path("logs/trade_lifecycle.db")
    if db_path.exists():
        def _schema():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cols  = [r[1] for r in conn.execute("PRAGMA table_info(trade_logs)").fetchall()]
            required = ["trace_id", "stage", "symbol", "ts", "result", "why", "signals"]
            missing = [c for c in required if c not in cols]
            assert not missing, f"missing columns: {missing}"
            return True

        if not check("trade_logs schema", _schema):
            failures += 1
    else:
        print(f"  [INFO] {db_path} not yet created (will be created on first emit())")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if failures == 0:
        print("PASS - all checks passed")
    else:
        print(f"FAIL - {failures} check(s) failed")
    print('='*50)
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip network API checks")
    args = parser.parse_args()
    sys.exit(main(quick=args.quick))
