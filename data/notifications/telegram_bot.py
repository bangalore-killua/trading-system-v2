"""
telegram_bot.py — Interactive Telegram Bot (Human ↔ System)

The human's primary interface to the live trading system. No SSH needed.
Uses long-polling (python-telegram-bot library) — runs alongside the main loop.

Commands:
  /status          regime, VIX, cash %, NAV, positions count
  /positions       table: symbol | entry | current | return | days | signal
  /signals NVDA    all signal scores for a specific symbol
  /trace B7E2M4K9  full lifecycle of one trace (all stages, why fields)
  /why   B7E2M4K9  only the WHY fields across stages (quick narrative)
  /today           all activity logs for today
  /logs 2026-06-25 activity for a specific date
  /health          model IC table + calibration + PSI + rollback history
  /report          trigger a fresh weekly health report right now
  /audit NVDA      last 10 trades for symbol with outcomes

Security: all commands silently rejected if chat_id != TELEGRAM_CHAT_ID env var.

Requirements:
  pip install python-telegram-bot>=20.0

Usage (in main loop):
    from data.notifications.telegram_bot import start_bot
    asyncio.create_task(start_bot())
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_state_ref: Optional[dict] = None   # injected by main loop: {"portfolio", "regime", "vix", ...}


def inject_state(state: dict) -> None:
    """Called by the main loop to share live state with the bot."""
    global _state_ref
    _state_ref = state


async def start_bot() -> None:
    """Start the Telegram bot. Runs until cancelled."""
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info("Telegram bot not started — TELEGRAM_BOT_TOKEN/CHAT_ID not set")
        return

    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
    except ImportError:
        logger.warning("python-telegram-bot not installed — bot disabled. Run: pip install python-telegram-bot>=20.0")
        return

    allowed_chat_id = int(chat_id)

    def guard(update: Update) -> bool:
        return update.effective_chat and update.effective_chat.id == allowed_chat_id

    async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        s = _state_ref or {}
        port = s.get("portfolio", {})
        text = (
            f"<b>System Status</b>\n"
            f"Regime:     {s.get('regime', 'N/A')}\n"
            f"VIX:        {s.get('vix', 'N/A')}\n"
            f"Cash:       {port.get('cash_pct', 0):.1f}%\n"
            f"NAV:        ${port.get('total_value', 0):,.0f}\n"
            f"Positions:  {port.get('n_positions', 0)}\n"
            f"Mode:       {'PAPER' if s.get('paper_trading', True) else 'LIVE'}"
        )
        await update.message.reply_html(text)

    async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        s = _state_ref or {}
        positions = s.get("portfolio", {}).get("positions", [])
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        lines = ["<b>Open Positions</b>", ""]
        for p in positions[:15]:
            ret = p.get("unrealized_pct", 0)
            icon = "📈" if ret >= 0 else "📉"
            lines.append(
                f"{icon} <code>{p['symbol']:6s}</code> "
                f"${p.get('entry_price', 0):.2f}→${p.get('current_price', 0):.2f} "
                f"({ret:+.1f}%) | {p.get('days_held', 0)}d"
            )
        await update.message.reply_html("\n".join(lines))

    async def cmd_signals(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /signals NVDA")
            return
        symbol = args[0].upper()
        s = _state_ref or {}
        sigs = s.get("latest_signals", {}).get(symbol)
        if not sigs:
            await update.message.reply_text(f"No cached signals for {symbol}.")
            return
        lines = [f"<b>Signals: {symbol}</b>", ""]
        for k, v in sigs.items():
            if k.startswith("_"): continue
            bar = "█" * int(float(v or 0) * 10) if isinstance(v, (int, float)) else ""
            lines.append(f"{k:18s}: {float(v or 0):.2f} {bar}")
        await update.message.reply_html("\n".join(lines))

    async def cmd_trace(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /trace B7E2M4K9")
            return
        trace_id = args[0].upper()
        rows = _query_db_trace(trace_id)
        if not rows:
            await update.message.reply_text(f"No trace found: {trace_id}")
            return
        symbol = rows[0].get("symbol", "?")
        ts     = rows[0].get("ts", "")[:10]
        lines  = [f"<b>TRACE: {trace_id} — {symbol} — {ts}</b>", ""]
        for r in rows:
            icon  = "✅" if r.get("result") == "PASS" else ("⏭️" if r.get("result") == "SKIP" else "❌")
            stage = r.get("stage", "")
            why   = r.get("why", "")
            lines.append(f"{icon} <b>{stage:15s}</b>  {why[:80]}")
        await update.message.reply_html("\n".join(lines[:30]))

    async def cmd_why(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /why B7E2M4K9")
            return
        trace_id = args[0].upper()
        rows = _query_db_trace(trace_id)
        if not rows:
            await update.message.reply_text(f"No trace found: {trace_id}")
            return
        lines = [f"<b>WHY: {trace_id}</b>", ""]
        for r in rows:
            if r.get("why"):
                lines.append(f"<b>{r.get('stage','')}</b>: {r['why'][:120]}")
        await update.message.reply_html("\n".join(lines[:25]))

    async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        rows = _query_db_today()
        if not rows:
            await update.message.reply_text("No activity logged today.")
            return
        lines = [f"<b>Today's Activity ({len(rows)} entries)</b>", ""]
        for r in rows[:20]:
            ts    = r.get("ts", "")[-14:-9]
            stage = r.get("stage", "")
            sym   = r.get("symbol", "")
            res   = r.get("result", "")
            icon  = "✅" if res == "PASS" else ("⏭️" if res == "SKIP" else "❌")
            lines.append(f"{ts} {icon} <code>{sym:6s}</code> {stage}")
        await update.message.reply_html("\n".join(lines))

    async def cmd_logs(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        args = ctx.args
        date_str = args[0] if args else datetime.now().strftime("%Y-%m-%d")
        rows = _query_db_date(date_str)
        if not rows:
            await update.message.reply_text(f"No logs found for {date_str}.")
            return
        lines = [f"<b>Logs: {date_str} ({len(rows)} entries)</b>", ""]
        for r in rows[:20]:
            ts    = r.get("ts", "")[-14:-9]
            stage = r.get("stage", "")
            sym   = r.get("symbol", "")
            res   = r.get("result", "")
            icon  = "✅" if res == "PASS" else ("⏭️" if res == "SKIP" else "❌")
            lines.append(f"{ts} {icon} <code>{sym:6s}</code> {stage}")
        await update.message.reply_html("\n".join(lines))

    async def cmd_health(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        meta = _load_model_meta()
        lines = ["<b>Model Health</b>", ""]
        for m in ["entry", "exit", "dip", "tranche"]:
            ic    = meta.get(f"{m}_ic", None)
            brier = meta.get(f"{m}_brier", None)
            psi   = meta.get(f"{m}_psi", None)
            if ic is None:
                lines.append(f"{m:8s}: no data")
                continue
            ic_icon = "✅" if ic > 0 else "❌"
            lines.append(f"{m:8s}: IC={ic:.3f}{ic_icon}  Brier={brier:.3f}  PSI={psi:.3f}")
        await update.message.reply_html("\n".join(lines))

    async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        try:
            from data.model_health_reporter import ModelHealthReporter
            reporter = ModelHealthReporter()
            msg = await asyncio.get_event_loop().run_in_executor(None, reporter.build_report)
            await update.message.reply_html(msg[:4000])
        except Exception as e:
            await update.message.reply_text(f"Report generation failed: {e}")

    async def cmd_audit(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not guard(update): return
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /audit NVDA")
            return
        symbol = args[0].upper()
        rows = _query_db_outcomes(symbol)
        if not rows:
            await update.message.reply_text(f"No outcome data for {symbol}.")
            return
        lines = [f"<b>Audit: {symbol} (last {len(rows)} trades)</b>", ""]
        for r in rows[:10]:
            model_out = json.loads(r.get("model_out") or "{}")
            alpha = model_out.get("alpha", 0) or 0
            icon  = "✅" if alpha > 0 else "❌"
            lines.append(f"{icon} {r.get('ts','')[:10]}  alpha={alpha:+.1%}  {r.get('why','')[:60]}")
        await update.message.reply_html("\n".join(lines))

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("signals",   cmd_signals))
    app.add_handler(CommandHandler("trace",     cmd_trace))
    app.add_handler(CommandHandler("why",       cmd_why))
    app.add_handler(CommandHandler("today",     cmd_today))
    app.add_handler(CommandHandler("logs",      cmd_logs))
    app.add_handler(CommandHandler("health",    cmd_health))
    app.add_handler(CommandHandler("report",    cmd_report))
    app.add_handler(CommandHandler("audit",     cmd_audit))

    logger.info("Telegram bot started (long-polling)")
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await asyncio.Event().wait()  # run forever until cancelled
        await app.updater.stop()
        await app.stop()


# ── SQLite query helpers (read-only) ─────────────────────────────────────────

_DB_PATH = Path("logs/trade_lifecycle.db")


def _query_db_trace(trace_id: str) -> list:
    try:
        import sqlite3
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_logs WHERE trace_id=? ORDER BY id", (trace_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _query_db_today() -> list:
    try:
        import sqlite3
        cutoff = datetime.utcnow().strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_logs WHERE ts >= ? AND symbol != '__CYCLE__' ORDER BY id DESC LIMIT 100",
            (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _query_db_date(date_str: str) -> list:
    try:
        import sqlite3
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_logs WHERE ts LIKE ? AND symbol != '__CYCLE__' ORDER BY id DESC LIMIT 100",
            (f"{date_str}%",)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _query_db_outcomes(symbol: str) -> list:
    try:
        import sqlite3
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_logs WHERE symbol=? AND stage='OUTCOME' ORDER BY id DESC LIMIT 10",
            (symbol,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _load_model_meta() -> dict:
    try:
        return json.loads(Path("models/scorecard_meta.json").read_text())
    except Exception:
        return {}
