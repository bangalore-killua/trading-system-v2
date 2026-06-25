"""
telegram.py — Telegram bot notification client.

Sends operational alerts to a personal Telegram chat:
  - Data source degradation / fallback activation
  - Critical data gaps (source + all fallbacks down)
  - Circuit breaker triggers (VIX, drawdown, consecutive losses)
  - Trade executions (entry tranche, exit tranche, alpha)
  - Model retraining results
  - Daily health report (09:00)
  - System startup / shutdown

Setup (2 minutes, free forever):
  1. Open Telegram → search @BotFather → /newbot → follow prompts
  2. Save the bot token BotFather gives you
  3. Send any message to your new bot
  4. curl "https://api.telegram.org/bot{TOKEN}/getUpdates" → find "chat": {"id": ...}
  5. Add to .env:
       TELEGRAM_BOT_TOKEN=7391234567:AAF...
       TELEGRAM_CHAT_ID=123456789

Rate limiting:
  - SOURCE_DEGRADATION and ERROR alerts: max once per source per ALERT_COOLDOWN seconds
  - TRADE, CIRCUIT, SYSTEM alerts: always sent immediately (no cooldown)
"""

import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ALERT_COOLDOWN_SECONDS = 3600  # 1 hour between repeated source-degradation alerts

_last_notified: dict[str, datetime] = {}

EMOJI = {
    "INFO":    "ℹ️",
    "WARNING": "⚠️",
    "ERROR":   "🚨",
    "CRITICAL":"🔴",
    "CIRCUIT": "🛑",
    "TRADE":   "💹",
    "SYSTEM":  "🟢",
}


def notify(message: str, level: str = "INFO", source_key: Optional[str] = None) -> bool:
    """Send a Telegram message.

    Parameters
    ----------
    message    : HTML-formatted message body (supports <b>, <code>, <i>)
    level      : one of INFO, WARNING, ERROR, CRITICAL, CIRCUIT, TRADE, SYSTEM
    source_key : if set, enforces ALERT_COOLDOWN (use for repeated source alerts)

    Returns True if sent successfully, False otherwise.
    """
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping alert (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
        return False

    # Rate limiting for source-level alerts
    if source_key is not None:
        last = _last_notified.get(source_key)
        now  = datetime.now()
        if last and (now - last).total_seconds() < ALERT_COOLDOWN_SECONDS:
            logger.debug("Telegram: suppressing repeat alert for %s (cooldown)", source_key)
            return False
        _last_notified[source_key] = now

    emoji     = EMOJI.get(level, "ℹ️")
    timestamp = datetime.now().strftime("%H:%M")
    text      = f"{emoji} <b>[{level}]</b> {timestamp}\n\n{message}"

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
        if not resp.ok:
            logger.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
            return False
        return True
    except requests.exceptions.Timeout:
        logger.warning("Telegram send timed out")
        return False
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


# ── Convenience wrappers ──────────────────────────────────────────────────────

def notify_source_degraded(source_name: str, symbol: str, fallback_name: str) -> bool:
    return notify(
        f"<b>{source_name}</b> unavailable for <code>{symbol}</code>\n"
        f"Fallback: <b>{fallback_name}</b> — data may be less precise",
        level="WARNING",
        source_key=source_name,
    )


def notify_source_down(source_name: str, symbol: str, error: str) -> bool:
    return notify(
        f"<b>{source_name}</b> AND all fallbacks DOWN for <code>{symbol}</code>\n"
        f"Signal degraded — neutral default applied\n"
        f"<code>{error[:200]}</code>",
        level="ERROR",
        source_key=f"{source_name}_critical",
    )


def notify_circuit_breaker(reason: str, action: str) -> bool:
    return notify(
        f"<b>Reason:</b> {reason}\n<b>Action:</b> {action}",
        level="CIRCUIT",
    )


def notify_trade(action: str, symbol: str, price: float, details: dict) -> bool:
    detail_lines = "\n".join(f"  {k}: {v}" for k, v in details.items())
    return notify(
        f"<b>{action}:</b> <code>{symbol}</code> @ ${price:.2f}\n{detail_lines}",
        level="TRADE",
    )


def notify_model_retrained(model_name: str, ic: float, ic_delta: float, n_samples: int) -> bool:
    sign = "+" if ic_delta >= 0 else ""
    return notify(
        f"<b>{model_name}</b> retrained\n"
        f"IC: <b>{ic:.3f}</b> ({sign}{ic_delta:.3f} vs previous)\n"
        f"Samples: {n_samples:,}",
        level="INFO",
    )


def notify_system_start(n_positions: int, cash_pct: float, vix: Optional[float]) -> bool:
    vix_str = f"{vix:.1f}" if vix else "N/A"
    return notify(
        f"Trading system started\n"
        f"Open positions: {n_positions} | Cash: {cash_pct:.1f}%\n"
        f"VIX: {vix_str}",
        level="SYSTEM",
    )


def notify_daily_health(source_statuses: dict, portfolio_summary: dict) -> bool:
    status_lines = "\n".join(
        f"{'✅' if ok else '⚠️'} {name}"
        for name, ok in source_statuses.items()
    )
    p = portfolio_summary
    return notify(
        f"<b>Daily Status — {datetime.now().strftime('%Y-%m-%d')}</b>\n\n"
        f"<b>Sources:</b>\n{status_lines}\n\n"
        f"<b>Portfolio:</b> ${p.get('total_value', 0):,.0f} | Cash: {p.get('cash_pct', 0):.1f}%\n"
        f"<b>Positions:</b> {p.get('n_positions', 0)} | VIX: {p.get('vix', 'N/A')}\n"
        f"<b>Regime:</b> {p.get('regime', 'UNKNOWN')}",
        level="SYSTEM",
    )
