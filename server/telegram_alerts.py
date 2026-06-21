# Block 1
# # telegram_alerts.py
# v3 — Telegram notification system
# Sends trade alerts, signal changes, and daily digests
# to your phone via Telegram bot

import os
import sys
import logging as _logging
import requests
from datetime import datetime
from dotenv import load_dotenv
from colorama import Fore, init

_tg_log = _logging.getLogger(__name__)

init(autoreset=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Core send function — all other methods use this.
    Returns True if sent successfully, False if failed.
    HTML formatting supported:
    <b>bold</b>  <i>italic</i>  <code>monospace</code>
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(Fore.YELLOW + "  Telegram not configured — skipping alert")
        return False
    try:
        url      = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": parse_mode,
        }, timeout=10)
        if response.status_code == 200:
            print(Fore.GREEN + "  Telegram alert sent")
            return True
        else:
            print(Fore.RED + f"  Telegram error: {response.status_code}")
            return False
    except Exception as e:
        print(Fore.RED + f"  Telegram failed: {e}")
        _tg_log.warning(
            f"Telegram alert failed: {e} — text: {text[:80]}"
        )
        return False

# Block 2
# What this does: seven alert functions each formatted for clear reading on a phone screen. alert_buy_signal fires when a buy signal generates with 
# the exact dollar amount to deploy and stop loss price. alert_sell_signal covers all exit types — hard stop, trailing stop, take profit, signal exit — 
# with a checkmark for profit and X for loss. alert_regime_change fires when the market shifts between bull/bear/sideways — one of the most important 
# alerts since it affects all position sizes. alert_fear_greed only fires at extremes (below 25 or above 75) — not on every check, only when sentiment 
# is historically significant. send_daily_digest sends a clean end-of-day summary to your phone. HTML formatting makes everything readable with bold headers 
# and monospace numbers.

def alert_buy_signal(symbol: str, price: float, rsi: float,
                     tranche: float, stop_loss: float,
                     reason: list = None) -> bool:
    """Fires when the strategy generates a BUY signal."""
    text = (
        f"<b>BUY SIGNAL</b>\n"
        f"<code>Symbol    : {symbol}</code>\n"
        f"<code>Price     : ${price:,.2f}</code>\n"
        f"<code>RSI       : {rsi:.1f}</code>\n"
        f"<code>Deploy    : ${tranche:,.2f}</code>\n"
        f"<code>Stop loss : ${stop_loss:,.2f}</code>\n"
        f"<code>Time      : {datetime.now().strftime('%H:%M:%S')}</code>"
    )
    if reason:
        text += f"\n\n<i>{reason[0]}</i>"
    return send_message(text)


def alert_trade_with_scorecard(symbol: str,
                                action: str,
                                price: float,
                                scorecard: dict,
                                stop_loss: float = None,
                                take_profit: float = None,
                                entry_type: str = None,
                                pnl_pct: float = None,
                                pnl_usd: float = None,
                                exit_type: str = None) -> bool:
    """
    Unified trade alert with full scorecard.
    Replaces alert_buy_signal and alert_sell_signal for v4.
    Every trade notification includes the complete scorecard
    so you can see exactly why the system acted.

    trade_id comes from the scorecard — TRD-XXXX format.
    """
    from strategy.scorer import StockScorer
    scorer = StockScorer()

    # format the full scorecard message
    scorecard_text = scorer.format_scorecard_telegram(
        scorecard   = scorecard,
        stop_loss   = stop_loss,
        take_profit = take_profit,
        entry_type  = entry_type,
    )

    # for SELL trades prepend P&L summary
    if action == "SELL" and pnl_pct is not None:
        emoji     = "✅" if pnl_usd >= 0 else "❌"
        exit_label = {
            "hard_stop":      "HARD STOP",
            "trailing_stop":  "TRAILING STOP",
            "take_profit":    "TAKE PROFIT",
            "signal_sell":    "SIGNAL EXIT",
            "partial_sell_recovery_possible": "PARTIAL EXIT",
            "full_sell_no_recovery":          "FULL EXIT",
        }.get(exit_type, exit_type.upper() if exit_type else "EXIT")

        pnl_line = (
            f"{emoji} <b>{exit_label}</b>\n"
            f"<code>P&L  : {pnl_pct:+.2f}% "
            f"(${pnl_usd:+.2f})</code>\n\n"
        )
        full_text = pnl_line + scorecard_text
    else:
        full_text = scorecard_text

    return send_message(full_text)


def alert_reentry_blocked(symbol: str,
                           reason: str,
                           trade_id: str,
                           cooldown_hours: float = None) -> bool:
    """
    Fires when re-entry is blocked by cooldown or price condition.
    Keeps you informed without spamming — only fires once per block.
    """
    text = (
        f"⏳ <b>RE-ENTRY BLOCKED — {symbol}</b>\n"
        f"<code>Reason   : {reason}</code>\n"
        f"<code>Last trade: {trade_id}</code>\n"
    )
    if cooldown_hours:
        text += (
            f"<code>Cooldown : {cooldown_hours:.1f}h remaining"
            f"</code>\n"
        )
    text += (
        f"<code>Time     : "
        f"{datetime.now().strftime('%H:%M:%S')}</code>"
    )
    return send_message(text)

def alert_sell_signal(symbol: str, price: float,
                       pnl_pct: float, pnl_usd: float,
                       exit_type: str, sell_pct: float = 1.0) -> bool:
    """Fires when the strategy generates a SELL or exit signal."""
    emoji     = "✅" if pnl_usd >= 0 else "❌"
    exit_label = {
        "hard_stop":    "HARD STOP",
        "trailing_stop": "TRAILING STOP",
        "take_profit":  "TAKE PROFIT",
        "signal_sell":  "SIGNAL EXIT",
        "partial_sell_recovery_possible": "PARTIAL EXIT",
        "full_sell_no_recovery": "FULL EXIT",
    }.get(exit_type, exit_type.upper())

    text = (
        f"{emoji} <b>{exit_label}</b>\n"
        f"<code>Symbol    : {symbol}</code>\n"
        f"<code>Price     : ${price:,.2f}</code>\n"
        f"<code>Sell      : {sell_pct*100:.0f}% of position</code>\n"
        f"<code>P&L       : {pnl_pct:+.2f}% (${pnl_usd:+.2f})</code>\n"
        f"<code>Time      : {datetime.now().strftime('%H:%M:%S')}</code>"
    )
    return send_message(text)


def alert_order_executed(symbol: str, side: str,
                          price: float, notional: float,
                          order_id: str) -> bool:
    """Fires when an Alpaca paper trade is actually executed."""
    side_label = "BOUGHT" if side == "buy" else "SOLD"
    text = (
        f"<b>ORDER EXECUTED — {side_label}</b>\n"
        f"<code>Symbol  : {symbol}</code>\n"
        f"<code>Price   : ${price:,.2f}</code>\n"
        f"<code>Value   : ${notional:,.2f}</code>\n"
        f"<code>Order ID: {order_id[:8]}...</code>\n"
        f"<code>Time    : {datetime.now().strftime('%H:%M:%S')}</code>"
    )
    return send_message(text)


def alert_regime_change(old_regime: str, new_regime: str,
                         spy_price: float, reason: str) -> bool:
    """Fires when market regime shifts — bull/bear/sideways."""
    emoji = {
        "bull":     "📈",
        "bear":     "📉",
        "sideways": "➡️",
    }.get(new_regime, "❓")

    text = (
        f"{emoji} <b>MARKET REGIME CHANGE</b>\n"
        f"<code>{old_regime.upper()} → {new_regime.upper()}</code>\n"
        f"<code>SPY price : ${spy_price:,.2f}</code>\n"
        f"<i>{reason}</i>\n"
        f"<code>Time      : {datetime.now().strftime('%H:%M:%S')}</code>"
    )
    return send_message(text)


def alert_fear_greed(score: int, label: str, signal: str) -> bool:
    """Fires when Fear & Greed crosses into extreme territory."""
    if score <= 24:
        emoji = "😱"
    elif score >= 75:
        emoji = "🤑"
    else:
        return False  # only alert on extremes

    text = (
        f"{emoji} <b>EXTREME MARKET SENTIMENT</b>\n"
        f"<code>Fear & Greed : {score}/100</code>\n"
        f"<code>Label        : {label.upper()}</code>\n"
        f"<code>Signal       : {signal.upper()}</code>\n"
        f"<i>Historically significant — review positions</i>\n"
        f"<code>Time         : {datetime.now().strftime('%H:%M:%S')}</code>"
    )
    return send_message(text)


def send_daily_digest(portfolio_value: float, total_pnl: float,
                       total_pnl_pct: float, signals: list,
                       top_movers: list = None) -> bool:
    """
    Sends a daily summary of portfolio and signals.
    Called by the scheduler at market close (4pm ET).
    """
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"
    signal_lines = ""

    for s in signals[:5]:
        label  = s.get("signal_label", "HOLD")
        sym    = s.get("symbol", "")
        price  = s.get("price", 0)
        rsi    = s.get("rsi", 0)
        from strategy.signal_types import SIGNAL_TYPE_EMOJI, SignalType
        try:
            badge = SIGNAL_TYPE_EMOJI.get(SignalType(label), "⚪")
        except Exception:
            badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(label, "⚪")
        signal_lines += (
            f"\n{badge} <code>{sym:8s} ${price:>10,.2f} RSI:{rsi:.0f}</code>"
        )

    text = (
        f"{pnl_emoji} <b>DAILY DIGEST</b>\n"
        f"<code>{datetime.now().strftime('%A, %d %b %Y')}</code>\n\n"
        f"<b>Portfolio</b>\n"
        f"<code>Value : ${portfolio_value:,.2f}</code>\n"
        f"<code>P&L   : {total_pnl_pct:+.2f}% (${total_pnl:+,.2f})</code>\n\n"
        f"<b>Signals</b>"
        f"{signal_lines}"
    )
    return send_message(text)


def alert_error(component: str, error: str) -> bool:
    """Fires when a critical system error occurs."""
    text = (
        f"⚠️ <b>SYSTEM ERROR</b>\n"
        f"<code>Component : {component}</code>\n"
        f"<code>Error     : {error[:200]}</code>\n"
        f"<code>Time      : {datetime.now().strftime('%H:%M:%S')}</code>"
    )
    return send_message(text)

# Block 3
if __name__ == "__main__":
    print(Fore.WHITE + "\n=== Testing Telegram Alerts ===\n")

    print("1. Testing buy signal alert...")
    alert_buy_signal(
        symbol    = "AAPL",
        price     = 260.48,
        rsi       = 32.5,
        tranche   = 594.00,
        stop_loss = 248.29,
        reason    = ["Bull market + Extreme fear — historically best buying conditions"]
    )

    print("2. Testing sell signal alert (profit)...")
    alert_sell_signal(
        symbol    = "AAPL",
        price     = 285.92,
        pnl_pct   = 20.13,
        pnl_usd   = 120.84,
        exit_type = "take_profit",
        sell_pct  = 0.5
    )

    print("3. Testing sell signal alert (loss)...")
    alert_sell_signal(
        symbol    = "BTC-USD",
        price     = 73000.00,
        pnl_pct   = -23.16,
        pnl_usd   = -2200.00,
        exit_type = "hard_stop",
        sell_pct  = 1.0
    )

    print("4. Testing regime change alert...")
    alert_regime_change(
        old_regime = "bull",
        new_regime = "bear",
        spy_price  = 520.00,
        reason     = "SPY broke below SMA200 — reducing all position sizes"
    )

    print("5. Testing fear & greed alert...")
    alert_fear_greed(score=15, label="extreme_fear", signal="strong_buy")

    print("6. Testing daily digest...")
    send_daily_digest(
        portfolio_value = 10482.00,
        total_pnl       = 482.00,
        total_pnl_pct   = 4.82,
        signals         = [
            {"symbol": "AAPL",    "signal_label": "HOLD", "price": 260.48, "rsi": 55.6},
            {"symbol": "MSFT",    "signal_label": "BUY",  "price": 370.87, "rsi": 38.4},
            {"symbol": "GOOGL",   "signal_label": "HOLD", "price": 317.24, "rsi": 61.5},
            {"symbol": "BTC-USD", "signal_label": "HOLD", "price": 71618,  "rsi": 55.8},
            {"symbol": "ETH-USD", "signal_label": "HOLD", "price": 2215,   "rsi": 56.1},
        ]
    )

    print("7. Testing error alert...")
    alert_error(
        component = "scheduler",
        error     = "yfinance connection timeout after 10 seconds"
    )

    print(Fore.GREEN + "\n  All alerts sent — check your Telegram!\n")