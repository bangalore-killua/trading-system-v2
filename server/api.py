# api.py
# v5.0 — FastAPI backend
# Serves live signals, portfolio data, trades and news
# as a JSON API consumed by the dashboard frontend
#
# Watchlist structure (v4.1):
#   config.json watchlist is a dict with three buckets:
#     screener_picks   — top 30 from weekly screen
#     manual_additions — user-added, never auto-removed
#     mandatory        — always present (BTC, ETH, SPY etc)

import os
import sys
import json
import threading
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor
from strategy.scorer import StockScorer
from strategy.trade_state import TradeStateManager

scorer      = StockScorer()
trade_state = TradeStateManager()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.json')


def load_config() -> dict:
    """Uses the 60-second TTL cache from config_loader — no disk read per request."""
    try:
        from strategy.config_loader import load_config as _cached_load
        result = _cached_load()
        if result:
            return result
    except Exception:
        pass
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "watchlist": {
                "screener_picks":   [],
                "manual_additions": ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"],
                "mandatory":        ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"],
            }
        }


def save_config(config: dict) -> bool:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data.fetcher import MarketDataFetcher
from strategy.technical_analysis import TechnicalAnalysis
from strategy.news_sentiment import NewsSentimentAnalyser

app = FastAPI(title="Trading System API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

fetcher   = MarketDataFetcher()
ta        = TechnicalAnalysis()
sentiment = NewsSentimentAnalyser()


def _get_combined_watchlist(config: dict = None) -> list:
    if config is None:
        config = load_config()
    wl = config.get("watchlist", {})
    if isinstance(wl, list):
        return wl if wl else ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
    picks  = wl.get("screener_picks",   [])
    manual = wl.get("manual_additions", [])
    mand   = wl.get("mandatory", ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"])
    seen     = set()
    combined = []
    for sym in picks + manual + mand:
        if sym not in seen:
            seen.add(sym)
            combined.append(sym)
    return combined if combined else ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]



# ── Health check ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":    "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version":   "5.0.0"
    }


# ── Prices ────────────────────────────────────────────────────

@app.get("/api/prices")
def get_prices():
    try:
        df = fetcher.get_multiple_prices(_get_combined_watchlist())
        return {"status": "ok", "data": df.to_dict(orient="records")}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Signals ───────────────────────────────────────────────────

@app.get("/api/signals")
def get_signals():
    cache_path = os.path.join(
        os.path.dirname(__file__), '..', 'logs', 'latest_signals.json'
    )

    # try database first
    try:
        from database import get_db
        db_signals = get_db().get_latest_signals()
        if db_signals:
            return {"status": "ok", "source": "database", "data": db_signals}
    except Exception:
        pass

    # fallback to JSON cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            age = (
                datetime.now() -
                datetime.fromtimestamp(os.path.getmtime(cache_path))
            ).seconds
            return {
                "status":    "ok",
                "source":    "cache",
                "cache_age": f"{age // 60}m {age % 60}s ago",
                "data":      cached,
            }
        except Exception:
            pass

    # no cache — calculate live
    current_watchlist = _get_combined_watchlist()

    def score_symbol(symbol):
        try:
            sc = scorer.score(symbol)
            return {
                "symbol":           symbol,
                "price":            sc.get("price", 0),
                "rsi":              sc.get("rsi", 0),
                "signal":           (1 if sc.get("signal") == "BUY" else
                                     -1 if sc.get("signal") == "SELL" else 0),
                "signal_label":     sc.get("signal_label",
                                           sc.get("signal", "HOLD")),
                "signal_type":      sc.get("signal_type", "HOLD"),
                "overall_display":  sc.get("overall_display", ""),
                "normalised_score": sc.get("normalised_score", 0),
                "score_velocity":   sc.get("score_velocity", 0),
                "conviction":       sc.get("conviction", ""),
                "bundle_name":      sc.get("bundle_name", ""),
                "bundle_emoji":     sc.get("bundle_emoji", ""),
                "asset_class":      sc.get("bundle_id", ""),
                "tier_id":          sc.get("tier_id", ""),
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e), "signal": 0,
                    "signal_label": "ERROR", "asset_class": "unknown"}

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(score_symbol, current_watchlist))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    return {"status": "ok", "source": "live", "data": results}


@app.get("/api/signals/refresh")
def refresh_signals():
    cache_path = os.path.join(
        os.path.dirname(__file__), '..', 'logs', 'latest_signals.json'
    )
    current_watchlist = _get_combined_watchlist()

    def score_symbol(symbol):
        try:
            sc = scorer.score(symbol)
            return {
                "symbol":           symbol,
                "price":            sc.get("price", 0),
                "signal":           (1 if sc.get("signal") == "BUY" else
                                     -1 if sc.get("signal") == "SELL" else 0),
                "signal_label":     sc.get("signal_label",
                                           sc.get("signal", "HOLD")),
                "signal_type":      sc.get("signal_type", "HOLD"),
                "overall_display":  sc.get("overall_display", ""),
                "normalised_score": sc.get("normalised_score", 0),
                "score_velocity":   sc.get("score_velocity", 0),
                "conviction":       sc.get("conviction", ""),
                "bundle_name":      sc.get("bundle_name", ""),
                "bundle_emoji":     sc.get("bundle_emoji", ""),
                "asset_class":      sc.get("bundle_id", ""),
                "tier_id":          sc.get("tier_id", ""),
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e), "signal": 0,
                    "signal_label": "ERROR", "asset_class": "unknown"}

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(score_symbol, current_watchlist))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    return {"status": "ok", "source": "live",
            "message": f"Refreshed {len(results)} symbols", "data": results}


# ── Sentiment / Sector / Insider ──────────────────────────────

@app.get("/api/sentiment/{symbol}")
def get_sentiment(symbol: str):
    try:
        company_names = {
            "AAPL": "Apple", "MSFT": "Microsoft",
            "GOOGL": "Google", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
        }
        company = company_names.get(symbol, symbol)
        result  = sentiment.analyse(symbol=symbol, company_name=company)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/fundamentals/{symbol}")
def get_fundamentals(symbol: str):
    try:
        data = fetcher.get_fundamentals(symbol)
        return {"status": "ok", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/sector/{symbol}")
def get_sector(symbol: str):
    try:
        sector = fetcher.get_sector_context(symbol)
        rs     = fetcher.get_relative_strength(symbol)
        return {"status": "ok", "data": {"sector_context": sector, "relative_strength": rs}}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/insider/{symbol}")
def get_insider(symbol: str):
    try:
        trades = fetcher.get_insider_trades(symbol)
        return {"status": "ok", "data": trades}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/analyst/{symbol}")
def get_analyst_ratings(symbol: str):
    """Returns recent analyst upgrades/downgrades for a symbol."""
    try:
        from data.news_aggregator import get_news_aggregator
        ratings = get_news_aggregator().scrape_finviz_analyst_ratings(
            symbol.upper().strip()
        )
        return {"status": "ok", "data": ratings}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/llm/status")
def get_llm_status():
    """Returns current LLM enabled state, mode, and whether API key is set."""
    try:
        from strategy.config_loader import load_config
        cfg     = load_config()
        enabled = bool(cfg.get("llm_enabled", False))
        mode    = cfg.get("llm_mode", "shadow")
        has_key = bool(os.getenv("ANTHROPIC_API_KEY", ""))
        return {
            "status": "ok",
            "data": {
                "enabled":  enabled,
                "mode":     mode,
                "has_key":  has_key,
                "ready":    enabled and has_key,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/llm/toggle")
async def toggle_llm(request: Request):
    """Enable or disable LLM analysis. Persists to config.json."""
    try:
        body    = await request.json()
        enabled = bool(body.get("enabled", False))
        cfg     = load_config()
        cfg["llm_enabled"] = enabled
        save_config(cfg)
        # Invalidate config_loader TTL cache so next cycle picks it up
        try:
            import strategy.config_loader as _cl
            _cl._config_cache["expires"] = 0.0
        except Exception:
            pass
        return {
            "status":  "ok",
            "enabled": enabled,
            "message": f"LLM analysis {'enabled' if enabled else 'disabled'}",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/llm/mode")
async def set_llm_mode(request: Request):
    """Set LLM mode: shadow | advisory | full. Persists to config.json."""
    try:
        body = await request.json()
        mode = body.get("mode", "shadow")
        if mode not in ("shadow", "advisory", "full"):
            return {
                "status":  "error",
                "message": "mode must be shadow | advisory | full",
            }
        cfg = load_config()
        cfg["llm_mode"] = mode
        save_config(cfg)
        try:
            import strategy.config_loader as _cl
            _cl._config_cache["expires"] = 0.0
        except Exception:
            pass
        return {
            "status": "ok",
            "mode":   mode,
            "message": f"LLM mode set to {mode}",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/market/breadth")
def get_market_breadth_endpoint():
    """V12-12: Market breadth indicators — futures, VIX term, sector rotation."""
    try:
        from data.fundamentals_fetcher import get_market_breadth
        return {"status": "ok", "data": get_market_breadth()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/fed-watch")
def get_fed_watch():
    """Returns CME FedWatch rate-move probability."""
    try:
        from data.news_aggregator import get_news_aggregator
        data = get_news_aggregator().scrape_cme_fedwatch()
        return {"status": "ok", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/economic-calendar")
def get_economic_calendar():
    """Returns upcoming US high-impact economic events."""
    try:
        from data.news_aggregator import get_news_aggregator
        data = get_news_aggregator().scrape_investing_calendar(
            days_ahead=7
        )
        return {"status": "ok", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Watchlist ─────────────────────────────────────────────────

@app.get("/api/watchlist")
def get_watchlist():
    config = load_config()
    return {"status": "ok", "data": _get_combined_watchlist(config)}


@app.get("/api/watchlist/buckets")
async def get_watchlist_buckets():
    try:
        from strategy.bundle_manager import get_bundle_manager
        bm     = get_bundle_manager()
        config = load_config()
        wl     = config.get("watchlist", {})

        if isinstance(wl, list):
            picks = []; manual = wl
            mandatory = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"]
        else:
            picks     = wl.get("screener_picks",   [])
            manual    = wl.get("manual_additions", [])
            mandatory = wl.get("mandatory", ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"])

        def enrich(symbols, bucket):
            result = []
            for sym in symbols:
                try:
                    info = bm.get_bundle_info(sym)
                    result.append({
                        "symbol": sym, "bucket": bucket,
                        "bundle_id": info["bundle_id"],
                        "bundle_name": info["bundle_name"],
                        "bundle_emoji": info["bundle_emoji"],
                    })
                except Exception:
                    result.append({
                        "symbol": sym, "bucket": bucket,
                        "bundle_id": "unknown", "bundle_name": "Unknown",
                        "bundle_emoji": "📌",
                    })
            return result

        return {
            "status": "ok",
            "data": {
                "screener_picks":   enrich(picks,     "screener"),
                "manual_additions": enrich(manual,    "manual"),
                "mandatory":        enrich(mandatory, "mandatory"),
                "totals": {
                    "screener_picks":   len(picks),
                    "manual_additions": len(manual),
                    "mandatory":        len(mandatory),
                    "combined":         len(set(picks + manual + mandatory)),
                },
                "screen_settings": config.get("screen_settings", {}),
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/watchlist/add")
def add_ticker(symbol: str):
    try:
        symbol  = symbol.upper().strip()
        test_df = fetcher.get_price_history(symbol, period="5d")
        if test_df.empty:
            return {"status": "error", "message": f"{symbol} not found"}
        config = load_config()
        wl     = config.get("watchlist", {})
        if isinstance(wl, list):
            config["watchlist"] = {
                "screener_picks": [], "manual_additions": wl,
                "mandatory": ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"],
            }
            wl = config["watchlist"]
        manual   = wl.setdefault("manual_additions", [])
        combined = _get_combined_watchlist(config)
        if symbol in combined:
            return {"status": "error", "message": f"{symbol} already in watchlist"}
        manual.append(symbol)
        save_config(config)
        return {"status": "ok", "message": f"{symbol} added to watchlist", "watchlist": _get_combined_watchlist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/watchlist/hygiene-preview")
def hygiene_preview():
    """V12-3: Preview symbols that would be removed on next hygiene run."""
    try:
        from database import get_db
        from strategy.trade_state import TradeStateManager
        TIER1_PROTECTED = {"BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD", "SLV"}
        open_syms   = set(TradeStateManager().get_symbols_to_monitor())
        config      = load_config()
        manual      = set(
            config.get("watchlist", {}).get("manual_additions", [])
        )
        candidates  = get_db().get_hygiene_candidates()
        eligible    = [
            c for c in candidates
            if c["symbol"] not in open_syms
            and c["symbol"] not in TIER1_PROTECTED
            and c["symbol"] not in manual
        ]
        return {"status": "ok", "data": eligible, "count": len(eligible)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/watchlist/restore/{symbol}")
def restore_archived_symbol(symbol: str):
    """V12-3: Move a symbol from watchlist_archived back to screener_picks."""
    try:
        symbol  = symbol.upper().strip()
        config  = load_config()
        archived = config.get("watchlist_archived", [])
        if symbol not in archived:
            return {"status": "error",
                    "message": f"{symbol} not in archived list"}
        archived.remove(symbol)
        picks = config.get("watchlist", {}).get("screener_picks", [])
        if symbol not in picks:
            picks.append(symbol)
        config["watchlist_archived"]                = archived
        config.setdefault("watchlist", {})["screener_picks"] = picks
        save_config(config)
        return {"status": "ok",
                "message": f"{symbol} restored to screener_picks"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/watchlist/remove")
def remove_ticker(symbol: str):
    try:
        symbol = symbol.upper().strip()
        config = load_config()
        wl     = config.get("watchlist", {})
        if isinstance(wl, list):
            if symbol not in wl:
                return {"status": "error", "message": f"{symbol} not in watchlist"}
            wl.remove(symbol)
            config["watchlist"] = wl
            save_config(config)
            return {"status": "ok", "watchlist": _get_combined_watchlist()}
        manual = wl.get("manual_additions", [])
        if symbol not in manual:
            if symbol in wl.get("screener_picks", []):
                return {"status": "error", "message": f"{symbol} is a screener pick — cannot remove manually"}
            if symbol in wl.get("mandatory", []):
                return {"status": "error", "message": f"{symbol} is mandatory and cannot be removed"}
            return {"status": "error", "message": f"{symbol} not in manual list"}
        manual.remove(symbol)
        save_config(config)
        return {"status": "ok", "message": f"{symbol} removed", "watchlist": _get_combined_watchlist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/watchlist/add-manual")
async def add_manual_symbol(symbol: str):
    try:
        from strategy.fundamental_screener import FundamentalScreener
        screener = FundamentalScreener()
        result   = screener.add_manual_symbol(symbol.upper().strip())
        # Score immediately so the symbol appears in signals without
        # waiting up to 30 min for the next scheduler cycle.
        sym = symbol.upper().strip()
        try:
            from data.fetcher import MarketDataFetcher
            try:
                price = MarketDataFetcher().get_current_price(sym)["price"]
            except Exception:
                price = 0

            scorecard = scorer.score(sym)
            signal_val = scorecard.get("signal", "HOLD")
            new_entry = {
                "symbol":           sym,
                "price":            round(price, 4),
                "signal":           signal_val,
                "label":            signal_val,
                "signal_label":     signal_val,
                "overall_display":  scorecard.get("overall_display", "?"),
                "normalised_score": scorecard.get("normalised_score", 0),
                "conviction":       scorecard.get("conviction", ""),
                "bundle_name":      scorecard.get("bundle_name", ""),
                "bundle_emoji":     scorecard.get("bundle_emoji", ""),
                "position_pct":     scorecard.get("position_pct", 0),
                "trade_id":         scorecard.get("trade_id", ""),
                "asset_class":      scorecard.get("bundle_id", ""),
                "cooldown":         False,
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Patch latest_signals.json — replace existing entry or append
            signals_path = os.path.join(
                os.path.dirname(__file__), '..', 'logs', 'latest_signals.json'
            )
            signals = []
            if os.path.exists(signals_path):
                try:
                    with open(signals_path) as f:
                        signals = json.load(f)
                    if not isinstance(signals, list):
                        signals = []
                except Exception:
                    signals = []

            signals = [s for s in signals if s.get("symbol") != sym]
            signals.append(new_entry)
            os.makedirs(os.path.dirname(signals_path), exist_ok=True)
            with open(signals_path, "w") as f:
                json.dump(signals, f, indent=2)

            # Persist to DB
            try:
                from database import get_db
                get_db().save_signals_batch([new_entry])
            except Exception:
                pass

        except Exception:
            pass  # scoring failure is non-fatal — symbol is still saved

        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/watchlist/remove-manual")
async def remove_manual_symbol(symbol: str):
    try:
        from strategy.fundamental_screener import FundamentalScreener
        screener = FundamentalScreener()
        result   = screener.remove_manual_symbol(symbol.upper().strip())
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Screen endpoints ──────────────────────────────────────────

@app.get("/api/screen/status")
async def get_screen_status():
    try:
        from strategy.fundamental_screener import FundamentalScreener
        screener = FundamentalScreener()
        results  = screener.get_last_results()
        config   = load_config()
        settings = config.get("screen_settings", {})
        wl       = config.get("watchlist", {})
        return {
            "status": "ok",
            "data": {
                "last_run":       settings.get("last_screened"),
                "screen_running": settings.get("screen_running", False),
                "screener_picks": len(wl.get("screener_picks", []) if isinstance(wl, dict) else []),
                "last_results":   results,
                "settings":       settings,
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/screen/run")
async def trigger_screen():
    try:
        config   = load_config()
        settings = config.setdefault("screen_settings", {})
        if settings.get("screen_running"):
            return {"status": "ok", "message": "Screen already running"}
        settings["screen_running"] = True
        save_config(config)

        def run_screen_thread():
            try:
                from strategy.fundamental_screener import FundamentalScreener
                screener = FundamentalScreener()
                screener.run_screen(save_results=True)
            except Exception as e:
                print(f"Screen thread error: {e}")
            finally:
                try:
                    cfg = load_config()
                    cfg.setdefault("screen_settings", {})["screen_running"] = False
                    save_config(cfg)
                except Exception:
                    pass

        threading.Thread(target=run_screen_thread, daemon=True).start()
        return {"status": "ok", "message": "Screen started in background."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Portfolio ─────────────────────────────────────────────────

portfolio = {"cash": 10000.0, "positions": {}, "trades": []}


@app.get("/api/portfolio")
def get_portfolio():
    try:
        from server.alpaca_trader import AlpacaTrader
        trader  = AlpacaTrader()
        account = trader.get_account()

        cash            = account.get("cash", 0.0)
        total_value     = account.get("portfolio_value", cash)
        buying_power    = account.get("buying_power", 0.0)
        from strategy.config_loader import get_settings as _gs
        initial_capital = float(_gs().get("initial_capital_usd", 100_000.0))

        # positions from Alpaca
        positions_data = []
        for p in trader.get_positions():
            cost_basis = p["qty"] * p["avg_entry"]
            pnl_pct    = round(float(p["unrealized_plpc"]) * 100, 2)
            positions_data.append({
                "symbol":        p["symbol"],
                "shares":        round(p["qty"], 6),
                "avg_price":     round(p["avg_entry"], 4),
                "current_price": round(p["current_price"], 4),
                "value":         round(p["market_value"], 2),
                "cost_basis":    round(cost_basis, 2),
                "pnl":           round(p["unrealized_pl"], 2),
                "pnl_pct":       pnl_pct,
            })

        # recent orders as trade history
        orders = trader.get_orders(limit=10)
        trades = [
            {
                "symbol":    o["symbol"],
                "action":    o["side"].upper(),
                "shares":    round(o["filled_qty"], 6),
                "price":     round(o["filled_price"], 4),
                "value":     round(o["filled_qty"] * o["filled_price"], 2),
                "status":    o["status"],
                "timestamp": o["created_at"],
            }
            for o in orders if o.get("filled_qty", 0) > 0
        ]

        total_pnl     = round(total_value - initial_capital, 2)
        total_pnl_pct = round((total_pnl / initial_capital) * 100, 2)

        return {
            "status": "ok",
            "source": "alpaca",
            "data": {
                "cash":          round(cash, 2),
                "buying_power":  round(buying_power, 2),
                "total_value":   round(total_value, 2),
                "total_pnl":     total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "positions":     positions_data,
                "trade_count":   len(trades),
                "trades":        trades,
            }
        }
    except Exception as e:
        # fallback to in-memory portfolio if Alpaca unavailable
        total_value    = portfolio["cash"]
        positions_data = []
        for symbol, position in portfolio["positions"].items():
            try:
                price      = fetcher.get_current_price(symbol)["price"]
                value      = position["shares"] * price
                cost_basis = position["shares"] * position["avg_price"]
                pnl        = value - cost_basis
                pnl_pct    = round((pnl / cost_basis) * 100, 2)
                total_value += value
                positions_data.append({
                    "symbol": symbol, "shares": round(position["shares"], 6),
                    "avg_price": round(position["avg_price"], 4),
                    "current_price": round(price, 4),
                    "value": round(value, 2), "pnl": round(pnl, 2), "pnl_pct": pnl_pct,
                })
            except Exception:
                positions_data.append({"symbol": symbol, "error": "price unavailable"})
        total_pnl     = round(total_value - 10000.0, 2)
        total_pnl_pct = round((total_pnl / 10000.0) * 100, 2)
        return {
            "status": "ok",
            "source": "fallback",
            "alpaca_error": str(e),
            "data": {
                "cash":          round(portfolio["cash"], 2),
                "total_value":   round(total_value, 2),
                "total_pnl":     total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "positions":     positions_data,
                "trade_count":   len(portfolio["trades"]),
                "trades":        portfolio["trades"][-10:],
            }
        }


@app.get("/api/portfolio/history")
def get_portfolio_history():
    try:
        from database import get_db
        curve = get_db().get_equity_curve(days=365)

        if curve:
            from strategy.config_loader import get_settings as _gs
            initial = float(_gs().get("initial_capital_usd", 100_000.0))
            labels  = [p["date"] for p in curve]
            values  = [round(p["value"], 2) for p in curve]
            return {
                "status": "ok",
                "source": "database",
                "data": {
                    "labels":          labels,
                    "values":          values,
                    "initial_capital": initial,
                    "final_value":     values[-1] if values else initial,
                    "return_pct":      round(
                        (values[-1] - initial) / initial * 100, 2
                    ) if values else 0,
                }
            }

        # no DB data yet — return empty chart with today's Alpaca value
        try:
            from server.alpaca_trader import AlpacaTrader
            account   = AlpacaTrader().get_account()
            today_val = account.get("portfolio_value", 100_000.0)
            today     = datetime.now().strftime("%Y-%m-%d")
            return {
                "status": "ok",
                "source": "alpaca_today",
                "data": {
                    "labels":          [today],
                    "values":          [round(today_val, 2)],
                    "initial_capital": 100_000.0,
                    "final_value":     round(today_val, 2),
                    "return_pct":      round(
                        (today_val - 100_000.0) / 100_000.0 * 100, 2
                    ),
                }
            }
        except Exception:
            return {
                "status": "ok",
                "source": "empty",
                "data": {
                    "labels": [], "values": [],
                    "initial_capital": 100_000.0,
                    "final_value": 100_000.0, "return_pct": 0,
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/portfolio/trade")
def execute_trade(symbol: str, action: str, shares: float):
    try:
        price  = fetcher.get_current_price(symbol)["price"]
        action = action.upper()
        pnl    = 0
        if action == "BUY":
            cost = shares * price
            if cost > portfolio["cash"]:
                return {"status": "error", "message": "Insufficient cash"}
            portfolio["cash"] -= cost
            if symbol in portfolio["positions"]:
                ex = portfolio["positions"][symbol]
                total_shares = ex["shares"] + shares
                total_cost   = (ex["shares"] * ex["avg_price"]) + cost
                portfolio["positions"][symbol] = {
                    "shares": total_shares, "avg_price": total_cost / total_shares,
                }
            else:
                portfolio["positions"][symbol] = {"shares": shares, "avg_price": price}
        elif action == "SELL":
            if symbol not in portfolio["positions"]:
                return {"status": "error", "message": f"No position in {symbol}"}
            position = portfolio["positions"][symbol]
            if shares > position["shares"]:
                return {"status": "error", "message": "Insufficient shares"}
            proceeds = shares * price
            portfolio["cash"] += proceeds
            pnl = proceeds - (shares * position["avg_price"])
            position["shares"] -= shares
            if position["shares"] <= 0:
                del portfolio["positions"][symbol]
        trade = {
            "symbol": symbol, "action": action, "shares": round(shares, 6),
            "price": round(price, 4), "value": round(shares * price, 2),
            "pnl": round(pnl, 2), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        portfolio["trades"].append(trade)
        return {"status": "ok", "data": trade}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Scorecard endpoints ───────────────────────────────────────

@app.get("/api/score/{symbol}")
def get_score(symbol: str):
    try:
        symbol = symbol.upper().strip()
        sc     = scorer.score(symbol)
        return {"status": "ok", "data": scorer.format_scorecard_dashboard(sc)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/backtest/{symbol}")
def run_clean_backtest(symbol: str,
                       period: str = "3y",
                       mode:   str = "single"):
    """
    Runs a clean point-in-time backtest — no lookahead bias.
    mode: "single" (in-sample) or "walk_forward" (out-of-sample)
    """
    try:
        symbol = symbol.upper().strip()
        from strategy.clean_backtester import CleanBacktester
        bt      = CleanBacktester(initial_capital=10_000.0)
        results = (
            bt.run_walk_forward(symbol)
            if mode == "walk_forward"
            else bt.run(symbol, period=period, verbose=False)
        )
        return {
            "status": "ok",
            "data": {
                "symbol":                results.get("symbol"),
                "period":                results.get("period"),
                "total_return_pct":      results.get("total_return_pct"),
                "win_rate":              results.get("win_rate"),
                "profit_factor":         results.get("profit_factor"),
                "max_drawdown":          results.get("max_drawdown"),
                "sharpe_ratio":          results.get("sharpe_ratio"),
                "total_trades":          results.get("total_trades"),
                "buy_hold_return":       results.get("buy_hold_return"),
                "outperformance":        results.get("outperformance"),
                "valid_for_performance": True,
                "fill_method":           results.get("fill_method"),
                "signal_method":         results.get("signal_method"),
                "windows":               results.get("windows", []),
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/scores")
def get_all_scores():
    try:
        config    = load_config()
        watchlist = _get_combined_watchlist(config)

        def score_symbol(sym):
            try:
                sc = scorer.score(sym)
                return scorer.format_scorecard_dashboard(sc)
            except Exception as e:
                return {"symbol": sym, "error": str(e), "signal": "ERROR"}

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(score_symbol, watchlist))
        return {"status": "ok", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Trade state endpoints ─────────────────────────────────────

@app.get("/api/trades/last")
def get_last_trades():
    try:
        summary = trade_state.get_summary()
        return {"status": "ok", "data": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/trades/last/{symbol}")
def get_last_trade(symbol: str):
    try:
        symbol = symbol.upper().strip()
        state  = trade_state.get_last_trade(symbol)
        if not state:
            return {"status": "ok", "data": None, "message": f"No trade history for {symbol}"}
        try:
            price   = fetcher.get_current_price(symbol)["price"]
            reentry = trade_state.can_reenter(symbol, price)
        except Exception:
            reentry = {"can_enter": None, "reason": "Price unavailable"}
        return {
            "status": "ok",
            "data": {**state, "reentry_gate": reentry,
                     "cooldown_hours": trade_state.get_cooldown_remaining(symbol)}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/trades/summary")
def get_trades_summary():
    try:
        all_trades       = trade_state.get_all_last_trades()
        in_cooldown      = [sym for sym in all_trades if trade_state.is_in_cooldown(sym)]
        active_positions = [sym for sym, state in all_trades.items()
                            if state.get("last_action") == "BUY"]
        return {
            "status": "ok",
            "data": {
                "total_tracked": len(all_trades),
                "in_cooldown": len(in_cooldown),
                "cooldown_symbols": in_cooldown,
                "active_positions": active_positions,
                "summary": trade_state.get_summary(),
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/trades/save")
def save_trade_state(symbol: str, action: str, price: float,
                     trade_id: str, exit_type: str = None, entry_price: float = None):
    try:
        symbol = symbol.upper().strip()
        record = trade_state.save_trade(
            symbol=symbol, action=action.upper(), price=price,
            trade_id=trade_id, exit_type=exit_type, entry_price=entry_price,
        )
        return {"status": "ok", "data": record}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/trades/cooldown/{symbol}")
def clear_cooldown(symbol: str):
    try:
        symbol = symbol.upper().strip()
        state  = trade_state.get_last_trade(symbol)
        if not state:
            return {"status": "error", "message": f"No trade state for {symbol}"}
        state["cooldown_until"] = None
        trade_state._state[symbol] = state
        trade_state._save()
        return {"status": "ok", "message": f"Cooldown cleared for {symbol}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Config ────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    config = load_config()
    return {"status": "ok", "data": config}


# ── Performance Analytics ─────────────────────────────────────

@app.get("/api/performance")
def get_performance():
    try:
        from database import get_db
        db      = get_db()
        summary = db.get_performance_summary()
        bundles = db.get_bundle_performance()
        return {"status": "ok", "data": {**summary, "bundles": bundles}}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Cache health (CACHE-1) ────────────────────────────────────

@app.get("/api/health/cache")
def get_cache_health():
    """Returns per-layer cache age and expiry status."""
    try:
        from data.cache_manager import get_cache
        import os
        cm  = get_cache()
        wl  = _get_combined_watchlist()[:5]
        result = {"global_layers": {}, "symbol_sample": {}}
        for layer in ["macro_regime", "intermarket", "breadth"]:
            result["global_layers"][layer] = {
                "age_hours": round(cm.get_age_hours(layer), 1),
                "expired":   cm.is_expired(layer),
            }
        for sym in wl:
            result["symbol_sample"][sym] = {}
            for layer in ["fundamental", "news", "insider"]:
                result["symbol_sample"][sym][layer] = {
                    "age_hours": round(cm.get_age_hours(layer, sym), 1),
                    "expired":   cm.is_expired(layer, sym),
                }
        cache_dir = os.path.join(
            os.path.dirname(__file__), '..', 'logs', 'cache'
        )
        result["total_cache_files"] = (
            len(os.listdir(cache_dir))
            if os.path.exists(cache_dir) else 0
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Scheduler health ──────────────────────────────────────────

@app.get("/api/health/websocket")
def get_websocket_health():
    """Returns Alpaca WebSocket connection status."""
    try:
        from data.price_stream import get_price_stream
        return {"status": "ok", "data": get_price_stream().is_connected()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/health/scheduler")
def get_scheduler_health():
    """
    Returns per-job health state populated by the _guarded()
    wrapper in scheduler.py. If the scheduler thread hasn't run
    a job yet, that job won't appear in the dict.
    """
    try:
        import time as _time
        from server.scheduler import _JOB_HEALTH, _last_full_cycle
        failed    = [
            name for name, h in _JOB_HEALTH.items()
            if h.get("last_status") == "error"
        ]
        cycle_ago = round(_time.monotonic() - _last_full_cycle, 1)
        return {
            "status": "ok",
            "data": {
                "jobs":                     _JOB_HEALTH,
                "failed_jobs":              failed,
                "healthy":                  len(failed) == 0,
                "last_full_cycle_ago_seconds": cycle_ago,
                "stale":                    cycle_ago > 2700,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Kill Switch ───────────────────────────────────────────────

@app.get("/api/kill-switch/status")
def get_kill_switch_status():
    try:
        from server.kill_switch import get_kill_switch_api, DailyLossGuard
        data = get_kill_switch_api()
        data["daily_loss"] = DailyLossGuard().get_status()
        return {"status": "ok", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/kill-switch/activate")
def activate_kill_switch(reason: str = "dashboard"):
    try:
        from server.kill_switch import activate_kill_switch_api
        return {"status": "ok", "data": activate_kill_switch_api(reason)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/kill-switch/deactivate")
def deactivate_kill_switch(reason: str = "dashboard"):
    try:
        from server.kill_switch import deactivate_kill_switch_api
        return {"status": "ok", "data": deactivate_kill_switch_api(reason)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/trading-mode")
def set_trading_mode(mode: str, confirm_code: str = None):
    try:
        from server.kill_switch import set_trading_mode_api
        return {"status": "ok", "data": set_trading_mode_api(mode, confirm_code)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Order Locks ───────────────────────────────────────────────

@app.get("/api/orders/locks")
def get_order_locks():
    try:
        from server.order_manager import OrderManager
        return {"status": "ok", "data": OrderManager().get_lock_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── BACKTEST-1: Three input modes ─────────────────────────────

@app.post("/api/backtest/run")
def backtest_run(body: dict):
    """
    MODE A: any symbol, returns single backtest result.
    Body: {"symbol": "TSLA", "period": "1y", "mode": "single"}
    """
    try:
        symbol = str(body.get("symbol", "")).upper().strip()
        period = str(body.get("period", "2y"))
        mode   = str(body.get("mode", "single"))
        if not symbol:
            return {"status": "error", "message": "symbol is required"}
        from strategy.clean_backtester import CleanBacktester
        bt     = CleanBacktester(initial_capital=10_000.0)
        result = bt.run_any_symbol(symbol, period=period, mode=mode)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/backtest/run_file")
def backtest_run_file(body: dict):
    """
    MODE B: batch from file.
    Body: {"symbols_file": "backtest_symbols.txt", "period": "2y"}
    """
    try:
        symbols_file = str(body.get("symbols_file", "backtest_symbols.txt"))
        period       = str(body.get("period", "2y"))
        from strategy.clean_backtester import CleanBacktester
        bt     = CleanBacktester(initial_capital=10_000.0)
        result = bt.run_from_file(symbols_file=symbols_file, period=period)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/backtest/scan")
def backtest_scan(body: dict):
    """
    MODE C: scan entire watchlist.
    Body: {"period": "6mo"}
    """
    try:
        period = str(body.get("period", "6mo"))
        from strategy.clean_backtester import CleanBacktester
        bt     = CleanBacktester(initial_capital=10_000.0)
        result = bt.run_scan(period=period)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── LEDGER-1: Trade decision ledger ───────────────────────────

@app.get("/api/ledger/open")
def ledger_open():
    """All open trades with entry context."""
    try:
        from database import get_db, TradeLedger
        db   = get_db()
        rows = db._get_session().__enter__().query(TradeLedger).filter(
            TradeLedger.exit_date.is_(None)
        ).order_by(TradeLedger.entry_date.desc()).all()
        return {"status": "ok", "data": [
            {
                "trade_id":    r.trade_id,
                "symbol":      r.symbol,
                "entry_date":  str(r.entry_date),
                "entry_price": r.entry_price,
                "setup_type":  r.setup_type,
                "regime":      r.regime_entry,
                "basket_score":r.basket_score_entry,
                "tier":        r.tier,
            }
            for r in rows
        ]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/ledger/closed")
def ledger_closed(days: int = 90):
    """Closed trades in descending time order."""
    try:
        from database import get_db, TradeLedger
        from datetime import timedelta
        db     = get_db()
        cutoff = datetime.now() - timedelta(days=days)
        rows   = db._get_session().__enter__().query(TradeLedger).filter(
            TradeLedger.exit_date >= cutoff
        ).order_by(TradeLedger.exit_date.desc()).all()
        return {"status": "ok", "data": [
            {
                "trade_id":   r.trade_id,
                "symbol":     r.symbol,
                "entry_date": str(r.entry_date),
                "exit_date":  str(r.exit_date),
                "entry_price":r.entry_price,
                "exit_price": r.exit_price,
                "return_pct": r.return_pct,
                "days_held":  r.days_held,
                "exit_reason":r.exit_reason,
                "setup_type": r.setup_type,
            }
            for r in rows
        ]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/ledger/trade/{trade_id}")
def ledger_by_trade_id(trade_id: str):
    """Fetch a single trade record by its unique trade_id."""
    try:
        from database import get_db, TradeLedger
        db  = get_db()
        row = db._get_session().__enter__().query(TradeLedger).filter(
            TradeLedger.trade_id == trade_id
        ).first()
        if not row:
            return {"status": "error", "message": f"Trade {trade_id} not found"}
        return {
            "status": "ok",
            "data": {
                "trade_id":     row.trade_id,
                "symbol":       row.symbol,
                "entry_date":   str(row.entry_date),
                "entry_price":  row.entry_price,
                "exit_date":    str(row.exit_date) if row.exit_date else None,
                "exit_price":   row.exit_price,
                "return_pct":   row.return_pct,
                "days_held":    row.days_held,
                "setup_type":   row.setup_type,
                "exit_reason":  row.exit_reason,
                "basket_score": row.basket_score_entry,
                "regime":       row.regime_entry,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/ledger/{symbol}")
def ledger_symbol(symbol: str):
    """All trades for one symbol (newest first)."""
    try:
        from database import get_db, TradeLedger
        db   = get_db()
        rows = db._get_session().__enter__().query(TradeLedger).filter(
            TradeLedger.symbol == symbol.upper()
        ).order_by(TradeLedger.entry_date.desc()).all()
        return {"status": "ok", "symbol": symbol.upper(), "data": [
            {
                "trade_id":    r.trade_id,
                "entry_date":  str(r.entry_date),
                "entry_price": r.entry_price,
                "exit_date":   str(r.exit_date) if r.exit_date else None,
                "exit_price":  r.exit_price,
                "return_pct":  r.return_pct,
                "days_held":   r.days_held,
                "setup_type":  r.setup_type,
                "exit_reason": r.exit_reason,
                "basket_score":r.basket_score_entry,
                "regime":      r.regime_entry,
            }
            for r in rows
        ]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/ledger/summary")
def ledger_summary():
    """Performance stats broken down by setup_type."""
    try:
        from database import get_db, TradeLedger
        db   = get_db()
        rows = db._get_session().__enter__().query(TradeLedger).filter(
            TradeLedger.exit_date.isnot(None)
        ).all()
        by_setup: dict = {}
        for r in rows:
            st = r.setup_type or "UNKNOWN"
            if st not in by_setup:
                by_setup[st] = {"trades": 0, "wins": 0, "total_return": 0.0}
            by_setup[st]["trades"]       += 1
            by_setup[st]["total_return"] += r.return_pct or 0.0
            if (r.return_pct or 0) > 0:
                by_setup[st]["wins"] += 1
        summary = {}
        for st, d in by_setup.items():
            n = d["trades"]
            summary[st] = {
                "trades":     n,
                "win_rate":   round(d["wins"] / n * 100, 1) if n else 0,
                "avg_return": round(d["total_return"] / n, 2) if n else 0,
            }
        total = len(rows)
        wins  = sum(1 for r in rows if (r.return_pct or 0) > 0)
        return {
            "status":       "ok",
            "total_trades": total,
            "win_rate":     round(wins / total * 100, 1) if total else 0,
            "by_setup":     summary,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── PERF-1: Performance reporting ─────────────────────────────

@app.get("/api/performance/summary")
def performance_summary(days: int = 30):
    """Full performance stats for the last N days."""
    try:
        from strategy.performance_calculator import calculate_performance
        return {"status": "ok", "data": calculate_performance(days)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/performance/by_setup")
def performance_by_setup(days: int = 30):
    """Performance breakdown by setup type."""
    try:
        from strategy.performance_calculator import calculate_performance
        stats = calculate_performance(days)
        return {
            "status":       "ok",
            "days":         days,
            "by_setup_type":stats["by_setup_type"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/performance/by_symbol")
def performance_by_symbol(days: int = 30):
    """Return % per closed trade grouped by symbol."""
    try:
        from database import get_db, TradeLedger
        from datetime import timedelta
        db     = get_db()
        cutoff = datetime.now() - timedelta(days=days)
        rows   = db._get_session().__enter__().query(TradeLedger).filter(
            TradeLedger.exit_date >= cutoff,
            TradeLedger.exit_date.isnot(None),
        ).all()
        by_sym: dict = {}
        for r in rows:
            s = r.symbol
            if s not in by_sym:
                by_sym[s] = {"trades": 0, "total_return": 0.0, "wins": 0}
            by_sym[s]["trades"]       += 1
            by_sym[s]["total_return"] += float(r.return_pct or 0)
            if (r.return_pct or 0) > 0:
                by_sym[s]["wins"] += 1
        summary = {
            s: {
                "trades":     d["trades"],
                "avg_return": round(d["total_return"] / d["trades"], 2),
                "win_rate":   round(d["wins"] / d["trades"] * 100, 1),
            }
            for s, d in by_sym.items()
        }
        return {"status": "ok", "days": days, "by_symbol": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/performance/daily")
def performance_daily(days: int = 30):
    """Stored PerformanceSummary snapshots for the last N days."""
    try:
        from database import get_db, PerformanceSummary
        from datetime import timedelta
        db     = get_db()
        cutoff = datetime.now() - timedelta(days=days)
        rows   = db._get_session().__enter__().query(PerformanceSummary).filter(
            PerformanceSummary.period_end >= cutoff
        ).order_by(PerformanceSummary.period_end.desc()).all()
        return {"status": "ok", "data": [
            {
                "period_end":      str(r.period_end)[:10],
                "total_trades":    r.total_trades,
                "win_rate":        r.win_rate,
                "avg_return_pct":  r.avg_return_pct,
                "sharpe_ratio":    r.sharpe_ratio,
                "max_drawdown_pct":r.max_drawdown_pct,
            }
            for r in rows
        ]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── News feed (fincept layer) ─────────────────────────────────

@app.get("/api/news")
def get_news(hours: int = 24, ticker: str = "", limit: int = 60):
    """Recent articles — existing 39 RSS sources + PRNewswire + BusinessWire, with FinBERT sentiment.

    v13: adds PRNewswire and BusinessWire (credibility=1.0, 22-minute alpha advantage).
    If ticker is specified, also fetches Google News for that symbol.
    Deduplicates by URL; PRNewswire/BusinessWire articles kept over duplicates from Google News.
    """
    try:
        from data.news_aggregator import get_news_aggregator
        agg     = get_news_aggregator()
        tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()] if ticker else None
        articles = agg.get_recent_articles(hours=hours, tickers=tickers)[:limit]
        now  = datetime.utcnow()
        data = []
        seen_urls: set = set()

        # Primary: existing 39 RSS sources
        for a in articles:
            if a.url in seen_urls:
                continue
            seen_urls.add(a.url)
            age_s = int((now - a.published_at).total_seconds())
            age_label = (
                f"{age_s // 3600}h ago" if age_s >= 3600
                else f"{age_s // 60}m ago" if age_s >= 60
                else "just now"
            )
            data.append({
                "title":        a.title,
                "url":          a.url,
                "source":       a.source_name,
                "published_at": a.published_at.isoformat(),
                "age_label":    age_label,
                "sentiment":    round(a.sentiment, 3),
                "threat":       a.threat,
                "tickers":      a.tickers,
                "credibility":  1.0,
            })

        # v13: PRNewswire + BusinessWire (credibility=1.0)
        try:
            from data.sources.prnewswire import get_prnewswire_for_symbol, get_prnewswire
            from data.sources.businesswire import get_businesswire_for_symbol, get_businesswire
            if tickers:
                for sym in tickers[:3]:
                    pr_articles = get_prnewswire_for_symbol(sym, limit=5)
                    bw_articles = get_businesswire_for_symbol(sym, limit=5)
                    for a in pr_articles + bw_articles:
                        if a.get("url") not in seen_urls:
                            seen_urls.add(a["url"])
                            data.insert(0, {  # prepend — highest priority
                                "title":        a.get("title", ""),
                                "url":          a.get("url", ""),
                                "source":       a.get("source", "prnewswire"),
                                "published_at": a.get("published", now.isoformat()),
                                "age_label":    "recent",
                                "sentiment":    0.0,
                                "threat":       False,
                                "tickers":      [sym],
                                "credibility":  a.get("credibility", 1.0),
                            })
            else:
                pr_general = get_prnewswire(limit=10)
                bw_general = get_businesswire(limit=10)
                for a in pr_general + bw_general:
                    if a.get("url") not in seen_urls:
                        seen_urls.add(a["url"])
                        data.insert(0, {
                            "title":        a.get("title", ""),
                            "url":          a.get("url", ""),
                            "source":       a.get("source", "prnewswire"),
                            "published_at": a.get("published", now.isoformat()),
                            "age_label":    "recent",
                            "sentiment":    0.0,
                            "threat":       False,
                            "tickers":      [],
                            "credibility":  a.get("credibility", 1.0),
                        })
        except Exception:
            pass  # PR/BW fetch failure is non-fatal

        # v13: Google News for ticker-specific requests (credibility=0.60)
        if tickers:
            try:
                from data.fincept.google_news import get_google_news
                for sym in tickers[:2]:
                    gn_articles = get_google_news(sym, limit=8)
                    for a in gn_articles:
                        if a.get("url") not in seen_urls:
                            seen_urls.add(a["url"])
                            data.append({
                                "title":        a.get("title", ""),
                                "url":          a.get("url", ""),
                                "source":       "google_news",
                                "published_at": a.get("published", now.isoformat()),
                                "age_label":    "recent",
                                "sentiment":    0.0,
                                "threat":       False,
                                "tickers":      [sym],
                                "credibility":  0.60,
                            })
            except Exception:
                pass

        return {"status": "ok", "count": len(data), "data": data[:limit]}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@app.get("/api/regime")
def get_regime():
    """Current market regime + macro snapshot."""
    try:
        reg = regime_detector.get_current_regime() if regime_detector else None
        return {
            "status": "ok",
            "data": {
                "regime":       reg.name if reg else "UNKNOWN",
                "regime_label": str(reg) if reg else "Unknown",
                "allowed_entry": reg is not None and reg.name in [
                    "EARLY_BULL", "MID_BULL", "NEUTRAL"
                ],
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e),
                "data": {"regime": "UNKNOWN", "allowed_entry": False}}


# ── News Intelligence ─────────────────────────────────────────

@app.get("/api/news/analysis")
async def get_news_analysis(tickers: str = "", hours: int = 24, limit: int = 30):
    """Cross-article correlation + Groq portfolio-aware impact analysis.

    Query params:
      tickers — comma-separated tickers to treat as portfolio (e.g. AAPL,MSFT,NVDA)
      hours   — look back this many hours for recent articles (default 24)
      limit   — max articles to analyse (default 30)

    Returns: {clusters, ticker_map, impact, cross_correlations, summary}
    """
    try:
        from data.news_aggregator import get_news_aggregator
        from data.news_intelligence import get_news_intelligence_engine

        agg = get_news_aggregator()
        portfolio = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        raw_articles = agg.get_recent_articles(hours=hours)[:limit]

        articles = [
            {
                "title":         a.title,
                "summary":       getattr(a, "summary", a.title),
                "url":           a.url,
                "finbert_score": a.sentiment,
                "tickers":       a.tickers,
            }
            for a in raw_articles
        ]

        engine = get_news_intelligence_engine()
        result = await engine.analyse(articles, portfolio_holdings=portfolio)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


# ── Knowledge Graph ────────────────────────────────────────────

@app.get("/api/graph/{symbol}")
def get_knowledge_graph(symbol: str, hops: int = 1):
    """Return Cytoscape.js-compatible knowledge graph for a symbol.

    Query params:
      symbol — the anchor ticker (e.g. NVDA)
      hops   — how many relationship hops to expand (default 1, max 2)

    Returns: {nodes: [{data:{id,label}}], edges: [{data:{source,target,rel_type}}]}
    """
    try:
        from data.knowledge_graph import get_knowledge_graph as get_kg
        kg = get_kg()
        graph_json = kg.get_graph_json(symbol.upper(), hops=min(hops, 2))
        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "node_count": len(graph_json["nodes"]),
            "edge_count": len(graph_json["edges"]),
            "data": graph_json,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {"nodes": [], "edges": []}}


# ── Fincept Data Endpoints ────────────────────────────────────

@app.get("/api/macro/snapshot")
def get_macro_snapshot():
    """FRED macro snapshot: GDP, CPI, UNRATE, DFF, 10Y yield."""
    try:
        from data.fincept.fred_data import get_macro_snapshot
        return {"status": "ok", "data": get_macro_snapshot()}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/macro/fred/{series_id}")
def get_fred_series(series_id: str, limit: int = 24):
    """Arbitrary FRED series — last N observations."""
    try:
        from data.fincept.fred_data import get_series
        data = get_series(series_id.upper(), limit=limit)
        return {"status": "ok", "series_id": series_id.upper(), "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@app.get("/api/global/indices")
def get_global_indices():
    """Global equity indices + ETF quotes (SPY, QQQ, DIA, GLD, NQ, DAX, Nikkei)."""
    try:
        from data.fincept.yfinance_data import get_batch_quotes
        symbols = ["SPY", "QQQ", "DIA", "GLD", "^NQ=F", "^GDAXI", "^N225", "^VIX", "^TNX"]
        data = get_batch_quotes(symbols)
        return {"status": "ok", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/options/{symbol}")
def get_options(symbol: str, expiry: str = ""):
    """Options chain + estimated Greeks for a symbol."""
    try:
        from data.fincept.cboe_data import get_options_chain, get_iv_skew
        chain = get_options_chain(symbol.upper(), expiry or None)
        skew  = get_iv_skew(symbol.upper(), expiry or None)
        return {"status": "ok", "data": {**chain, "iv_skew": skew}}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/cot")
def get_cot():
    """COT report snapshot for key futures contracts."""
    try:
        from data.fincept.cftc_data import get_cot_snapshot
        return {"status": "ok", "data": get_cot_snapshot()}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@app.get("/api/sec/filings/{symbol}")
def get_sec_filings(symbol: str, forms: str = "10-K,10-Q,8-K,4", limit: int = 10):
    """Recent SEC filings + insider Form 4 trades for a symbol."""
    try:
        from data.fincept.sec_data import get_filings, get_insider_trading
        filings = get_filings(symbol.upper(), forms=forms, limit=limit)
        insider = get_insider_trading(symbol.upper(), days_back=90)
        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "data": {"filings": filings, "insider_trades": insider},
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/patterns/{symbol}")
def get_chart_patterns(symbol: str, period: str = "3mo"):
    """YOLOv8 detected chart patterns for a symbol.

    Returns list of {pattern, confidence, direction, implication, bbox}.
    Returns empty list if model not loaded or chart unavailable.
    """
    try:
        from strategy.pattern_api import detect_patterns
        patterns = detect_patterns(symbol.upper(), period=period)
        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "period": period,
            "data":   patterns,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@app.get("/api/ai/agent/{agent_name}")
async def run_ai_agent(agent_name: str):
    """Run a Fincept AI agent (portfolio_manager / risk_quant / signal_scientist).

    Passes current portfolio state + macro snapshot + top signals as context.
    Returns markdown-formatted report from Groq Llama 3.3 70B.
    """
    try:
        from data.fincept.agents.base import get_agent_factory
        from data.fincept.fred_data import get_macro_snapshot

        agents = get_agent_factory()
        if agent_name not in agents:
            return {
                "status": "error",
                "message": f"Unknown agent '{agent_name}'. Available: {list(agents.keys())}",
            }

        # Build context from current system state
        context: dict = {"regime": "NEUTRAL", "portfolio": [], "top_signals": [], "macro": {}}
        try:
            context["macro"] = get_macro_snapshot()
        except Exception:
            pass
        try:
            context["portfolio"] = [
                p for p in (db.get_positions() if db else [])
            ]
        except Exception:
            pass
        try:
            context["top_signals"] = [
                s for s in (db.get_latest_signals()[:15] if db else [])
            ]
        except Exception:
            pass
        try:
            if regime_detector:
                reg = regime_detector.get_current_regime()
                context["regime"] = reg.name if reg else "NEUTRAL"
        except Exception:
            pass

        # Run agent (synchronous Groq call, wrapped in executor)
        import asyncio
        loop = asyncio.get_event_loop()
        agent = agents[agent_name]
        report = await loop.run_in_executor(None, lambda: agent.run(context))
        return {"status": "ok", "agent": agent_name, "report": report}
    except Exception as e:
        return {"status": "error", "message": str(e), "report": ""}


@app.get("/api/ohlcv/{symbol}")
def get_ohlcv(symbol: str, period: str = "3mo"):
    """OHLCV price history in Lightweight Charts format (time, open, high, low, close)."""
    try:
        from data.fincept.yfinance_data import get_ohlcv
        df = get_ohlcv(symbol.upper(), period=period)
        if df is None or df.empty:
            return {"status": "ok", "data": []}
        bars = []
        for ts, row in df.iterrows():
            # Lightweight Charts expects Unix timestamp seconds
            bars.append({
                "time":  int(ts.timestamp()),
                "open":  round(float(row["Open"]), 4),
                "high":  round(float(row["High"]), 4),
                "low":   round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
            })
        return {"status": "ok", "symbol": symbol.upper(), "data": bars}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}


@app.get("/api/stock/{symbol}")
def get_stock_detail(symbol: str):
    """Full stock detail: quote + fundamentals + recent news."""
    try:
        from data.fincept.yfinance_data import get_quote, get_info, get_news
        quote = get_quote(symbol.upper())
        info  = get_info(symbol.upper())
        news  = get_news(symbol.upper(), max_items=5)
        return {
            "status": "ok",
            "symbol": symbol.upper(),
            "data": {"quote": quote, "fundamentals": info, "news": news},
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


# ── v13: Analysis / Earnings / Financials / Holdings tabs ─────

@app.get("/api/analysis/{symbol}")
def get_symbol_analysis(symbol: str):
    """Analyst consensus, price targets, and Claude Haiku bear/base/bull scenarios.

    Returns
    -------
    {
      "recommendations": [...], "price_target": {...},
      "recommendation_mean": float, "recommendation_key": str,
      "scenarios": {"bear": str, "base": str, "bull": str}
    }
    """
    try:
        symbol = symbol.upper().strip()
        from data.fincept.yfinance_data import get_analysis
        data   = get_analysis(symbol)

        # Claude Haiku 3/6/9/12-month scenarios (non-blocking — returns empty if unavailable)
        scenarios = {}
        try:
            import os
            key = os.getenv("ANTHROPIC_API_KEY")
            if key:
                import anthropic  # type: ignore
                client = anthropic.Anthropic(api_key=key)
                pt     = data.get("price_target", {})
                recs   = data.get("recommendations", [])[:5]
                prompt = (
                    f"Stock: {symbol}\n"
                    f"Analyst mean PT: {pt.get('mean')} | High: {pt.get('high')} | Low: {pt.get('low')}\n"
                    f"Recent ratings: {[(r['firm'], r['to_grade']) for r in recs]}\n\n"
                    "Give 3–12 month bear/base/bull price scenarios in one sentence each. "
                    "Format: BEAR: ... BASE: ... BULL: ..."
                )
                msg   = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text
                for label in ("BEAR", "BASE", "BULL"):
                    import re
                    m = re.search(rf"{label}:\s*(.+?)(?=(?:BEAR|BASE|BULL):|$)", raw, re.I | re.S)
                    if m:
                        scenarios[label.lower()] = m.group(1).strip()
        except Exception:
            pass

        data["scenarios"] = scenarios
        return {"status": "ok", "symbol": symbol, "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/earnings/{symbol}")
def get_symbol_earnings(symbol: str):
    """8-quarter earnings history with beat/miss + next earnings date.

    Returns
    -------
    {
      "quarterly": [{date, actual_eps, estimate_eps, revenue, surprise_pct}],
      "next_earnings_date": str | None,
      "trailing_eps": float,  "forward_eps": float,
      "filings": [{type, date, url}]   (from SEC 8-K / EX-99)
    }
    """
    try:
        symbol = symbol.upper().strip()
        from data.fincept.yfinance_data import get_earnings_data
        data   = get_earnings_data(symbol)

        # Attach SEC filing links (8-K, EX-99) for earnings context
        try:
            from data.fincept.sec_data import get_filings
            filings = get_filings(symbol, forms="8-K,EX-99.1", limit=6)
            data["filings"] = filings
        except Exception:
            data["filings"] = []

        return {"status": "ok", "symbol": symbol, "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/financials/{symbol}")
def get_symbol_financials(symbol: str, period: str = "quarterly"):
    """Detailed income statement, balance sheet, and cash flow.

    Query params
    ------------
    period : "quarterly" (default) | "annual"

    Returns rows with human-readable numbers (1.23bn, 456.78m).
    Negatives are signaled via a "negative" boolean for frontend coloring.
    """
    try:
        symbol = symbol.upper().strip()
        if period not in ("quarterly", "annual"):
            return {"status": "error", "message": "period must be quarterly or annual"}

        from data.fincept.yfinance_data import get_financials_detailed
        data = get_financials_detailed(symbol, period=period)

        # Humanize numbers for frontend display
        def humanize(rows: list) -> list:
            out = []
            for row in rows:
                new_row = {}
                for k, v in row.items():
                    if isinstance(v, (int, float)) and v is not None:
                        new_row[k]              = _fmt_financial(v)
                        new_row[k + "_raw"]     = v
                        new_row[k + "_negative"] = v < 0
                    else:
                        new_row[k] = v
                out.append(new_row)
            return out

        data["income_statement"] = humanize(data.get("income_statement", []))
        data["balance_sheet"]    = humanize(data.get("balance_sheet", []))
        data["cash_flow"]        = humanize(data.get("cash_flow", []))

        return {"status": "ok", "symbol": symbol, "period": period, "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


@app.get("/api/holdings/{symbol}")
def get_symbol_holdings(symbol: str):
    """Institutional holders, mutual funds, insider Form 4, and politicians (STOCK Act).

    Returns
    -------
    {
      "institutional": [{holder, shares, value, pct_held, date_reported}],
      "mutual_funds":  [{holder, shares, value, pct_held, date_reported}],
      "insiders":      [{name, title, transaction, shares, value, date}],
      "politicians":   [{politician, chamber, trade_type, disclosure_date}]
    }
    """
    try:
        symbol = symbol.upper().strip()
        from data.fincept.yfinance_data import get_holdings_data
        data = get_holdings_data(symbol)

        # v13: add congressional STOCK Act trades (6h cache, no API key)
        try:
            from data.fincept.congressional_trades import get_congressional_trades
            data["politicians"] = get_congressional_trades(symbol, days_back=90)
        except Exception:
            data["politicians"] = []

        return {"status": "ok", "symbol": symbol, "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {}}


def _fmt_financial(val: float) -> str:
    """Format financial number: >1B → '1.23bn', >1M → '456.78m', else integer."""
    if val is None:
        return ""
    abs_val = abs(val)
    sign    = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.2f}bn"
    if abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.2f}m"
    return f"{sign}{abs_val:,.0f}"


# ── Dashboard ─────────────────────────────────────────────────

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse("dashboard/index.html")


@app.get("/terminal")
def serve_terminal():
    return FileResponse("dashboard/terminal.html")