"""Flask dashboard server — serves the real-time trading dashboard.

Runs in a background daemon thread so it never blocks the trading loop.
All persistent data is read from Neon PostgreSQL via bot.database.

Endpoints
---------
GET /              → dashboard.html (no-cache)
GET /api/health    → liveness probe
GET /api/status    → bot status, balance, P&L
GET /api/positions → open positions with live unrealized P&L
GET /api/trades    → trade history (last 50, newest first)
GET /api/risk      → risk metrics + safe-to-trade flag
GET /api/price     → live EURUSD bid/ask (yfinance → raw Yahoo → cache)
GET /api/news      → next 3 high-impact EUR/USD events
GET /api/equity    → account snapshots for equity curve
GET /api/logs      → last 50 bot log entries
GET /api/stats     → full performance statistics
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests as _req
from flask import Flask, jsonify, make_response, send_from_directory
from flask_cors import CORS
from loguru import logger

_INITIAL_BAL   = 10_000.0
_DASHBOARD_DIR = Path(__file__).parent.parent   # project root
_START_TIME    = time.monotonic()

# ── Shared bot-state (populated by main.py via update_bot_state) ──────
_bot_state: dict[str, Any] = {
    "running":        False,
    "mode":           "paper",
    "feed_type":      "PaperFeed",
    "last_signal_ts": None,
    "start_ts":       None,
    "_feed_ref":      None,
}

# ── Price cache — avoids blocking Flask threads on yfinance calls ─────
_price_cache: dict[str, Any] = {
    "bid": None, "ask": None, "spread_pips": 1.2,
    "ts": None, "source": "unavailable",
}
_price_lock = threading.Lock()


def update_bot_state(**kwargs: Any) -> None:
    """Called from main.py to push live bot state into the dashboard."""
    _bot_state.update(kwargs)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _seconds_to_london() -> int:
    """Seconds until next 07:00 UTC London open."""
    now = datetime.now(tz=timezone.utc)
    if now.hour < 7:
        delta = (7 - now.hour) * 3600 - now.minute * 60 - now.second
    else:
        delta = (24 - now.hour + 7) * 3600 - now.minute * 60 - now.second
    return max(0, delta)


def _fetch_price_raw() -> dict[str, Any]:
    """Fetch EURUSD mid price.  Three-tier: feed → yfinance → raw Yahoo API."""
    half_spread = 1.2 * 0.0001 / 2

    # Tier 1 — live feed object
    try:
        feed = _bot_state.get("_feed_ref")
        if feed is not None:
            tick = feed.get_tick("EURUSD")
            return {
                "bid":         tick["bid"],
                "ask":         tick["ask"],
                "spread_pips": tick.get("spread_pips", 0.6),
                "ts":          datetime.now(tz=timezone.utc).isoformat(),
                "source":      "feed",
            }
    except Exception as exc:
        logger.debug("price tier-1 (feed) failed: {}", exc)

    # Tier 2 — yfinance thread
    try:
        result: dict[str, Any] = {}

        def _yf_fetch() -> None:
            import yfinance as yf
            hist = yf.Ticker("EURUSD=X").history(period="1d", interval="1m")
            if not hist.empty:
                result["mid"] = float(hist["Close"].iloc[-1])

        t = threading.Thread(target=_yf_fetch, daemon=True)
        t.start()
        t.join(timeout=10)
        if "mid" in result:
            mid = result["mid"]
            return {
                "bid":         round(mid - half_spread, 5),
                "ask":         round(mid + half_spread, 5),
                "spread_pips": 0.6,
                "ts":          datetime.now(tz=timezone.utc).isoformat(),
                "source":      "yfinance",
            }
    except Exception as exc:
        logger.warning("price tier-2 (yfinance) failed: {}", exc)

    # Tier 3 — raw Yahoo API
    try:
        resp = _req.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD%3DX"
            "?interval=1m&range=1d",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        mid = float(resp.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
        return {
            "bid":         round(mid - half_spread, 5),
            "ask":         round(mid + half_spread, 5),
            "spread_pips": 0.6,
            "ts":          datetime.now(tz=timezone.utc).isoformat(),
            "source":      "yahoo-api",
        }
    except Exception as exc:
        logger.warning("price tier-3 (raw Yahoo API) failed: {}", exc)

    return {
        "bid": None, "ask": None, "spread_pips": None,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "source": "unavailable",
    }


# News cache — re-fetch at most once per 15 minutes
_news_cache: list[dict[str, Any]] = []
_news_cache_ts: float = 0.0
_NEWS_TTL = 900


def _get_news() -> list[dict[str, Any]]:
    global _news_cache, _news_cache_ts
    now_mono = time.monotonic()
    if _news_cache and (now_mono - _news_cache_ts) < _NEWS_TTL:
        return _news_cache
    try:
        resp = _req.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning("dashboard news fetch failed: {} — returning cached", exc)
        return _news_cache

    now = datetime.now(tz=timezone.utc)
    result: list[dict[str, Any]] = []
    for ev in raw:
        impact  = ev.get("impact", "").lower()
        country = ev.get("country", ev.get("currency", "")).upper()
        if impact != "high" or country not in ("EUR", "USD"):
            continue
        raw_date = ev.get("date", "")
        if not raw_date:
            continue
        try:
            dt = datetime.fromisoformat(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
        secs = (dt - now).total_seconds()
        # Show events up to 6 hours in the past (so today's full schedule is visible)
        if secs < -6 * 3600:
            continue
        from config import settings
        # Normalise forecast/previous — may be empty string or None
        forecast = ev.get("forecast") or ev.get("Forecast") or ""
        previous = ev.get("previous") or ev.get("Previous") or ""
        result.append({
            "title":         ev.get("title", "Unknown"),
            "currency":      country,
            "impact":        "High",
            "event_time":    dt.isoformat(),
            "minutes_until": round(secs / 60, 1),
            "warning":       0 < secs <= settings.news_filter_before_minutes * 60,
            "forecast":      str(forecast).strip(),
            "previous":      str(previous).strip(),
        })
    result.sort(key=lambda x: x["event_time"])
    _news_cache    = result
    _news_cache_ts = now_mono
    return result


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(_DASHBOARD_DIR))
app.config["JSON_SORT_KEYS"] = False
CORS(app)

import logging as _logging
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)


@app.route("/")
def index():  # type: ignore[return]
    resp = make_response(send_from_directory(str(_DASHBOARD_DIR), "dashboard.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/health")
def api_health():  # type: ignore[return]
    return jsonify({
        "status":         "ok",
        "timestamp":      datetime.now(tz=timezone.utc).isoformat(),
        "uptime_seconds": round(time.monotonic() - _START_TIME),
        "bot_running":    _bot_state.get("running", False),
    })


@app.route("/api/status")
def api_status():  # type: ignore[return]
    from bot.database import get_db
    db = get_db()
    stats = db.get_stats()

    balance     = float(stats.get("current_balance", _INITIAL_BAL))
    wins        = int(stats.get("wins", 0))
    losses      = int(stats.get("losses", 0))
    total       = wins + losses
    win_rate    = round(wins / total * 100, 1) if total else 0.0
    total_ret   = round((balance - _INITIAL_BAL) / _INITIAL_BAL * 100, 2)

    today_pnl = sum(
        float(t.get("pnl") or 0)
        for t in db.get_today_trades()
    )

    return jsonify({
        "balance":          round(balance, 2),
        "starting_balance": _INITIAL_BAL,
        "today_pnl":        round(today_pnl, 2),
        "today_pnl_pct":    round(today_pnl / _INITIAL_BAL * 100, 2),
        "total_return_pct": total_ret,
        "total_trades":     total,
        "wins":             wins,
        "losses":           losses,
        "win_rate":         win_rate,
        "mode":             _bot_state.get("mode", "paper").upper(),
        "feed":             _bot_state.get("feed_type", "PaperFeed"),
        "status":           "Running" if _bot_state.get("running") else "Stopped",
        "running":          _bot_state.get("running", False),
        "last_signal":      _bot_state.get("last_signal_ts"),
        "running_since":    _bot_state.get("start_ts"),
        "seconds_to_london": _seconds_to_london(),
        "server_time_utc":  datetime.now(tz=timezone.utc).isoformat(),
    })


@app.route("/api/positions")
def api_positions():  # type: ignore[return]
    from bot.database import get_db
    db = get_db()
    open_trades = db.get_open_trades()

    with _price_lock:
        cached = dict(_price_cache)

    positions = []
    for pos in open_trades:
        direction     = pos.get("direction", "")
        entry         = float(pos.get("entry_price") or 0)
        current_price = cached.get("ask") if direction == "BUY" else cached.get("bid")
        pip_move      = None
        unrealized    = None
        if current_price:
            pip_move   = ((current_price - entry) if direction == "BUY"
                          else (entry - current_price)) / 0.0001
            unrealized = round(pip_move * 10 * float(pos.get("lot_size") or 0), 2)
        positions.append({
            "symbol":         pos.get("symbol", "EURUSD"),
            "direction":      direction,
            "entry":          entry,
            "current_price":  current_price,
            "sl":             float(pos.get("stop_loss") or 0),
            "tp":             float(pos.get("take_profit") or 0),
            "lot":            float(pos.get("lot_size") or 0),
            "pip_move":       round(pip_move, 1) if pip_move is not None else None,
            "unrealized_pnl": unrealized,
            "opened_at":      str(pos.get("opened_at", "")),
        })
    return jsonify({"positions": positions})


@app.route("/api/trades")
def api_trades():  # type: ignore[return]
    from bot.database import get_db
    db = get_db()
    trades = db.get_all_trades(limit=50)
    out = []
    for t in trades:
        entry  = float(t.get("entry_price") or 0)
        exit_p = float(t.get("exit_price") or 0)
        direction = t.get("direction", "")
        out.append({
            "date":      str(t.get("opened_at", ""))[:10],
            "direction": direction,
            "entry":     entry,
            "exit":      exit_p,
            "sl":        float(t.get("stop_loss") or 0),
            "tp":        float(t.get("take_profit") or 0),
            "lot":       float(t.get("lot_size") or 0),
            "sl_pips":   0.0,
            "pips":      round(float(t.get("pips") or 0), 1),
            "pnl":       round(float(t.get("pnl") or 0), 2),
            "outcome":   t.get("result", ""),
            "rsi":       float(t.get("rsi") or 0),
        })
    return jsonify({"trades": out})


@app.route("/api/risk")
def api_risk():  # type: ignore[return]
    from bot.database import get_db
    from config import settings
    db = get_db()
    stats   = db.get_stats()
    balance = float(stats.get("current_balance", _INITIAL_BAL))

    today_pnl = sum(
        float(t.get("pnl") or 0)
        for t in db.get_today_trades()
    )
    daily_loss_pct = max(0.0, -today_pnl / _INITIAL_BAL * 100)
    drawdown_pct   = max(0.0, (_INITIAL_BAL - balance) / _INITIAL_BAL * 100)
    daily_limit    = settings.max_daily_loss * 100
    total_limit    = settings.max_total_drawdown * 100
    safe = daily_loss_pct < daily_limit * 0.9 and drawdown_pct < total_limit * 0.9

    return jsonify({
        "daily_loss_pct":  round(daily_loss_pct, 2),
        "daily_limit_pct": daily_limit,
        "drawdown_pct":    round(drawdown_pct, 2),
        "total_limit_pct": total_limit,
        "safe_to_trade":   safe,
        "balance":         round(balance, 2),
    })


@app.route("/api/price")
def api_price():  # type: ignore[return]
    with _price_lock:
        return jsonify(dict(_price_cache))


@app.route("/api/news")
def api_news():  # type: ignore[return]
    return jsonify(_get_news())


@app.route("/api/equity")
def api_equity():  # type: ignore[return]
    from bot.database import get_db
    db = get_db()
    snapshots = db.get_snapshots(days=30)
    out = []
    for s in snapshots:
        out.append({
            "balance":    float(s.get("balance") or _INITIAL_BAL),
            "daily_pnl":  float(s.get("daily_pnl") or 0),
            "return_pct": float(s.get("total_return_pct") or 0),
            "ts":         str(s.get("timestamp", "")),
        })
    return jsonify({"snapshots": out})


@app.route("/api/logs")
def api_logs():  # type: ignore[return]
    from bot.database import get_db
    db = get_db()
    logs = db.get_recent_logs(limit=50)
    out = []
    for log in logs:
        out.append({
            "level":   log.get("level", "INFO"),
            "message": log.get("message", ""),
            "ts":      str(log.get("created_at", "")),
        })
    return jsonify({"logs": out})


@app.route("/api/stats")
def api_stats():  # type: ignore[return]
    from bot.database import get_db
    db = get_db()
    return jsonify(db.get_trade_stats())


# ──────────────────────────────────────────────────────────────────────
# Background price poller
# ──────────────────────────────────────────────────────────────────────

def _price_poller() -> None:
    while True:
        try:
            fresh = _fetch_price_raw()
            with _price_lock:
                _price_cache.update(fresh)
        except Exception as exc:
            logger.warning("price poller error: {}", exc)
        time.sleep(15)


# ──────────────────────────────────────────────────────────────────────
# Server launcher
# ──────────────────────────────────────────────────────────────────────

def start_dashboard(feed_ref: Any = None, port: int | None = None) -> None:
    """Start Flask + price poller in background daemon threads."""
    _bot_state["_feed_ref"] = feed_ref
    _bot_state["start_ts"]  = datetime.now(tz=timezone.utc).isoformat()

    _port = port or int(os.environ.get("PORT", 8080))

    threading.Thread(target=_price_poller, daemon=True, name="PricePoller").start()

    def _run() -> None:
        logger.info("Dashboard server starting on port {}", _port)
        app.run(host="0.0.0.0", port=_port, debug=False, use_reloader=False)

    threading.Thread(target=_run, daemon=True, name="DashboardServer").start()
    logger.info("Dashboard available at http://0.0.0.0:{}/", _port)
