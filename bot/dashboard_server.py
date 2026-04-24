"""Flask dashboard server — serves the real-time trading dashboard.

Runs in a background daemon thread so it never blocks the trading loop.
Reads state from logs/paper_account.json and the live bot state object.

Endpoints
---------
GET /              → dashboard.html
GET /api/health    → liveness probe
GET /api/status    → bot status, balance, P&L, equity curve
GET /api/positions → open positions with live unrealized P&L
GET /api/trades    → trade history (last 100, newest first)
GET /api/risk      → risk metrics + safe-to-trade flag
GET /api/price     → live EURUSD bid/ask (yfinance → raw Yahoo → cache)
GET /api/news      → next 3 high-impact EUR/USD events
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests as _req
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from loguru import logger

_STATE_FILE    = Path("logs/paper_account.json")
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

# ── Price cache — avoids hammering Yahoo on every /api/price call ─────
_price_cache: dict[str, Any] = {"bid": None, "ask": None, "spread_pips": 0.6,
                                 "ts": None, "source": "unavailable"}
_price_lock = threading.Lock()


def update_bot_state(**kwargs: Any) -> None:
    """Called from main.py to push live bot state into the dashboard."""
    _bot_state.update(kwargs)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _read_state() -> dict[str, Any]:
    """Read paper_account.json; return safe defaults if missing/corrupt."""
    if _STATE_FILE.exists():
        try:
            with open(_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # Back-fill any missing keys so callers never get KeyError
            data.setdefault("balance",       _INITIAL_BAL)
            data.setdefault("total_trades",  0)
            data.setdefault("wins",          0)
            data.setdefault("losses",        0)
            data.setdefault("gross_profit",  0.0)
            data.setdefault("gross_loss",    0.0)
            data.setdefault("open_position", None)
            data.setdefault("trades",        [])
            return data
        except Exception as exc:
            logger.warning("dashboard: could not read state file: {}", exc)
    return {
        "balance":       _INITIAL_BAL,
        "total_trades":  0,
        "wins":          0,
        "losses":        0,
        "gross_profit":  0.0,
        "gross_loss":    0.0,
        "open_position": None,
        "trades":        [],
    }


def _init_state_file() -> None:
    """Create logs/paper_account.json with defaults if it doesn't exist."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_FILE.exists():
        defaults = {
            "balance":       _INITIAL_BAL,
            "total_trades":  0,
            "wins":          0,
            "losses":        0,
            "gross_profit":  0.0,
            "gross_loss":    0.0,
            "open_position": None,
            "trades":        [],
        }
        tmp = str(_STATE_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=2)
        os.replace(tmp, str(_STATE_FILE))
        logger.info("dashboard: created default {}", _STATE_FILE)


def _seconds_to_london() -> int:
    """Seconds until next 07:00 UTC London open."""
    now = datetime.now(tz=timezone.utc)
    if now.hour < 7:
        delta = (7 - now.hour) * 3600 - now.minute * 60 - now.second
    else:
        delta = (24 - now.hour + 7) * 3600 - now.minute * 60 - now.second
    return max(0, delta)


def _today_pnl(trades: list[dict[str, Any]]) -> float:
    today = datetime.now(tz=timezone.utc).date().isoformat()
    return sum(
        float(t.get("pnl", 0))
        for t in trades
        if str(t.get("date", "")).startswith(today)
    )


def _equity_curve(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bal = _INITIAL_BAL
    curve: list[dict[str, Any]] = [{"t": "start", "balance": bal}]
    for t in trades:
        bal += float(t.get("pnl", 0))
        curve.append({"t": str(t.get("date", "")), "balance": round(bal, 2)})
    return curve


def _fetch_price_raw() -> dict[str, Any]:
    """Fetch EURUSD mid price.  Three-tier: feed → yfinance → raw Yahoo API."""
    half_spread = 0.6 * 0.0001 / 2

    # Tier 1 — live feed object (already has the price from the main loop tick)
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

    # Tier 2 — yfinance (with a hard 10-second timeout via thread)
    try:
        result: dict[str, Any] = {}

        def _yf_fetch() -> None:
            import yfinance as yf
            hist = yf.Ticker("EURUSD=X").history(period="1d", interval="1m")
            if not hist.empty:
                mid = float(hist["Close"].iloc[-1])
                result["mid"] = mid

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
        logger.warning("price tier-2 (yfinance) timed out or returned empty")
    except Exception as exc:
        logger.warning("price tier-2 (yfinance) failed: {}", exc)

    # Tier 3 — raw Yahoo Finance chart API
    try:
        resp = _req.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD%3DX"
            "?interval=1m&range=1d",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        mid = float(
            data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        )
        return {
            "bid":         round(mid - half_spread, 5),
            "ask":         round(mid + half_spread, 5),
            "spread_pips": 0.6,
            "ts":          datetime.now(tz=timezone.utc).isoformat(),
            "source":      "yahoo-api",
        }
    except Exception as exc:
        logger.warning("price tier-3 (raw Yahoo API) failed: {}", exc)

    return {"bid": None, "ask": None, "spread_pips": None,
            "ts": datetime.now(tz=timezone.utc).isoformat(), "source": "unavailable"}


# News cache — re-fetch at most once per 15 minutes to avoid rate-limiting
_news_cache: list[dict[str, Any]] = []
_news_cache_ts: float = 0.0
_NEWS_TTL = 900  # 15 minutes


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
        logger.info("dashboard news: fetched {} events", len(raw))
    except Exception as exc:
        logger.warning("dashboard news fetch failed: {} — returning cached", exc)
        return _news_cache   # stale cache beats nothing

    now = datetime.now(tz=timezone.utc)
    result: list[dict[str, Any]] = []

    for ev in raw:
        impact  = ev.get("impact", "").lower()
        # FF API uses "country" not "currency"
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
        except (ValueError, TypeError) as exc:
            logger.debug("news date parse error: {} for {}", exc, raw_date)
            continue

        secs = (dt - now).total_seconds()
        if secs < -3600:   # skip events > 1h in the past
            continue

        from config import settings
        result.append({
            "title":         ev.get("title", "Unknown"),
            "currency":      country,
            "impact":        "High",
            "event_time":    dt.isoformat(),
            "minutes_until": round(secs / 60, 1),
            "warning":       0 < secs <= settings.news_filter_before_minutes * 60,
        })

    result.sort(key=lambda x: x["minutes_until"])
    _news_cache    = result
    _news_cache_ts = now_mono
    return result


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(_DASHBOARD_DIR))
app.config["JSON_SORT_KEYS"] = False
CORS(app)  # allow browser fetch() from any origin

import logging as _logging
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)


@app.route("/")
def index():  # type: ignore[return]
    from flask import make_response
    resp = make_response(send_from_directory(str(_DASHBOARD_DIR), "dashboard.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/health")
def api_health():  # type: ignore[return]
    return jsonify({
        "status":          "ok",
        "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
        "uptime_seconds":  round(time.monotonic() - _START_TIME),
        "bot_running":     _bot_state.get("running", False),
    })


@app.route("/api/status")
def api_status():  # type: ignore[return]
    state     = _read_state()
    balance   = float(state["balance"])
    total_ret = (balance - _INITIAL_BAL) / _INITIAL_BAL
    wins      = int(state["wins"])
    losses    = int(state["losses"])
    total     = wins + losses
    win_rate  = wins / total if total else 0.0
    today_pnl = _today_pnl(state["trades"])

    return jsonify({
        "running":           _bot_state.get("running", False),
        "mode":              _bot_state.get("mode", "paper"),
        "feed_type":         _bot_state.get("feed_type", "PaperFeed"),
        "balance":           round(balance, 2),
        "initial_balance":   _INITIAL_BAL,
        "today_pnl":         round(today_pnl, 2),
        "total_return_pct":  round(total_ret * 100, 2),
        "win_rate_pct":      round(win_rate * 100, 1),
        "total_trades":      total,
        "wins":              wins,
        "losses":            losses,
        "seconds_to_london": _seconds_to_london(),
        "last_signal_ts":    _bot_state.get("last_signal_ts"),
        "start_ts":          _bot_state.get("start_ts"),
        "server_time_utc":   datetime.now(tz=timezone.utc).isoformat(),
        "equity_curve":      _equity_curve(state["trades"]),
    })


@app.route("/api/positions")
def api_positions():  # type: ignore[return]
    state     = _read_state()
    pos       = state.get("open_position")
    positions = []
    if pos:
        current_price = None
        try:
            with _price_lock:
                cached = dict(_price_cache)
            direction = pos.get("direction", "")
            current_price = cached.get("ask") if direction == "BUY" else cached.get("bid")
        except Exception:
            pass

        entry     = float(pos.get("entry", 0))
        direction = pos.get("direction", "")
        lot       = float(pos.get("lot", 0))
        pip_move  = None
        unrealized = None
        if current_price:
            pip_move   = ((current_price - entry) if direction == "BUY"
                          else (entry - current_price)) / 0.0001
            unrealized = round(pip_move * 10 * lot, 2)

        positions.append({
            "symbol":         "EURUSD",
            "direction":      direction,
            "entry":          entry,
            "current_price":  current_price,
            "sl":             float(pos.get("sl", 0)),
            "tp":             float(pos.get("tp", 0)),
            "lot":            lot,
            "pip_move":       round(pip_move, 1) if pip_move is not None else None,
            "unrealized_pnl": unrealized,
            "opened_at":      pos.get("opened_at", ""),
        })
    return jsonify(positions)


@app.route("/api/trades")
def api_trades():  # type: ignore[return]
    state  = _read_state()
    trades = state["trades"][-100:]
    out    = []
    for t in reversed(trades):
        entry     = float(t.get("entry", 0))
        exit_p    = float(t.get("exit",  0))
        direction = t.get("direction", "")
        pip_delta = ((exit_p - entry) if direction == "BUY"
                     else (entry - exit_p)) / 0.0001
        out.append({
            "date":      t.get("date", ""),
            "direction": direction,
            "entry":     entry,
            "exit":      exit_p,
            "sl":        float(t.get("sl", 0)),
            "tp":        float(t.get("tp", 0)),
            "lot":       float(t.get("lot", 0)),
            "sl_pips":   float(t.get("sl_pips", 0)),
            "pips":      round(pip_delta, 1),
            "pnl":       round(float(t.get("pnl", 0)), 2),
            "outcome":   t.get("outcome", ""),
            "rsi":       float(t.get("rsi", 0)),
        })
    return jsonify(out)


@app.route("/api/risk")
def api_risk():  # type: ignore[return]
    from config import settings

    state     = _read_state()
    balance   = float(state["balance"])
    trades    = state["trades"]
    today     = datetime.now(tz=timezone.utc).date().isoformat()
    today_pnl = sum(
        float(t.get("pnl", 0))
        for t in trades
        if str(t.get("date", "")).startswith(today)
    )
    daily_loss_pct = max(0.0, -today_pnl / _INITIAL_BAL * 100)
    total_dd_pct   = max(0.0, (_INITIAL_BAL - balance) / _INITIAL_BAL * 100)
    daily_limit    = settings.max_daily_loss * 100
    total_limit    = settings.max_total_drawdown * 100
    safe = daily_loss_pct < daily_limit * 0.9 and total_dd_pct < total_limit * 0.9

    return jsonify({
        "daily_loss_pct":  round(daily_loss_pct, 2),
        "daily_limit_pct": daily_limit,
        "total_dd_pct":    round(total_dd_pct, 2),
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
    return jsonify(_get_news()[:3])


# ──────────────────────────────────────────────────────────────────────
# Background price poller — updates cache every 15 s so /api/price
# is always instant (no blocking yfinance call on the request thread)
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
    _init_state_file()

    _port = port or int(os.environ.get("PORT", 8080))

    # Price poller thread
    threading.Thread(target=_price_poller, daemon=True, name="PricePoller").start()

    # Flask thread
    def _run() -> None:
        logger.info("Dashboard server starting on port {}", _port)
        app.run(host="0.0.0.0", port=_port, debug=False, use_reloader=False)

    threading.Thread(target=_run, daemon=True, name="DashboardServer").start()
    logger.info("Dashboard available at http://0.0.0.0:{}/", _port)
