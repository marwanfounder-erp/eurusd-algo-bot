"""Flask dashboard server — serves the real-time trading dashboard.

Runs in a background daemon thread so it never blocks the trading loop.
Reads state from logs/paper_account.json and the live bot state object.

Endpoints
---------
GET /              → dashboard.html
GET /api/status    → bot status, balance, P&L
GET /api/positions → open positions
GET /api/trades    → trade history (last 100)
GET /api/risk      → risk metrics
GET /api/price     → live EURUSD bid/ask
GET /api/news      → next 3 high-impact events
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, send_from_directory
from loguru import logger

_STATE_FILE   = Path("logs/paper_account.json")
_INITIAL_BAL  = 10_000.0
_DASHBOARD_DIR = Path(__file__).parent.parent  # project root

# ── Shared bot-state registry (populated by main.py) ──────────────────
_bot_state: dict[str, Any] = {
    "running":        False,
    "mode":           "paper",
    "feed_type":      "PaperFeed",
    "last_signal_ts": None,
    "start_ts":       None,
}


def update_bot_state(**kwargs: Any) -> None:
    """Called from main.py to push live bot state into the dashboard."""
    _bot_state.update(kwargs)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _read_state() -> dict[str, Any]:
    if _STATE_FILE.exists():
        try:
            with open(_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "balance": _INITIAL_BAL,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "open_position": None,
        "trades": [],
    }


def _seconds_to_london() -> int:
    """Seconds until next London session open (07:00 UTC)."""
    now = datetime.now(tz=timezone.utc)
    target_hour = 7
    if now.hour < target_hour:
        delta = (target_hour - now.hour) * 3600 - now.minute * 60 - now.second
    else:
        # next day
        delta = (24 - now.hour + target_hour) * 3600 - now.minute * 60 - now.second
    return max(0, delta)


def _today_pnl(trades: list[dict[str, Any]]) -> float:
    """Sum P&L for trades closed today (UTC)."""
    today = datetime.now(tz=timezone.utc).date().isoformat()
    return sum(
        float(t.get("pnl", 0))
        for t in trades
        if str(t.get("date", "")).startswith(today)
    )


def _equity_curve(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct balance-over-time from trade history."""
    bal = _INITIAL_BAL
    curve = [{"t": "start", "balance": bal}]
    for t in trades:
        bal += float(t.get("pnl", 0))
        curve.append({"t": str(t.get("date", "")), "balance": round(bal, 2)})
    return curve


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(_DASHBOARD_DIR))
app.config["JSON_SORT_KEYS"] = False

# Silence Flask's default werkzeug access log to keep bot logs clean
import logging as _logging
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)


@app.route("/")
def index():  # type: ignore[return]
    return send_from_directory(str(_DASHBOARD_DIR), "dashboard.html")


@app.route("/api/status")
def api_status():  # type: ignore[return]
    state = _read_state()
    balance   = float(state.get("balance", _INITIAL_BAL))
    total_ret = (balance - _INITIAL_BAL) / _INITIAL_BAL
    wins      = int(state.get("wins", 0))
    losses    = int(state.get("losses", 0))
    total     = wins + losses
    win_rate  = wins / total if total else 0.0
    today_pnl = _today_pnl(state.get("trades", []))

    last_ts = _bot_state.get("last_signal_ts")
    start_ts = _bot_state.get("start_ts")

    return jsonify({
        "running":            _bot_state.get("running", False),
        "mode":               _bot_state.get("mode", "paper"),
        "feed_type":          _bot_state.get("feed_type", "PaperFeed"),
        "balance":            round(balance, 2),
        "initial_balance":    _INITIAL_BAL,
        "today_pnl":          round(today_pnl, 2),
        "total_return_pct":   round(total_ret * 100, 2),
        "win_rate_pct":       round(win_rate * 100, 1),
        "total_trades":       total,
        "wins":               wins,
        "losses":             losses,
        "seconds_to_london":  _seconds_to_london(),
        "last_signal_ts":     last_ts,
        "start_ts":           start_ts,
        "server_time_utc":    datetime.now(tz=timezone.utc).isoformat(),
        "equity_curve":       _equity_curve(state.get("trades", [])),
    })


@app.route("/api/positions")
def api_positions():  # type: ignore[return]
    state = _read_state()
    pos   = state.get("open_position")
    positions = []
    if pos:
        # Enrich with current price if PaperFeed is active
        current_price = None
        try:
            pf = _bot_state.get("_feed_ref")
            if pf is not None:
                tick = pf.get_tick("EURUSD")
                current_price = tick.get("ask") if pos.get("direction") == "BUY" else tick.get("bid")
        except Exception:
            pass

        entry     = float(pos.get("entry", 0))
        sl        = float(pos.get("sl", 0))
        tp        = float(pos.get("tp", 0))
        lot       = float(pos.get("lot", 0))
        direction = pos.get("direction", "")
        opened_at = pos.get("opened_at", "")

        unrealized = None
        pip_move   = None
        if current_price:
            pip_move = (
                (current_price - entry) if direction == "BUY" else (entry - current_price)
            ) / 0.0001
            unrealized = round(pip_move * 10 * lot, 2)  # $10/pip/lot

        positions.append({
            "symbol":        "EURUSD",
            "direction":     direction,
            "entry":         entry,
            "current_price": current_price,
            "sl":            sl,
            "tp":            tp,
            "lot":           lot,
            "pip_move":      round(pip_move, 1) if pip_move is not None else None,
            "unrealized_pnl": unrealized,
            "opened_at":     opened_at,
        })
    return jsonify(positions)


@app.route("/api/trades")
def api_trades():  # type: ignore[return]
    state  = _read_state()
    trades = state.get("trades", [])[-100:]  # last 100
    out    = []
    for t in reversed(trades):
        entry  = float(t.get("entry", 0))
        exit_p = float(t.get("exit", 0))
        direction = t.get("direction", "")
        pip_delta = (
            (exit_p - entry) if direction == "BUY" else (entry - exit_p)
        ) / 0.0001
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

    state   = _read_state()
    balance = float(state.get("balance", _INITIAL_BAL))

    # Daily loss: sum of today's losing trades
    trades    = state.get("trades", [])
    today     = datetime.now(tz=timezone.utc).date().isoformat()
    today_pnl = sum(
        float(t.get("pnl", 0))
        for t in trades
        if str(t.get("date", "")).startswith(today)
    )
    daily_loss_pct = max(0.0, -today_pnl / _INITIAL_BAL * 100)

    # Total drawdown from initial
    total_dd_pct = max(0.0, (_INITIAL_BAL - balance) / _INITIAL_BAL * 100)

    daily_limit_pct = settings.max_daily_loss * 100
    total_limit_pct = settings.max_total_drawdown * 100

    safe = (
        daily_loss_pct < daily_limit_pct * 0.9
        and total_dd_pct < total_limit_pct * 0.9
    )

    return jsonify({
        "daily_loss_pct":    round(daily_loss_pct, 2),
        "daily_limit_pct":   daily_limit_pct,
        "total_dd_pct":      round(total_dd_pct, 2),
        "total_limit_pct":   total_limit_pct,
        "safe_to_trade":     safe,
        "balance":           round(balance, 2),
        "peak_balance":      round(max(
            (float(t.get("pnl", 0)) for t in trades),
            default=0.0
        ) + _INITIAL_BAL, 2),
    })


@app.route("/api/price")
def api_price():  # type: ignore[return]
    try:
        pf = _bot_state.get("_feed_ref")
        if pf is not None:
            tick = pf.get_tick("EURUSD")
            return jsonify({
                "bid":         tick.get("bid"),
                "ask":         tick.get("ask"),
                "spread_pips": tick.get("spread_pips"),
                "ts":          datetime.now(tz=timezone.utc).isoformat(),
                "source":      "live",
            })
    except Exception as exc:
        logger.debug("Price fetch via feed failed: {}", exc)

    # Fallback: yfinance direct
    try:
        import yfinance as yf
        t    = yf.Ticker("EURUSD=X")
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            mid = float(hist["Close"].iloc[-1])
            half = 0.6 * 0.0001 / 2
            return jsonify({
                "bid":         round(mid - half, 5),
                "ask":         round(mid + half, 5),
                "spread_pips": 0.6,
                "ts":          datetime.now(tz=timezone.utc).isoformat(),
                "source":      "yfinance",
            })
    except Exception as exc:
        logger.warning("yfinance price fallback failed: {}", exc)

    return jsonify({"bid": None, "ask": None, "spread_pips": None, "source": "unavailable"})


@app.route("/api/news")
def api_news():  # type: ignore[return]
    try:
        import time as _time
        import requests as _req
        from config import settings

        resp = _req.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=8,
        )
        resp.raise_for_status()
        events = resp.json()

        now    = datetime.now(tz=timezone.utc)
        result = []

        for ev in events:
            impact   = ev.get("impact", "").lower()
            currency = ev.get("currency", "").upper()
            if impact != "high" or currency not in ("EUR", "USD"):
                continue

            raw = ev.get("date", "")
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue

            secs = (dt - now).total_seconds()
            if secs < -3600:  # skip events more than 1h in the past
                continue

            result.append({
                "title":       ev.get("title", "Unknown"),
                "currency":    currency,
                "impact":      "High",
                "event_time":  dt.isoformat(),
                "minutes_until": round(secs / 60, 1),
                "warning":     0 < secs <= settings.news_filter_before_minutes * 60,
            })

        result.sort(key=lambda x: x["minutes_until"])
        return jsonify(result[:3])

    except Exception as exc:
        logger.warning("News API failed: {}", exc)
        return jsonify([])


# ──────────────────────────────────────────────────────────────────────
# Server launcher
# ──────────────────────────────────────────────────────────────────────

def start_dashboard(feed_ref: Any = None, port: int | None = None) -> None:
    """Start Flask in a background daemon thread.

    Parameters
    ----------
    feed_ref : PaperFeed | DataFeed | None
        Live feed object so /api/price and /api/positions can fetch
        real-time prices without going through yfinance separately.
    port : int | None
        Port to listen on.  Defaults to PORT env var, then 8080.
    """
    _bot_state["_feed_ref"] = feed_ref
    _bot_state["start_ts"]  = datetime.now(tz=timezone.utc).isoformat()

    _port = port or int(os.environ.get("PORT", 8080))

    def _run() -> None:
        logger.info("Dashboard server starting on port {}", _port)
        app.run(host="0.0.0.0", port=_port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="DashboardServer")
    t.start()
    logger.info("Dashboard available at http://0.0.0.0:{}/", _port)
