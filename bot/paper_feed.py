"""Paper trading data feed backed by yfinance real-time EURUSD data.

Used automatically when MetaTrader5 is unavailable (Linux / Railway).

Account state is persisted to Neon PostgreSQL via bot.database.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from bot.database import get_db

_SYMBOL_YF = "EURUSD=X"
_SPREAD_PIP = 0.6
_PIP = 0.0001
_INITIAL_BALANCE = 10_000.0
_SNAPSHOT_INTERVAL = 3600  # seconds between hourly snapshots


class PaperFeed:
    """yfinance-backed paper trading data feed.

    Implements the same interface as DataFeed so TradingBot, strategy, and
    risk manager all work without modification.
    """

    def __init__(self) -> None:
        self._ticker = yf.Ticker(_SYMBOL_YF)
        self._open_position: dict[str, Any] | None = None
        self._last_snapshot_ts: float = 0.0

        # Load persisted balance from DB; fall back to initial if DB empty
        db = get_db()
        stats = db.get_stats()
        self._balance: float = float(stats.get("balance", _INITIAL_BALANCE))

        logger.info(
            "PaperFeed initialised | symbol={} balance=${:.2f}",
            _SYMBOL_YF, self._balance,
        )

    # ── DataFeed compatibility shims ───────────────────────────────────

    def initialize_mt5(self) -> bool:           # pragma: no cover
        """Always succeeds — no MT5 needed in paper mode."""
        return True

    def shutdown(self) -> None:
        """Persist final state on graceful exit."""
        self._save_snapshot()
        logger.info("PaperFeed shut down")

    def ensure_connected(self) -> None:
        """No-op — yfinance is stateless."""

    # ── Market data ────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> dict[str, float]:  # noqa: ARG002
        """Return current EURUSD bid/ask from Yahoo Finance.

        Yahoo Finance returns the mid price; we add ±0.6 pip spread.
        Uses 1-minute history for the freshest possible price.
        Falls back to 1-hour history on network errors.
        """
        try:
            hist = self._ticker.history(period="1d", interval="1m")
            if hist.empty:
                raise RuntimeError("1m history empty")
            mid: float = float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("yfinance 1m tick failed: {} — falling back to 1h", exc)
            try:
                hist = self._ticker.history(period="2d", interval="1h")
                if hist.empty:
                    raise RuntimeError("1h history also empty") from exc
                mid = float(hist["Close"].iloc[-1])
            except Exception as exc2:
                raise RuntimeError("yfinance: no price data available") from exc2

        half_spread = _SPREAD_PIP * _PIP / 2
        result = {
            "bid": round(mid - half_spread, 5),
            "ask": round(mid + half_spread, 5),
            "spread_pips": _SPREAD_PIP,
            "time": datetime.now(tz=timezone.utc),
        }
        logger.debug(
            "PaperFeed tick | bid={bid} ask={ask} spread={spread_pips}pips",
            **result,
        )
        return result

    def get_candles(
        self,
        symbol: str,  # noqa: ARG002
        timeframe: str,  # noqa: ARG002
        count: int = 200,
    ) -> pd.DataFrame:
        """Return real H1 OHLCV data from Yahoo Finance."""
        days = min(int(count / 24) + 2, 729)
        period = f"{days}d"
        try:
            df = self._ticker.history(period=period, interval="1h")
        except Exception as exc:
            raise RuntimeError(f"yfinance get_candles failed: {exc}") from exc

        if df.empty:
            raise RuntimeError("yfinance returned empty candle data")

        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "datetime"
        df = df.sort_index().tail(count)

        logger.debug(
            "PaperFeed candles | bars={} from={} to={}",
            len(df),
            df.index[0].isoformat() if len(df) else "n/a",
            df.index[-1].isoformat() if len(df) else "n/a",
        )
        return df

    def get_account_info(self) -> dict[str, float]:
        """Return simulated account snapshot."""
        return {
            "balance":     self._balance,
            "equity":      self._balance,
            "margin":      0.0,
            "free_margin": self._balance,
            "profit":      self._balance - _INITIAL_BALANCE,
            "margin_level": 0.0,
        }

    # ── Position simulation ────────────────────────────────────────────

    def open_position(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        sl_pips: float,
        rsi: float | None = None,
    ) -> dict[str, Any]:
        """Record a simulated open position and persist to DB."""
        opened_at = datetime.now(tz=timezone.utc).isoformat()
        trade_dict = {
            "direction": direction,
            "entry":     entry,
            "sl":        sl,
            "tp":        tp,
            "lot":       lot,
            "sl_pips":   sl_pips,
            "rsi":       rsi,
            "opened_at": opened_at,
            "date":      date.today().isoformat(),
        }
        db = get_db()
        trade_id = db.save_trade(trade_dict)

        pos = {**trade_dict, "trade_id": trade_id}
        self._open_position = pos

        logger.info(
            "PaperFeed | position opened {} {:.2f}lots @ {:.5f} sl={:.5f} tp={:.5f}",
            direction, lot, entry, sl, tp,
        )
        return pos

    def close_position(self, exit_price: float, outcome: str) -> dict[str, Any]:
        """Close the open position, update DB and in-memory balance."""
        pos = self._open_position
        if pos is None:
            logger.warning("close_position called but no open position")
            return {}

        _PIP_VALUE = 10.0  # $10 per pip per standard lot for EURUSD
        direction = pos["direction"]
        entry = pos["entry"]
        lot = pos["lot"]

        if outcome == "win":
            from config import settings
            risk_amt = self._balance * 0.01
            pnl = risk_amt * settings.rr_ratio
        elif outcome == "loss":
            risk_amt = self._balance * 0.01
            pnl = -risk_amt
        else:
            pnl_pips = (
                (exit_price - entry) if direction == "BUY" else (entry - exit_price)
            ) / _PIP
            pnl = pnl_pips * _PIP_VALUE * lot

        pip_delta = (
            (exit_price - entry) if direction == "BUY" else (entry - exit_price)
        ) / _PIP

        self._balance += pnl

        db = get_db()
        trade_id = pos.get("trade_id", "")

        # Close in DB
        if trade_id:
            db.close_trade(trade_id, exit_price, round(pnl, 2), round(pip_delta, 1), outcome)

        # Update aggregate stats in DB
        stats = db.get_stats()
        wins   = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        gross_profit = float(stats.get("gross_profit", 0.0))
        gross_loss   = float(stats.get("gross_loss", 0.0))
        total_trades = int(stats.get("total_trades", 0)) + 1

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

        db.update_stats({
            "balance":      round(self._balance, 2),
            "total_trades": total_trades,
            "wins":         wins,
            "losses":       losses,
            "gross_profit": round(gross_profit, 2),
            "gross_loss":   round(gross_loss, 2),
        })

        # Snapshot after every trade close
        daily_pnl = self._today_pnl_from_db()
        return_pct = (self._balance - _INITIAL_BALANCE) / _INITIAL_BALANCE * 100
        db.save_snapshot(
            balance=round(self._balance, 2),
            equity=round(self._balance, 2),
            daily_pnl=round(daily_pnl, 2),
            return_pct=round(return_pct, 2),
        )
        self._last_snapshot_ts = time.monotonic()

        trade_record = {
            **pos,
            "exit":    exit_price,
            "pnl":     round(pnl, 2),
            "outcome": outcome,
        }
        self._open_position = None

        logger.info(
            "PaperFeed | position closed {} @ {:.5f} pnl=${:.2f} balance=${:.2f}",
            outcome.upper(), exit_price, pnl, self._balance,
        )
        return trade_record

    def get_open_position(self) -> dict[str, Any] | None:
        """Return the current open position or None."""
        return self._open_position

    # ── Hourly snapshot helper ─────────────────────────────────────────

    def maybe_save_snapshot(self) -> None:
        """Call from the bot main loop; saves hourly DB snapshot."""
        if time.monotonic() - self._last_snapshot_ts >= _SNAPSHOT_INTERVAL:
            self._save_snapshot()

    def _save_snapshot(self) -> None:
        db = get_db()
        daily_pnl = self._today_pnl_from_db()
        return_pct = (self._balance - _INITIAL_BALANCE) / _INITIAL_BALANCE * 100
        db.save_snapshot(
            balance=round(self._balance, 2),
            equity=round(self._balance, 2),
            daily_pnl=round(daily_pnl, 2),
            return_pct=round(return_pct, 2),
        )
        self._last_snapshot_ts = time.monotonic()

    def _today_pnl_from_db(self) -> float:
        db = get_db()
        trades = db.get_today_trades()
        return sum(float(t.get("pnl") or 0) for t in trades if t.get("closed_at"))

    # ── Account summary helper ─────────────────────────────────────────

    def account_summary(self) -> str:
        """Human-readable paper account summary."""
        db = get_db()
        stats = db.get_stats()
        bal   = self._balance
        ret   = (bal - _INITIAL_BALANCE) / _INITIAL_BALANCE
        wins  = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        total = wins + losses
        wr    = wins / total if total else 0.0
        return (
            f"Balance: ${bal:,.2f} ({ret:+.2%}) | "
            f"Trades: {total} | WR: {wr:.1%} | "
            f"Wins: {wins} Losses: {losses}"
        )
