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
_SPREAD_PIP = 1.2
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
        self._balance: float = float(stats.get("current_balance", _INITIAL_BALANCE))

        from bot.notifier import TelegramNotifier
        self._notifier = TelegramNotifier()

        logger.info(
            "PaperFeed initialised | symbol={} balance=${:.2f}",
            _SYMBOL_YF, self._balance,
        )

        # Restore any open positions from DB (survives restarts)
        self._restore_positions()

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
            "current_balance": round(self._balance, 2),
            "total_trades":    total_trades,
            "wins":            wins,
            "losses":          losses,
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

    # ── DB-backed position helpers ─────────────────────────────────────

    def has_open_position(self) -> bool:
        """Check the DB directly — survives restarts."""
        try:
            return len(get_db().get_open_trades()) > 0
        except Exception:
            return False  # safe default: don't block trading on DB error

    def _restore_positions(self) -> None:
        """On startup, reload any open trades from DB into in-memory state.

        If more than one open trade exists (duplicate from a crash/restart),
        keep the newest and cancel the rest automatically.
        """
        try:
            db = get_db()
            open_trades = db.get_open_trades()

            if not open_trades:
                logger.info("No open positions to restore")
                return

            # Auto-cancel duplicates — keep only the most recent
            if len(open_trades) > 1:
                sorted_trades = sorted(
                    open_trades,
                    key=lambda x: str(x.get("opened_at", "")),
                    reverse=True,
                )
                open_trades = sorted_trades[:1]
                duplicates  = sorted_trades[1:]
                logger.warning(
                    "Found {} open trade(s) in DB — cancelling {} duplicate(s)",
                    len(sorted_trades), len(duplicates),
                )
                for dup in duplicates:
                    dup_id = str(dup["id"])
                    # Only update status — result column is varchar(4) and can't hold 'cancelled'
                    db.update_trade(dup_id, {"status": "closed"})
                    logger.info("  Cancelled duplicate trade {}", dup_id[:8])

            t = open_trades[0]
            self._open_position = {
                "trade_id": str(t["id"]),
                "direction": t["direction"],
                "entry":     float(t["entry_price"]),
                "sl":        float(t["stop_loss"]),
                "tp":        float(t["take_profit"]),
                "lot":       float(t["lot_size"]),
                "sl_pips":   abs(float(t["entry_price"]) - float(t["stop_loss"])) / _PIP,
                "rsi":       t.get("rsi"),
                "opened_at": str(t.get("opened_at", "")),
                "date":      str(t.get("opened_at", ""))[:10],
            }
            logger.info("Restored 1 open position from DB")
            logger.info(
                "  Restored: {} entry={} sl={} tp={}",
                t["direction"], t["entry_price"], t["stop_loss"], t["take_profit"],
            )
        except Exception as exc:
            logger.error("Failed to restore positions: {}", exc)

    def monitor_positions(self) -> None:
        """Check every open DB trade against current price.

        Order of checks per trade:
          1. Time-based exit   — close after 20 hours at current price
          2. TP / SL hit       — close at target or stop
          3. Breakeven stop    — move SL to entry+2pips once price is halfway to TP
        """
        db = get_db()
        open_trades = db.get_open_trades()
        if not open_trades:
            return

        try:
            tick = self.get_tick("EURUSD")
        except Exception:
            return

        current_price = tick["bid"]

        for trade in open_trades:
            entry     = float(trade["entry_price"])
            sl        = float(trade["stop_loss"])
            tp        = float(trade["take_profit"])
            direction = trade["direction"]
            lot       = float(trade["lot_size"])
            trade_id  = str(trade["id"])  # UUID → str for slicing and DB calls

            # ── 1. Time-based exit (20-hour limit) ────────────────────
            opened_at_raw = trade.get("opened_at")
            if opened_at_raw:
                opened_at = datetime.fromisoformat(str(opened_at_raw))
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                hours_open = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
                if hours_open >= 20:
                    pips = (
                        (current_price - entry) if direction == "BUY" else (entry - current_price)
                    ) / _PIP
                    pnl = pips * 10 * lot
                    result = "win" if pnl > 0 else "loss"
                    db.close_trade(trade_id, current_price, round(pnl, 2), round(pips, 1), result)
                    self._balance += pnl
                    self._open_position = None
                    logger.info(
                        "Trade {} CLOSED (time limit {:.1f}h) @ {:.5f} pnl=${:.2f}",
                        trade_id[:8], hours_open, current_price, pnl,
                    )
                    self._notifier.send(
                        f"⏰ Trade closed (time limit) | P&L: ${pnl:.2f} | {hours_open:.1f}h open"
                    )
                    continue

            # ── 2. TP / SL hit ────────────────────────────────────────
            if direction == "BUY":
                if current_price >= tp:
                    pips = (tp - entry) / _PIP
                    pnl  = pips * 10 * lot
                    db.close_trade(trade_id, tp, round(pnl, 2), round(pips, 1), "win")
                    self._balance += pnl
                    self._open_position = None
                    logger.info("Trade {} CLOSED WIN @ {} pnl=+${:.2f}", trade_id[:8], tp, pnl)
                    self._notifier.send(f"✅ WIN +${pnl:.2f} ({pips:.1f} pips)")
                elif current_price <= sl:
                    pips = (entry - sl) / _PIP
                    pnl  = -(pips * 10 * lot)
                    db.close_trade(trade_id, sl, round(pnl, 2), round(-pips, 1), "loss")
                    self._balance += pnl
                    self._open_position = None
                    logger.info("Trade {} CLOSED LOSS @ {} pnl=-${:.2f}", trade_id[:8], sl, abs(pnl))
                    self._notifier.send(f"❌ LOSS -${abs(pnl):.2f} ({pips:.1f} pips)")
                else:
                    # ── 3. Breakeven — only when SL is still below entry ──
                    if sl < entry:
                        sl_risk_pips = (entry - sl) / _PIP
                        be_trigger = entry + sl_risk_pips * 0.5 * _PIP
                        if current_price >= be_trigger:
                            breakeven_sl = round(entry + 2 * _PIP, 5)
                            db.update_trade(trade_id, {"stop_loss": breakeven_sl})
                            if self._open_position and self._open_position.get("trade_id") == trade_id:
                                self._open_position["sl"] = breakeven_sl
                            logger.info(
                                "Trade {} SL moved to breakeven {:.5f}", trade_id[:8], breakeven_sl
                            )
                            self._notifier.send("🔒 SL moved to breakeven — trade protected")

                    unrealized = (current_price - entry) / _PIP
                    logger.info(
                        "Trade {} OPEN | price={:.5f} unrealized={:.1f}pips",
                        trade_id[:8], current_price, unrealized,
                    )

            elif direction == "SELL":
                if current_price <= tp:
                    pips = (entry - tp) / _PIP
                    pnl  = pips * 10 * lot
                    db.close_trade(trade_id, tp, round(pnl, 2), round(pips, 1), "win")
                    self._balance += pnl
                    self._open_position = None
                    logger.info("Trade {} CLOSED WIN @ {} pnl=+${:.2f}", trade_id[:8], tp, pnl)
                    self._notifier.send(f"✅ WIN +${pnl:.2f} ({pips:.1f} pips)")
                elif current_price >= sl:
                    pips = (sl - entry) / _PIP
                    pnl  = -(pips * 10 * lot)
                    db.close_trade(trade_id, sl, round(pnl, 2), round(-pips, 1), "loss")
                    self._balance += pnl
                    self._open_position = None
                    logger.info("Trade {} CLOSED LOSS @ {} pnl=-${:.2f}", trade_id[:8], sl, abs(pnl))
                    self._notifier.send(f"❌ LOSS -${abs(pnl):.2f} ({pips:.1f} pips)")
                else:
                    # ── 3. Breakeven — only when SL is still above entry ──
                    if sl > entry:
                        sl_risk_pips = (sl - entry) / _PIP
                        be_trigger = entry - sl_risk_pips * 0.5 * _PIP
                        if current_price <= be_trigger:
                            breakeven_sl = round(entry - 2 * _PIP, 5)
                            db.update_trade(trade_id, {"stop_loss": breakeven_sl})
                            if self._open_position and self._open_position.get("trade_id") == trade_id:
                                self._open_position["sl"] = breakeven_sl
                            logger.info(
                                "Trade {} SL moved to breakeven {:.5f}", trade_id[:8], breakeven_sl
                            )
                            self._notifier.send("🔒 SL moved to breakeven — trade protected")

                    unrealized = (entry - current_price) / _PIP
                    logger.info(
                        "Trade {} OPEN | price={:.5f} unrealized={:.1f}pips",
                        trade_id[:8], current_price, unrealized,
                    )

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
        db    = get_db()
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
