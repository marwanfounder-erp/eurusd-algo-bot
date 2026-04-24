"""Paper trading data feed backed by yfinance real-time EURUSD data.

Used automatically when MetaTrader5 is unavailable (Linux / Railway).

Account state is persisted to logs/paper_account.json so the simulated
balance survives container restarts.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

_SYMBOL_YF = "EURUSD=X"          # Yahoo Finance ticker for EUR/USD
_SPREAD_PIP = 0.6                 # half-spread each side in pips
_PIP = 0.0001
_STATE_FILE = Path("logs/paper_account.json")
_INITIAL_BALANCE = 10_000.0


def _load_state() -> dict[str, Any]:
    """Load persisted paper account state, or return defaults."""
    if _STATE_FILE.exists():
        try:
            with open(_STATE_FILE, encoding="utf-8") as f:
                state = json.load(f)
            logger.info(
                "Paper account loaded | balance=${:.2f} trades={}",
                state.get("balance", _INITIAL_BALANCE),
                state.get("total_trades", 0),
            )
            return state
        except Exception as exc:
            logger.warning("Could not load paper account state: {} — resetting", exc)
    return {
        "balance": _INITIAL_BALANCE,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "open_position": None,   # dict | None
        "trades": [],
    }


def _save_state(state: dict[str, Any]) -> None:
    """Persist account state to disk."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(_STATE_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, _STATE_FILE)  # atomic write


class PaperFeed:
    """yfinance-backed paper trading data feed.

    Implements the same interface as DataFeed so TradingBot, strategy, and
    risk manager all work without modification.
    """

    def __init__(self) -> None:
        self._state = _load_state()
        self._ticker = yf.Ticker(_SYMBOL_YF)
        logger.info("PaperFeed initialised | symbol={}", _SYMBOL_YF)

    # ------------------------------------------------------------------
    # DataFeed compatibility shims
    # ------------------------------------------------------------------

    def initialize_mt5(self) -> bool:          # pragma: no cover
        """Always succeeds — no MT5 needed in paper mode."""
        return True

    def shutdown(self) -> None:
        """Persist state on graceful exit."""
        _save_state(self._state)
        logger.info("PaperFeed shut down — state saved")

    def ensure_connected(self) -> None:
        """No-op — yfinance is stateless."""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

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
        """Return real H1 OHLCV data from Yahoo Finance.

        *timeframe* is ignored — always returns H1 bars.
        *count* is capped at 730 days (Yahoo Finance H1 limit).
        """
        # Yahoo Finance allows up to ~730 days of 1h data
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

        # Ensure UTC-aware index
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
        bal = self._state["balance"]
        return {
            "balance": bal,
            "equity": bal,
            "margin": 0.0,
            "free_margin": bal,
            "profit": bal - _INITIAL_BALANCE,
            "margin_level": 0.0,
        }

    # ------------------------------------------------------------------
    # Position simulation (called by PaperExecutor)
    # ------------------------------------------------------------------

    def open_position(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        sl_pips: float,
    ) -> dict[str, Any]:
        """Record a simulated open position."""
        pos = {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "sl_pips": sl_pips,
            "opened_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._state["open_position"] = pos
        _save_state(self._state)
        logger.info(
            "PaperFeed | position opened {} {:.2f}lots @ {:.5f} sl={:.5f} tp={:.5f}",
            direction, lot, entry, sl, tp,
        )
        return pos

    def close_position(self, exit_price: float, outcome: str) -> dict[str, Any]:
        """Close the open position and update P&L."""
        pos = self._state.get("open_position")
        if pos is None:
            logger.warning("close_position called but no open position")
            return {}

        _PIP_VALUE = 10.0  # $10 per pip per lot for EURUSD standard lot
        direction = pos["direction"]
        entry = pos["entry"]
        lot = pos["lot"]
        sl_pips = pos["sl_pips"]
        risk_amt = self._state["balance"] * 0.01  # 1% risk per trade

        if outcome == "win":
            from config import settings
            pnl = risk_amt * settings.rr_ratio
        elif outcome == "loss":
            pnl = -risk_amt
        else:
            pnl_pips = (
                (exit_price - entry) if direction == "BUY" else (entry - exit_price)
            ) / _PIP
            pnl = pnl_pips * _PIP_VALUE * lot

        self._state["balance"] += pnl
        self._state["total_trades"] = self._state.get("total_trades", 0) + 1
        if pnl > 0:
            self._state["wins"] = self._state.get("wins", 0) + 1
            self._state["gross_profit"] = self._state.get("gross_profit", 0.0) + pnl
        else:
            self._state["losses"] = self._state.get("losses", 0) + 1
            self._state["gross_loss"] = self._state.get("gross_loss", 0.0) + abs(pnl)

        trade_record = {**pos, "exit": exit_price, "pnl": round(pnl, 2), "outcome": outcome}
        self._state.setdefault("trades", []).append(trade_record)
        self._state["open_position"] = None
        _save_state(self._state)

        logger.info(
            "PaperFeed | position closed {} @ {:.5f} pnl=${:.2f} balance=${:.2f}",
            outcome.upper(), exit_price, pnl, self._state["balance"],
        )
        return trade_record

    def get_open_position(self) -> dict[str, Any] | None:
        """Return the current open position or None."""
        return self._state.get("open_position")

    # ------------------------------------------------------------------
    # Account summary helper
    # ------------------------------------------------------------------

    def account_summary(self) -> str:
        """Human-readable paper account summary."""
        s = self._state
        bal = s["balance"]
        ret = (bal - _INITIAL_BALANCE) / _INITIAL_BALANCE
        wins = s.get("wins", 0)
        losses = s.get("losses", 0)
        total = wins + losses
        wr = wins / total if total else 0.0
        return (
            f"Balance: ${bal:,.2f} ({ret:+.2%}) | "
            f"Trades: {total} | WR: {wr:.1%} | "
            f"Wins: {wins} Losses: {losses}"
        )
