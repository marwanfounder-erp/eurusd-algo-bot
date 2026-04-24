"""MT5 data feed — OHLCV candles, ticks, and account info."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]

from config import settings

_TIMEFRAME_MAP: dict[str, Any] = {}

# Populate map lazily so import never fails if mt5 is absent
def _tf(name: str) -> Any:
    if mt5 is None:
        return None
    return getattr(mt5, name, None)


TIMEFRAME_M1 = "TIMEFRAME_M1"
TIMEFRAME_M5 = "TIMEFRAME_M5"
TIMEFRAME_M15 = "TIMEFRAME_M15"
TIMEFRAME_M30 = "TIMEFRAME_M30"
TIMEFRAME_H1 = "TIMEFRAME_H1"
TIMEFRAME_H4 = "TIMEFRAME_H4"
TIMEFRAME_D1 = "TIMEFRAME_D1"


class DataFeedError(RuntimeError):
    """Raised when MT5 data cannot be retrieved."""


class DataFeed:
    """Thin wrapper around MetaTrader5 for market data access."""

    _RECONNECT_ATTEMPTS: int = 5
    _RECONNECT_DELAY: float = 2.0

    def __init__(self) -> None:
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def initialize_mt5(self) -> bool:
        """Connect to MT5 terminal using credentials from settings."""
        if mt5 is None:
            logger.error("MetaTrader5 package not installed")
            return False

        logger.info(
            "Connecting to MT5 | login={} server={}",
            settings.mt5_login,
            settings.mt5_server,
        )

        # Only pass login/password/server when credentials are configured
        init_kwargs: dict[str, object] = {}
        if settings.mt5_login is not None:
            init_kwargs["login"] = settings.mt5_login
        if settings.mt5_password:
            init_kwargs["password"] = settings.mt5_password
        if settings.mt5_server:
            init_kwargs["server"] = settings.mt5_server

        ok = mt5.initialize(**init_kwargs)
        if not ok:
            code, msg = mt5.last_error()
            logger.error("MT5 init failed | code={} msg={}", code, msg)
            self._connected = False
            return False

        info = mt5.account_info()
        logger.info(
            "MT5 connected | account={} balance={:.2f} {}",
            info.login,
            info.balance,
            info.currency,
        )
        self._connected = True
        return True

    def shutdown(self) -> None:
        """Gracefully disconnect from MT5."""
        if mt5 is not None and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def ensure_connected(self) -> None:
        """Attempt reconnection if MT5 dropped the connection."""
        if mt5 is None:
            raise DataFeedError("MetaTrader5 package not available")

        if mt5.terminal_info() is None:
            logger.warning("MT5 connection lost — attempting reconnect")
            self._connected = False
            for attempt in range(1, self._RECONNECT_ATTEMPTS + 1):
                logger.info("Reconnect attempt {}/{}", attempt, self._RECONNECT_ATTEMPTS)
                if self.initialize_mt5():
                    logger.info("Reconnected on attempt {}", attempt)
                    return
                time.sleep(self._RECONNECT_DELAY)
            raise DataFeedError(
                f"Could not reconnect to MT5 after {self._RECONNECT_ATTEMPTS} attempts"
            )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
    ) -> pd.DataFrame:
        """Return the last *count* OHLCV candles as a DataFrame.

        Parameters
        ----------
        symbol:    Broker symbol string, e.g. "EURUSD".
        timeframe: One of the TIMEFRAME_* constants defined in this module.
        count:     Number of bars to retrieve (most recent).

        Returns
        -------
        DataFrame with columns: datetime (UTC, tz-aware), open, high, low,
        close, volume.  Raises DataFeedError on failure.
        """
        self.ensure_connected()

        tf_attr = timeframe  # e.g. "TIMEFRAME_H1"
        mt5_tf = getattr(mt5, tf_attr, None)
        if mt5_tf is None:
            raise DataFeedError(f"Unknown timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            logger.error(
                "get_candles failed | symbol={} tf={} code={} msg={}",
                symbol, timeframe, code, msg,
            )
            raise DataFeedError(
                f"No data returned for {symbol}/{timeframe}: {msg}"
            )

        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            }
        )[["datetime", "open", "high", "low", "close", "volume"]]
        df = df.set_index("datetime").sort_index()
        logger.debug(
            "Retrieved {} candles | symbol={} tf={}", len(df), symbol, timeframe
        )
        return df

    def get_tick(self, symbol: str) -> dict[str, float]:
        """Return current bid, ask, and spread in pips for *symbol*.

        Returns
        -------
        dict with keys: bid, ask, spread_pips, time (UTC datetime).
        """
        self.ensure_connected()

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            code, msg = mt5.last_error()
            logger.error(
                "get_tick failed | symbol={} code={} msg={}", symbol, code, msg
            )
            raise DataFeedError(f"Tick unavailable for {symbol}: {msg}")

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise DataFeedError(f"symbol_info unavailable for {symbol}")

        # For most FX pairs 1 pip = 0.0001 (4-digit), spread in pips
        pip_size: float = 10 * sym_info.point  # point * 10 = 1 pip
        spread_pips: float = (tick.ask - tick.bid) / pip_size

        result = {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_pips": round(spread_pips, 1),
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }
        logger.debug(
            "Tick | symbol={} bid={} ask={} spread={:.1f}pips",
            symbol, tick.bid, tick.ask, spread_pips,
        )
        return result

    def get_account_info(self) -> dict[str, float]:
        """Return current account snapshot.

        Returns
        -------
        dict with keys: balance, equity, margin, free_margin, profit,
        margin_level (%).
        """
        self.ensure_connected()

        info = mt5.account_info()
        if info is None:
            code, msg = mt5.last_error()
            logger.error("get_account_info failed | code={} msg={}", code, msg)
            raise DataFeedError(f"account_info unavailable: {msg}")

        result = {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "margin_level": info.margin_level,
        }
        logger.debug(
            "Account | balance={:.2f} equity={:.2f} profit={:.2f}",
            info.balance, info.equity, info.profit,
        )
        return result
