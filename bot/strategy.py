"""London Breakout strategy with RSI confirmation and 1:2 R:R."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

try:
    import talib  # type: ignore[import-untyped]
    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available — using pure-pandas RSI fallback")

from bot.data_feed import DataFeed, TIMEFRAME_H1, TIMEFRAME_M5
from config import settings

Direction = Literal["BUY", "SELL", "NONE"]
Confidence = Literal["low", "medium", "high"]


def _rsi_pandas(closes: pd.Series, period: int = 14) -> pd.Series:
    """Pure-pandas RSI implementation (fallback when TA-Lib is absent).

    Handles edge cases:
      loss == 0 (pure uptrend)  → RSI = 100
      gain == 0 (pure downtrend) → RSI = 0
    """
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    # Safe division: where loss=0 treat RS as infinity → RSI=100
    rs = gain / loss.where(loss != 0, other=np.nan)
    rsi = 100 - (100 / (1 + rs))
    # loss=0 means overbought (RSI=100); gain=0 means oversold (RSI=0)
    rsi = rsi.where(loss != 0, other=100.0)
    rsi = rsi.where(gain != 0, other=0.0)
    return rsi


class LondonBreakoutStrategy:
    """
    Strategy logic:

    1. During Asian session (00:00–07:00 UTC) identify the high/low range.
    2. Validate range is 10–50 pips.
    3. During London session (07:00–10:00 UTC) watch for a breakout with a
       2-pip buffer above/below the range.
    4. RSI(14) on H1 must confirm: >50 for BUY, <50 for SELL.
    5. SL = opposite side of range; TP = entry ± (risk × 2) → 1:2 R:R.
    """

    def __init__(self, data_feed: DataFeed) -> None:
        self._feed = data_feed

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_london_session(dt: datetime | None = None) -> bool:
        """Return True if *dt* (UTC) falls within the London breakout window."""
        now = dt or datetime.now(tz=timezone.utc)
        return (
            settings.london_session_start_utc
            <= now.hour
            < settings.london_session_end_utc
        )

    # ------------------------------------------------------------------
    # Range calculation
    # ------------------------------------------------------------------

    def calculate_asian_range(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate the Asian-session high/low range from OHLCV data.

        Parameters
        ----------
        df: H1 DataFrame indexed by UTC datetime with columns open/high/low/close.

        Returns
        -------
        dict with keys: high, low, range_pips (rounded to 1 dp).
        """
        # Filter bars that fall entirely within today's Asian session
        today = datetime.now(tz=timezone.utc).date()
        asian = df[
            (df.index.date == today)  # type: ignore[attr-defined]
            & (df.index.hour >= settings.asian_session_start_utc)  # type: ignore[attr-defined]
            & (df.index.hour < settings.asian_session_end_utc)  # type: ignore[attr-defined]
        ]

        if asian.empty:
            logger.warning("No Asian-session bars found for today")
            return {"high": 0.0, "low": 0.0, "range_pips": 0.0}

        high = float(asian["high"].max())
        low = float(asian["low"].min())
        _pip = 0.01 if settings.symbol.upper().endswith("JPY") else 0.0001
        range_pips = round((high - low) / _pip, 1)

        logger.info(
            "Asian range | high={:.5f} low={:.5f} range={:.1f}pips",
            high, low, range_pips,
        )
        return {"high": high, "low": low, "range_pips": range_pips}

    def is_valid_range(self, range_pips: float) -> bool:
        """Return True if the Asian range is within the acceptable band."""
        valid = settings.min_range_pips <= range_pips <= settings.max_range_pips
        logger.debug(
            "Range validity | {:.1f}pips in [{},{}]pips → {}",
            range_pips, settings.min_range_pips, settings.max_range_pips, valid,
        )
        return valid

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def _get_rsi(self, df: pd.DataFrame) -> float:
        """Return the latest RSI value from the DataFrame's close prices."""
        closes = df["close"].astype(float)
        if len(closes) < settings.rsi_period + 1:
            logger.warning("Insufficient bars for RSI — returning 50 (neutral)")
            return 50.0

        if _TALIB_AVAILABLE:
            rsi_series = talib.RSI(closes.values, timeperiod=settings.rsi_period)
        else:
            rsi_series = _rsi_pandas(closes, period=settings.rsi_period).values  # type: ignore[assignment]

        latest = float(rsi_series[-1])
        logger.debug("RSI({}) = {:.2f}", settings.rsi_period, latest)
        return latest

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def get_signal(self, symbol: str) -> dict[str, object]:
        """Evaluate the London Breakout setup and return a trade signal.

        Returns
        -------
        dict with keys:
            direction  – BUY | SELL | NONE
            entry      – float (ask for BUY, bid for SELL, 0 for NONE)
            stop_loss  – float
            take_profit– float
            range_pips – float
            confidence – low | medium | high
        """
        _null: dict[str, object] = {
            "direction": "NONE",
            "entry": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "range_pips": 0.0,
            "confidence": "low",
        }

        # 1. Must be inside London session
        if not self.is_london_session():
            logger.debug("Outside London session — no signal")
            return _null

        # 2. Fetch H1 candles for Asian range + RSI
        try:
            h1_df = self._feed.get_candles(symbol, TIMEFRAME_H1, count=200)
        except Exception as exc:
            logger.error("get_signal — H1 fetch failed: {}", exc)
            return _null

        # 3. Asian range
        asian = self.calculate_asian_range(h1_df)
        range_pips = asian["range_pips"]
        if not self.is_valid_range(range_pips):
            logger.info("Asian range {:.1f}pips outside valid band — no signal", range_pips)
            return _null

        asian_high = asian["high"]
        asian_low = asian["low"]

        # 4. Current price
        try:
            tick = self._feed.get_tick(symbol)
        except Exception as exc:
            logger.error("get_signal — tick fetch failed: {}", exc)
            return _null

        ask = tick["ask"]
        bid = tick["bid"]

        # pip size
        pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
        buffer = settings.breakout_buffer_pips * pip

        buy_trigger = asian_high + buffer
        sell_trigger = asian_low - buffer

        # 5. Detect breakout direction
        direction: Direction = "NONE"
        if ask >= buy_trigger:
            direction = "BUY"
        elif bid <= sell_trigger:
            direction = "SELL"

        if direction == "NONE":
            logger.debug(
                "No breakout | ask={:.5f} buy_trigger={:.5f} bid={:.5f} sell_trigger={:.5f}",
                ask, buy_trigger, bid, sell_trigger,
            )
            return _null

        # 6. RSI confirmation
        rsi = self._get_rsi(h1_df)
        if direction == "BUY" and rsi < 50:
            logger.info("BUY signal rejected — RSI={:.2f} < 50", rsi)
            return _null
        if direction == "SELL" and rsi > 50:
            logger.info("SELL signal rejected — RSI={:.2f} > 50", rsi)
            return _null

        # 7. Build levels
        if direction == "BUY":
            entry = ask
            stop_loss = asian_low - buffer          # below Asian low
            risk_distance = entry - stop_loss
            take_profit = entry + risk_distance * settings.rr_ratio
        else:  # SELL
            entry = bid
            stop_loss = asian_high + buffer         # above Asian high
            risk_distance = stop_loss - entry
            take_profit = entry - risk_distance * settings.rr_ratio

        # 8. Confidence rating (based on RSI extremity)
        confidence: Confidence
        if direction == "BUY":
            if rsi >= 65:
                confidence = "high"
            elif rsi >= 55:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            if rsi <= 35:
                confidence = "high"
            elif rsi <= 45:
                confidence = "medium"
            else:
                confidence = "low"

        signal: dict[str, object] = {
            "direction": direction,
            "entry": round(entry, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "range_pips": range_pips,
            "confidence": confidence,
            "rsi": round(rsi, 2),
            "asian_high": asian_high,
            "asian_low": asian_low,
        }
        logger.info(
            "Signal | {} entry={} sl={} tp={} range={:.1f}pips rsi={:.2f} conf={}",
            direction, entry, stop_loss, take_profit, range_pips, rsi, confidence,
        )
        return signal
