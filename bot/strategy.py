"""London Breakout and Silver Bullet strategies."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

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


# ──────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────

def _rsi_pandas(closes: pd.Series, period: int = 14) -> pd.Series:
    """Pure-pandas RSI fallback when TA-Lib is absent.

    Handles edge cases:
      loss == 0 (pure uptrend)  → RSI = 100
      gain == 0 (pure downtrend) → RSI = 0
    """
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.where(loss != 0, other=np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, other=100.0)
    rsi = rsi.where(gain != 0, other=0.0)
    return rsi


def _get_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest RSI value from *df*'s close prices."""
    closes = df["close"].astype(float)
    if len(closes) < period + 1:
        logger.warning("Insufficient bars for RSI — returning 50 (neutral)")
        return 50.0
    if _TALIB_AVAILABLE:
        rsi_series = talib.RSI(closes.values, timeperiod=period)
    else:
        rsi_series = _rsi_pandas(closes, period=period).values  # type: ignore[assignment]
    latest = float(rsi_series[-1])
    logger.debug("RSI({}) = {:.2f}", period, latest)
    return latest


def _confidence(direction: Direction, rsi: float) -> Confidence:
    if direction == "BUY":
        return "high" if rsi >= 65 else "medium" if rsi >= 55 else "low"
    return "high" if rsi <= 35 else "medium" if rsi <= 45 else "low"


# ──────────────────────────────────────────────────────────────────────
# London Breakout
# ──────────────────────────────────────────────────────────────────────

class LondonBreakoutStrategy:
    """
    Strategy logic:

    1. During Asian session (00:00–07:00 UTC) identify the high/low range.
    2. Validate range is 10–50 pips.
    3. During London session (07:00–10:00 UTC) watch for a breakout with a
       2-pip buffer above/below the range.
    4. RSI(14) on H1 must confirm: >50 for BUY, <50 for SELL.
    5. SL = opposite side of range; TP = entry ± (risk × rr_ratio) → 1:1.5 R:R.
    """

    def __init__(self, data_feed: DataFeed) -> None:
        self._feed = data_feed

    @staticmethod
    def is_market_open(dt: datetime | None = None) -> bool:
        """Return False on weekends when forex is closed."""
        now = dt or datetime.now(tz=timezone.utc)
        if now.weekday() == 5:
            return False
        if now.weekday() == 6 and now.hour < 21:
            return False
        return True

    @staticmethod
    def is_london_session(dt: datetime | None = None) -> bool:
        """Return True if *dt* (UTC) falls within the London breakout window."""
        now = dt or datetime.now(tz=timezone.utc)
        return (
            settings.london_session_start_utc
            <= now.hour
            < settings.london_session_end_utc
        )

    def calculate_asian_range(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate the Asian-session high/low range from OHLCV data."""
        today = datetime.now(tz=timezone.utc).date()
        asian = df[
            (df.index.date == today)  # type: ignore[attr-defined]
            & (df.index.hour >= settings.asian_session_start_utc)  # type: ignore[attr-defined]
            & (df.index.hour < settings.asian_session_end_utc)  # type: ignore[attr-defined]
        ]
        if asian.empty:
            logger.debug("No Asian-session bars found for today")
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
        valid = settings.min_range_pips <= range_pips <= settings.max_range_pips
        logger.debug(
            "Range validity | {:.1f}pips in [{},{}]pips → {}",
            range_pips, settings.min_range_pips, settings.max_range_pips, valid,
        )
        return valid

    def get_signal(self, symbol: str) -> dict[str, object]:
        """Evaluate the London Breakout setup and return a trade signal."""
        _null: dict[str, object] = {
            "direction": "NONE",
            "entry": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "range_pips": 0.0,
            "confidence": "low",
            "strategy": "LB",
        }

        if not self.is_market_open():
            return {**_null, "reason": "market_closed"}

        if not self.is_london_session():
            logger.debug("Outside London session — no signal")
            return _null

        try:
            h1_df = self._feed.get_candles(symbol, TIMEFRAME_H1, count=200)
        except Exception as exc:
            logger.error("LB get_signal — H1 fetch failed: {}", exc)
            return _null

        asian = self.calculate_asian_range(h1_df)
        range_pips = asian["range_pips"]
        if not self.is_valid_range(range_pips):
            logger.info("Asian range {:.1f}pips outside valid band — no signal", range_pips)
            return _null

        asian_high = asian["high"]
        asian_low = asian["low"]

        try:
            tick = self._feed.get_tick(symbol)
        except Exception as exc:
            logger.error("LB get_signal — tick fetch failed: {}", exc)
            return _null

        ask = tick["ask"]
        bid = tick["bid"]
        pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
        buffer = settings.breakout_buffer_pips * pip

        direction: Direction = "NONE"
        if ask >= asian_high + buffer:
            direction = "BUY"
        elif bid <= asian_low - buffer:
            direction = "SELL"

        if direction == "NONE":
            logger.debug(
                "No breakout | ask={:.5f} buy_trigger={:.5f} bid={:.5f} sell_trigger={:.5f}",
                ask, asian_high + buffer, bid, asian_low - buffer,
            )
            return _null

        rsi = _get_rsi(h1_df, settings.rsi_period)
        if direction == "BUY" and rsi < 50:
            logger.info("LB BUY rejected — RSI={:.2f} < 50", rsi)
            return _null
        if direction == "SELL" and rsi > 50:
            logger.info("LB SELL rejected — RSI={:.2f} > 50", rsi)
            return _null

        if direction == "BUY":
            entry = ask
            stop_loss = asian_low - buffer
            risk_distance = entry - stop_loss
            take_profit = entry + risk_distance * settings.rr_ratio
        else:
            entry = bid
            stop_loss = asian_high + buffer
            risk_distance = stop_loss - entry
            take_profit = entry - risk_distance * settings.rr_ratio

        conf = _confidence(direction, rsi)
        signal: dict[str, object] = {
            "direction": direction,
            "entry": round(entry, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "range_pips": range_pips,
            "confidence": conf,
            "rsi": round(rsi, 2),
            "asian_high": asian_high,
            "asian_low": asian_low,
            "strategy": "LB",
        }
        logger.info(
            "LB Signal | {} entry={} sl={} tp={} range={:.1f}pips rsi={:.2f} conf={}",
            direction, entry, stop_loss, take_profit, range_pips, rsi, conf,
        )
        return signal


# ──────────────────────────────────────────────────────────────────────
# Silver Bullet
# ──────────────────────────────────────────────────────────────────────

class SilverBulletStrategy:
    """ICT Silver Bullet — FVG entry during three intraday windows.

    Windows (UTC):
      03:00–04:00  Silver Bullet 1 (07:00–08:00 Sharjah)
      10:00–11:00  Silver Bullet 2 (14:00–15:00 Sharjah)
      14:00–15:00  Silver Bullet 3 (18:00–19:00 Sharjah)

    Logic:
      1. Fetch 50 M5 bars during the active window.
      2. Scan last 15 bars for the most recent Fair Value Gap (3-candle imbalance).
      3. Current price must be inside the FVG.
      4. RSI(14) on M5 must confirm: >50 for BUY, <50 for SELL.
      5. SL: beyond FVG boundary + buffer; TP: entry ± risk × rr_ratio.
    """

    _WINDOWS = [
        ("SB1", "sb_window1_start_utc", "sb_window1_end_utc"),
        ("SB2", "sb_window2_start_utc", "sb_window2_end_utc"),
        ("SB3", "sb_window3_start_utc", "sb_window3_end_utc"),
    ]

    def __init__(self, data_feed: DataFeed) -> None:
        self._feed = data_feed

    @staticmethod
    def is_market_open(dt: datetime | None = None) -> bool:
        return LondonBreakoutStrategy.is_market_open(dt)

    @staticmethod
    def is_silver_bullet_window(dt: datetime | None = None) -> bool:
        now = dt or datetime.now(tz=timezone.utc)
        windows = [
            (settings.sb_window1_start_utc, settings.sb_window1_end_utc),
            (settings.sb_window2_start_utc, settings.sb_window2_end_utc),
            (settings.sb_window3_start_utc, settings.sb_window3_end_utc),
        ]
        return any(start <= now.hour < end for start, end in windows)

    @staticmethod
    def current_window_name(dt: datetime | None = None) -> str:
        now = dt or datetime.now(tz=timezone.utc)
        windows = [
            (settings.sb_window1_start_utc, settings.sb_window1_end_utc, "SB1"),
            (settings.sb_window2_start_utc, settings.sb_window2_end_utc, "SB2"),
            (settings.sb_window3_start_utc, settings.sb_window3_end_utc, "SB3"),
        ]
        for start, end, name in windows:
            if start <= now.hour < end:
                return name
        return "SB"

    @staticmethod
    def _find_fvg(df: pd.DataFrame, pip: float = 0.0001) -> dict[str, Any] | None:
        """Return the most recent Fair Value Gap from *df*, or None."""
        bars = df.tail(15)
        if len(bars) < 3:
            return None
        fvgs: list[dict[str, Any]] = []
        for i in range(2, len(bars)):
            c1 = bars.iloc[i - 2]
            c3 = bars.iloc[i]
            # Bullish FVG: gap between c1.high and c3.low
            if c1["high"] < c3["low"]:
                gap_pips = (c3["low"] - c1["high"]) / pip
                if gap_pips >= 0.5:
                    fvgs.append({
                        "type": "bullish",
                        "top": float(c3["low"]),
                        "bottom": float(c1["high"]),
                    })
            # Bearish FVG: gap between c3.high and c1.low
            if c1["low"] > c3["high"]:
                gap_pips = (c1["low"] - c3["high"]) / pip
                if gap_pips >= 0.5:
                    fvgs.append({
                        "type": "bearish",
                        "top": float(c1["low"]),
                        "bottom": float(c3["high"]),
                    })
        return fvgs[-1] if fvgs else None

    def get_signal(self, symbol: str) -> dict[str, object]:
        """Evaluate the Silver Bullet setup and return a trade signal."""
        _null: dict[str, object] = {
            "direction": "NONE",
            "entry": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "range_pips": 0.0,
            "confidence": "low",
            "strategy": "SB",
        }

        if not self.is_market_open():
            return {**_null, "reason": "market_closed"}

        if not self.is_silver_bullet_window():
            return _null

        try:
            m5_df = self._feed.get_candles(symbol, TIMEFRAME_M5, count=50)
        except Exception as exc:
            logger.error("SB get_signal — M5 fetch failed: {}", exc)
            return _null

        pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
        fvg = self._find_fvg(m5_df, pip)
        if fvg is None:
            logger.debug("SilverBullet: no FVG found in recent M5 bars")
            return _null

        try:
            tick = self._feed.get_tick(symbol)
        except Exception as exc:
            logger.error("SB get_signal — tick fetch failed: {}", exc)
            return _null

        ask = tick["ask"]
        bid = tick["bid"]
        buffer = settings.breakout_buffer_pips * pip

        direction: Direction = "NONE"
        if fvg["type"] == "bullish" and fvg["bottom"] <= ask <= fvg["top"]:
            direction = "BUY"
        elif fvg["type"] == "bearish" and fvg["bottom"] <= bid <= fvg["top"]:
            direction = "SELL"

        if direction == "NONE":
            logger.debug(
                "SB: price not in FVG | {} [{:.5f}–{:.5f}] ask={:.5f} bid={:.5f}",
                fvg["type"], fvg["bottom"], fvg["top"], ask, bid,
            )
            return _null

        rsi = _get_rsi(m5_df, settings.rsi_period)
        if direction == "BUY" and rsi < 50:
            logger.info("SB BUY rejected — RSI={:.2f} < 50", rsi)
            return _null
        if direction == "SELL" and rsi > 50:
            logger.info("SB SELL rejected — RSI={:.2f} > 50", rsi)
            return _null

        if direction == "BUY":
            entry = ask
            stop_loss = fvg["bottom"] - buffer
            risk_distance = entry - stop_loss
            take_profit = entry + risk_distance * settings.rr_ratio
        else:
            entry = bid
            stop_loss = fvg["top"] + buffer
            risk_distance = stop_loss - entry
            take_profit = entry - risk_distance * settings.rr_ratio

        conf = _confidence(direction, rsi)
        fvg_size_pips = (fvg["top"] - fvg["bottom"]) / pip
        window_name = self.current_window_name()

        signal: dict[str, object] = {
            "direction": direction,
            "entry": round(entry, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "range_pips": round(fvg_size_pips, 1),
            "confidence": conf,
            "rsi": round(rsi, 2),
            "fvg_top": fvg["top"],
            "fvg_bottom": fvg["bottom"],
            "strategy": "SB",
            "window": window_name,
        }
        logger.info(
            "SB Signal | {} window={} entry={} sl={} tp={} fvg=[{:.5f}–{:.5f}] rsi={:.2f} conf={}",
            direction, window_name, entry, stop_loss, take_profit,
            fvg["bottom"], fvg["top"], rsi, conf,
        )
        return signal
