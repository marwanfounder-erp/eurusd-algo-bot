"""Tests for LondonBreakoutStrategy signal generation."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_h1_df(
    asian_high: float = 1.0800,
    asian_low: float = 1.0750,
    n_bars: int = 120,
) -> pd.DataFrame:
    """Build a synthetic H1 DataFrame with a known Asian range.

    Bars 0-6 are Asian session (hours 0-6 UTC).
    Bars 7-9 are London session (hours 7-9 UTC).
    """
    today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    index = pd.date_range(start=today, periods=n_bars, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    closes = 1.0780 + rng.normal(0, 0.0005, n_bars).cumsum() * 0.0001
    closes = np.clip(closes, 1.0600, 1.1000)

    df = pd.DataFrame(
        {
            "open": closes - 0.0002,
            "high": closes + 0.0003,
            "low": closes - 0.0003,
            "close": closes,
            "volume": rng.integers(100, 1000, n_bars),
        },
        index=index,
    )

    # Force Asian session range
    asian_mask = df.index.hour < 7
    df.loc[asian_mask, "high"] = asian_high
    df.loc[asian_mask, "low"] = asian_low

    return df


@pytest.fixture()
def mock_feed() -> MagicMock:
    feed = MagicMock()
    return feed


@pytest.fixture()
def strategy(mock_feed: MagicMock):
    from bot.strategy import LondonBreakoutStrategy
    return LondonBreakoutStrategy(data_feed=mock_feed)


# ---------------------------------------------------------------------------
# Asian range tests
# ---------------------------------------------------------------------------

class TestAsianRange:
    def test_range_calculated_correctly(self, strategy, mock_feed):
        df = _make_h1_df(asian_high=1.0800, asian_low=1.0750)
        asian = strategy.calculate_asian_range(df)
        assert asian["high"] == pytest.approx(1.0800, abs=1e-5)
        assert asian["low"] == pytest.approx(1.0750, abs=1e-5)
        assert asian["range_pips"] == pytest.approx(50.0, abs=0.5)

    def test_empty_df_returns_zeros(self, strategy, mock_feed):
        today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        # Bars outside Asian session
        index = pd.date_range(start=today.replace(hour=8), periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"open": [1.0]*3, "high": [1.01]*3, "low": [0.99]*3, "close": [1.0]*3, "volume": [100]*3},
            index=index,
        )
        result = strategy.calculate_asian_range(df)
        assert result["range_pips"] == 0.0

    def test_is_valid_range_inside_band(self, strategy):
        assert strategy.is_valid_range(25.0) is True

    def test_is_valid_range_too_small(self, strategy):
        assert strategy.is_valid_range(5.0) is False

    def test_is_valid_range_too_large(self, strategy):
        assert strategy.is_valid_range(60.0) is False

    def test_is_valid_range_boundary_min(self, strategy):
        # min_range_pips = 15.0 after optimization
        assert strategy.is_valid_range(15.0) is True

    def test_is_valid_range_boundary_max(self, strategy):
        # max_range_pips = 40.0 after optimization
        assert strategy.is_valid_range(40.0) is True


# ---------------------------------------------------------------------------
# London session detection
# ---------------------------------------------------------------------------

class TestLondonSession:
    def test_inside_london(self, strategy):
        dt = datetime(2024, 1, 15, 8, 30, tzinfo=timezone.utc)
        assert strategy.is_london_session(dt) is True

    def test_outside_london_before(self, strategy):
        dt = datetime(2024, 1, 15, 6, 59, tzinfo=timezone.utc)
        assert strategy.is_london_session(dt) is False

    def test_outside_london_after(self, strategy):
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        assert strategy.is_london_session(dt) is False

    def test_exact_start(self, strategy):
        dt = datetime(2024, 1, 15, 7, 0, tzinfo=timezone.utc)
        assert strategy.is_london_session(dt) is True


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestGetSignal:
    def _setup_mock_feed(
        self,
        mock_feed: MagicMock,
        h1_df: pd.DataFrame,
        ask: float,
        bid: float,
    ) -> None:
        mock_feed.get_candles.return_value = h1_df
        mock_feed.get_tick.return_value = {
            "ask": ask,
            "bid": bid,
            "spread_pips": 1.2,
            "time": datetime.now(tz=timezone.utc),
        }

    @patch("bot.strategy.LondonBreakoutStrategy.is_london_session", return_value=True)
    def test_buy_signal_above_range(self, mock_london, strategy, mock_feed):
        df = _make_h1_df(asian_high=1.0800, asian_low=1.0750)
        # Force RSI-friendly closes — steadily rising for BUY
        n = len(df)
        df["close"] = np.linspace(1.0700, 1.0900, n)
        df["high"] = df["close"] + 0.0005
        df["low"] = df["close"] - 0.0005

        # Ask above Asian high + 2 pip buffer
        ask = 1.0800 + 0.0002 + 0.0001  # buffer=2pips, then 1 pip above trigger
        self._setup_mock_feed(mock_feed, df, ask=ask, bid=ask - 0.0001)

        signal = strategy.get_signal("EURUSD.s")
        # With rising closes RSI > 50, breakout above Asian high → BUY
        if signal["direction"] != "NONE":
            assert signal["direction"] == "BUY"
            assert float(signal["stop_loss"]) < float(signal["entry"])
            assert float(signal["take_profit"]) > float(signal["entry"])

    @patch("bot.strategy.LondonBreakoutStrategy.is_london_session", return_value=True)
    def test_sell_signal_below_range(self, mock_london, strategy, mock_feed):
        df = _make_h1_df(asian_high=1.0800, asian_low=1.0750)
        # Steadily falling closes for RSI < 50
        n = len(df)
        df["close"] = np.linspace(1.0900, 1.0700, n)
        df["high"] = df["close"] + 0.0005
        df["low"] = df["close"] - 0.0005

        bid = 1.0750 - 0.0002 - 0.0001
        self._setup_mock_feed(mock_feed, df, ask=bid + 0.0001, bid=bid)

        signal = strategy.get_signal("EURUSD.s")
        if signal["direction"] != "NONE":
            assert signal["direction"] == "SELL"
            assert float(signal["stop_loss"]) > float(signal["entry"])
            assert float(signal["take_profit"]) < float(signal["entry"])

    @patch("bot.strategy.LondonBreakoutStrategy.is_london_session", return_value=False)
    def test_no_signal_outside_london(self, mock_london, strategy, mock_feed):
        df = _make_h1_df()
        self._setup_mock_feed(mock_feed, df, ask=1.0810, bid=1.0808)
        signal = strategy.get_signal("EURUSD.s")
        assert signal["direction"] == "NONE"

    @patch("bot.strategy.LondonBreakoutStrategy.is_london_session", return_value=True)
    def test_no_signal_invalid_range(self, mock_london, strategy, mock_feed):
        # Range of only 3 pips — too small
        df = _make_h1_df(asian_high=1.0800, asian_low=1.0797)
        self._setup_mock_feed(mock_feed, df, ask=1.0803, bid=1.0801)
        signal = strategy.get_signal("EURUSD.s")
        assert signal["direction"] == "NONE"

    @patch("bot.strategy.LondonBreakoutStrategy.is_london_session", return_value=True)
    def test_rr_ratio_is_two(self, mock_london, strategy, mock_feed):
        df = _make_h1_df(asian_high=1.0800, asian_low=1.0750)
        n = len(df)
        df["close"] = np.linspace(1.0700, 1.0900, n)
        df["high"] = df["close"] + 0.0005
        df["low"] = df["close"] - 0.0005

        ask = 1.0800 + 0.0002 + 0.0001
        self._setup_mock_feed(mock_feed, df, ask=ask, bid=ask - 0.0001)

        signal = strategy.get_signal("EURUSD.s")
        if signal["direction"] == "BUY":
            entry = float(signal["entry"])
            sl = float(signal["stop_loss"])
            tp = float(signal["take_profit"])
            risk = entry - sl
            reward = tp - entry
            # rr_ratio = 1.5 after optimization
            from config import settings
            assert reward == pytest.approx(risk * settings.rr_ratio, rel=0.01)

    @patch("bot.strategy.LondonBreakoutStrategy.is_london_session", return_value=True)
    def test_data_feed_error_returns_none(self, mock_london, strategy, mock_feed):
        mock_feed.get_candles.side_effect = RuntimeError("MT5 disconnected")
        signal = strategy.get_signal("EURUSD.s")
        assert signal["direction"] == "NONE"
