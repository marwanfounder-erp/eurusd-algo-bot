"""
Run the bot in paper mode with a fully mocked MT5 terminal.

Usage:
    python run_paper_mock.py

This script installs a realistic MetaTrader5 mock into sys.modules BEFORE
importing any bot code, so the full pipeline (strategy → risk → notifier)
executes without a live MT5 terminal.  Three ticks are simulated:
  tick 1 — London session, valid Asian range, RSI > 50 → BUY signal
  tick 2 — London session, position already open   → skipped
  tick 3 — outside London session                  → no signal
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, date, timezone
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# 1. Build and install the MetaTrader5 mock BEFORE any bot import
# ──────────────────────────────────────────────────────────────────────

def _build_mt5_mock() -> ModuleType:
    mt5 = MagicMock(name="MetaTrader5")

    # Constants
    mt5.TIMEFRAME_M1 = 1
    mt5.TIMEFRAME_M5 = 5
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_M30 = 30
    mt5.TIMEFRAME_H1 = 16385
    mt5.TIMEFRAME_H4 = 16388
    mt5.TIMEFRAME_D1 = 16408
    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.TRADE_ACTION_DEAL = 1
    mt5.ORDER_FILLING_FOK = 0
    mt5.ORDER_TIME_GTC = 0
    mt5.TRADE_RETCODE_DONE = 10009

    # Connection
    mt5.initialize.return_value = True
    mt5.terminal_info.return_value = MagicMock()
    mt5.last_error.return_value = (0, "")
    mt5.shutdown.return_value = None

    # Account: $10,000 balance
    acc = MagicMock()
    acc.login = 1234567
    acc.balance = 10_000.0
    acc.equity  = 10_000.0
    acc.margin  = 0.0
    acc.margin_free = 10_000.0
    acc.profit  = 0.0
    acc.margin_level = 0.0
    acc.currency = "USD"
    mt5.account_info.return_value = acc

    # Symbol info: EURUSD standard account
    sym = MagicMock()
    sym.point = 0.00001
    sym.trade_contract_size = 100_000.0
    mt5.symbol_info.return_value = sym

    # Tick: price above Asian high (triggers BUY)
    tick = MagicMock()
    tick.bid = 1.08065
    tick.ask = 1.08078
    tick.time = int(datetime.now(tz=timezone.utc).timestamp())
    mt5.symbol_info_tick.return_value = tick

    # H1 candles — 120 bars with a clear Asian range 1.0750–1.0800 (50 pips)
    # and rising closes so RSI > 50
    today = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    index = pd.date_range(start=today, periods=120, freq="h", tz="UTC")

    # Steadily rising closes to produce RSI > 50
    closes = np.linspace(1.0700, 1.0830, 120)
    opens  = closes - 0.0002
    highs  = closes + 0.0003
    lows   = closes - 0.0003

    # Force Asian session bars (hours 0-6) to span exactly 1.0750–1.0800
    asian_mask = index.hour < 7
    highs_arr = highs.copy()
    lows_arr  = lows.copy()
    highs_arr[asian_mask] = 1.0800
    lows_arr[asian_mask]  = 1.0750

    # MT5 returns structured array with 'time' as Unix int
    times_unix = (index.astype("int64") // 10**9).tolist()

    records = [
        {
            "time": t,
            "open": o, "high": h, "low": l, "close": c,
            "tick_volume": 500, "spread": 1, "real_volume": 0,
        }
        for t, o, h, l, c in zip(
            times_unix,
            opens.tolist(),
            highs_arr.tolist(),
            lows_arr.tolist(),
            closes.tolist(),
        )
    ]
    import numpy.lib.recfunctions as rfn  # noqa: F401

    dtype = np.dtype([
        ("time", "i8"), ("open", "f8"), ("high", "f8"),
        ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
        ("spread", "i4"), ("real_volume", "i8"),
    ])
    rates_array = np.array(
        [(r["time"], r["open"], r["high"], r["low"], r["close"],
          r["tick_volume"], r["spread"], r["real_volume"])
         for r in records],
        dtype=dtype,
    )
    mt5.copy_rates_from_pos.return_value = rates_array

    # No open positions on tick 1
    mt5.positions_get.return_value = []

    return mt5  # type: ignore[return-value]


_mt5_mock = _build_mt5_mock()
sys.modules["MetaTrader5"] = _mt5_mock  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# 2. Now safe to import bot code
# ──────────────────────────────────────────────────────────────────────

from bot.data_feed import DataFeed
from bot.executor import OrderExecutor
from bot.news_filter import NewsFilter
from bot.notifier import TelegramNotifier
from bot.risk_manager import RiskManager
from bot.strategy import LondonBreakoutStrategy
from config import settings
from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# 3. Patch strategy to think it's inside London session for all ticks
# ──────────────────────────────────────────────────────────────────────

class _LondonAlwaysOpen(LondonBreakoutStrategy):
    @staticmethod
    def is_london_session(dt: datetime | None = None) -> bool:  # type: ignore[override]
        return True


# ──────────────────────────────────────────────────────────────────────
# 4. Patch news filter to report no events (don't block trading)
# ──────────────────────────────────────────────────────────────────────

class _QuietNewsFilter(NewsFilter):
    def high_impact_soon(self) -> bool:
        logger.debug("NewsFilter (mock) — no events nearby")
        return False

    def get_next_event(self) -> dict:
        return {}


# ──────────────────────────────────────────────────────────────────────
# 5. Run 3 demo ticks manually
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  EUR/USD ALGO BOT — PAPER MODE DEMO (MT5 mocked)")
    logger.info("  Symbol : {} | Risk : {:.0%} | Max daily loss : {:.0%}",
                settings.symbol, settings.risk_per_trade, settings.max_daily_loss)
    logger.info("=" * 60)

    feed     = DataFeed()
    feed.initialize_mt5()

    executor = OrderExecutor(feed)
    risk     = RiskManager(feed, executor)
    strategy = _LondonAlwaysOpen(feed)
    news     = _QuietNewsFilter()
    notifier = TelegramNotifier()

    notifier.send("<b>[DEMO]</b> Paper bot started with mocked MT5")

    # ── Tick 1: valid signal should be detected ────────────────────────
    logger.info("--- TICK 1 (London session open, valid Asian range) ---")
    _tick(feed, executor, risk, strategy, news, notifier, tick_num=1)

    # ── Tick 2: simulate open position already present ─────────────────
    logger.info("--- TICK 2 (simulate open position — should skip) ---")
    fake_pos = MagicMock()
    fake_pos.ticket = 11111
    fake_pos.symbol = settings.symbol
    fake_pos.type   = 0  # BUY
    fake_pos.volume = 0.01
    fake_pos.price_open = 1.08078
    fake_pos.sl = 1.0748
    fake_pos.tp = 1.0860
    fake_pos.profit = 12.0
    fake_pos.comment = "LondonBreakout"
    _mt5_mock.positions_get.return_value = [fake_pos]
    _tick(feed, executor, risk, strategy, news, notifier, tick_num=2)

    # ── Tick 3: outside London session ────────────────────────────────
    logger.info("--- TICK 3 (outside London session — no signal) ---")
    _mt5_mock.positions_get.return_value = []

    class _NoLondon(LondonBreakoutStrategy):
        @staticmethod
        def is_london_session(dt: datetime | None = None) -> bool:  # type: ignore[override]
            return False

    strategy3 = _NoLondon(feed)
    _tick(feed, executor, risk, strategy3, news, notifier, tick_num=3)

    logger.info("=" * 60)
    logger.info("  DEMO COMPLETE — 3 ticks simulated")
    logger.info("  Risk report: {}", risk.get_risk_report())
    logger.info("=" * 60)


def _tick(
    feed: DataFeed,
    executor: OrderExecutor,
    risk: RiskManager,
    strategy: LondonBreakoutStrategy,
    news: NewsFilter,
    notifier: TelegramNotifier,
    tick_num: int,
) -> None:
    # a. Risk check
    if not risk.is_safe_to_trade(settings.symbol):
        logger.warning("[tick {}] Risk limit hit — skip", tick_num)
        return

    # b. News
    if news.high_impact_soon():
        logger.info("[tick {}] News filter active — skip", tick_num)
        return

    # c. Weekend
    now = datetime.now(tz=timezone.utc)
    if now.weekday() == 4 and now.hour >= settings.friday_close_hour_utc:
        logger.info("[tick {}] Friday close — skip", tick_num)
        return

    # d. Position check
    open_pos = executor.get_open_positions(settings.symbol)
    if open_pos:
        logger.info("[tick {}] Position already open (count={}) — skip", tick_num, len(open_pos))
        return

    # e. Signal
    sig = strategy.get_signal(settings.symbol)
    if sig["direction"] == "NONE":
        logger.info("[tick {}] No signal", tick_num)
        return

    # f. Lot
    entry  = float(sig["entry"])
    sl_val = float(sig["stop_loss"])
    pip    = 0.01 if settings.symbol.upper().endswith("JPY") else 0.0001
    sl_pips = abs(entry - sl_val) / pip
    lot = risk.calculate_lot_size(settings.symbol, sl_pips)

    logger.info(
        "[tick {}] PAPER SIGNAL | {} {} lots | entry={} sl={} tp={} rsi={} conf={}",
        tick_num,
        sig["direction"], lot,
        sig["entry"], sig["stop_loss"], sig["take_profit"],
        sig.get("rsi", "N/A"), sig.get("confidence", "N/A"),
    )
    notifier.send(
        f"[PAPER tick {tick_num}] {sig['direction']} {lot} lots "
        f"| entry={sig['entry']} SL={sig['stop_loss']} TP={sig['take_profit']}"
    )


if __name__ == "__main__":
    main()
