"""
Synthetic EURUSD.s H1 data generator for strategy validation.

Produces 3 years (2021-2024) of realistic H1 candles with:
- Session-specific volatility (Asian tight, London breakout, NY drift)
- Regime switching  : trending 60% / ranging 35% / news-spike 5%
- Long-term price path matching broad 2021-2024 EURUSD.s movement
- Reproducible via numpy seed
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

PIP = 0.0001

# Approximate quarterly EURUSD.s close prices 2021-2024
_QUARTERLY_TARGETS: dict[tuple[int, int], float] = {
    (2021, 1): 1.1950, (2021, 2): 1.1870, (2021, 3): 1.1600, (2021, 4): 1.1280,
    (2022, 1): 1.1050, (2022, 2): 1.0500, (2022, 3): 0.9900, (2022, 4): 1.0560,
    (2023, 1): 1.0800, (2023, 2): 1.0900, (2023, 3): 1.0580, (2023, 4): 1.1050,
    (2024, 1): 1.0850, (2024, 2): 1.0720, (2024, 3): 1.1150, (2024, 4): 1.0480,
}


def _quarter_target(dt: datetime) -> float:
    q = (dt.month - 1) // 3 + 1
    return _QUARTERLY_TARGETS.get((dt.year, q), 1.0800)


# ──────────────────────────────────────────────────────────────────────
# Bar-level helpers
# ──────────────────────────────────────────────────────────────────────

def _sim_bar(
    rng: np.random.Generator,
    open_price: float,
    vol_pips: float,
    drift_pips: float = 0.0,
    n_steps: int = 20,
) -> dict[str, float]:
    """Simulate a single H1 OHLC bar via sub-step Brownian motion."""
    step_vol   = vol_pips   * PIP / np.sqrt(n_steps)
    step_drift = drift_pips * PIP / n_steps
    prices = [open_price]
    for _ in range(n_steps):
        prices.append(prices[-1] + step_drift + rng.normal(0.0, step_vol))
    return {
        "open":   open_price,
        "high":   float(max(prices)),
        "low":    float(min(prices)),
        "close":  float(prices[-1]),
        "volume": int(rng.integers(300, 3000)),
    }


def _clamp(bar: dict[str, float], lo: float, hi: float) -> dict[str, float]:
    bar["high"]  = min(bar["high"],  hi)
    bar["low"]   = max(bar["low"],   lo)
    bar["close"] = float(np.clip(bar["close"], lo, hi))
    return bar


# ──────────────────────────────────────────────────────────────────────
# Session generators
# ──────────────────────────────────────────────────────────────────────

def _asian_bars(
    rng: np.random.Generator,
    prev_close: float,
    target_range_pips: float,
) -> list[dict[str, float]]:
    """7 H1 bars (00–07 UTC) — tight mean-reverting consolidation."""
    # Slight overnight gap
    center = prev_close + rng.normal(0.0, 2.0) * PIP
    half   = (target_range_pips / 2.0) * PIP
    lo, hi = center - half, center + half
    bars: list[dict[str, float]] = []
    price = prev_close
    for _ in range(7):
        drift = -0.30 * (price - center) / PIP   # pip-level mean reversion
        vol   = float(rng.uniform(1.5, 4.0))
        bar   = _sim_bar(rng, price, vol, drift_pips=drift, n_steps=12)
        bar   = _clamp(bar, lo, hi)
        bars.append(bar)
        price = bar["close"]
    return bars


def _london_trend_bars(
    rng: np.random.Generator,
    prev_close: float,
    a_high: float,
    a_low: float,
    direction: int,
    move_pips: float,
) -> list[dict[str, float]]:
    """3 H1 bars (07–10 UTC) with a strong directional breakout."""
    bars: list[dict[str, float]] = []
    price = prev_close
    # Bar 1: the large breakout candle
    drift1 = direction * move_pips * 0.55
    bar1 = _sim_bar(rng, price, float(rng.uniform(10.0, 18.0)), drift1, n_steps=20)
    if direction == 1:
        bar1["high"] = max(bar1["high"], a_high + 4 * PIP)
    else:
        bar1["low"]  = min(bar1["low"],  a_low  - 4 * PIP)
    bars.append(bar1)
    price = bar1["close"]
    # Bars 2-3: continuation / slight pullback
    for _ in range(2):
        drift = direction * float(rng.uniform(4.0, 10.0))
        bar   = _sim_bar(rng, price, float(rng.uniform(5.0, 12.0)), drift)
        bars.append(bar)
        price = bar["close"]
    return bars


def _london_range_bars(
    rng: np.random.Generator,
    prev_close: float,
    a_high: float,
    a_low: float,
) -> list[dict[str, float]]:
    """3 H1 bars (07–10 UTC) — no clean breakout, price stays near range."""
    bars: list[dict[str, float]] = []
    price  = prev_close
    center = (a_high + a_low) / 2.0
    # Allow a very small extension on 30% of ranging days → false signal
    jitter = float(rng.uniform(0.5, 2.5))  # pips beyond range
    lo = a_low  - jitter * PIP
    hi = a_high + jitter * PIP
    for _ in range(3):
        drift = -0.15 * (price - center) / PIP
        vol   = float(rng.uniform(4.0, 9.0))
        bar   = _sim_bar(rng, price, vol, drift_pips=drift, n_steps=15)
        bar   = _clamp(bar, lo, hi)
        bars.append(bar)
        price = bar["close"]
    return bars


def _london_news_bars(
    rng: np.random.Generator,
    prev_close: float,
    direction: int,
) -> list[dict[str, float]]:
    """3 H1 bars for a news-spike day (60-130 pip move)."""
    bars: list[dict[str, float]] = []
    price = prev_close
    spike = float(rng.uniform(60.0, 130.0))
    # Big spike bar
    bar1 = _sim_bar(rng, price, float(rng.uniform(25.0, 45.0)),
                    direction * spike * 0.7, n_steps=20)
    bars.append(bar1); price = bar1["close"]
    # Partial retracement
    bar2 = _sim_bar(rng, price, float(rng.uniform(10.0, 20.0)),
                    -direction * spike * 0.15)
    bars.append(bar2); price = bar2["close"]
    # Settling
    bar3 = _sim_bar(rng, price, float(rng.uniform(5.0, 12.0)),
                    direction * spike * 0.08)
    bars.append(bar3)
    return bars


def _rest_of_day_bars(
    rng: np.random.Generator,
    prev_close: float,
    close_target: float,
    n_bars: int = 14,
) -> list[dict[str, float]]:
    """H1 bars for hours 10–23 UTC — gradual drift toward the day's target."""
    bars: list[dict[str, float]] = []
    price = prev_close
    for i in range(n_bars):
        # Drift progressively toward close_target
        remaining = n_bars - i
        drift = 0.25 * (close_target - price) / PIP / remaining
        vol   = float(rng.uniform(2.5, 7.0))
        bar   = _sim_bar(rng, price, vol, drift_pips=drift, n_steps=15)
        bars.append(bar)
        price = bar["close"]
    return bars


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def generate_EURUSD.s_h1(
    start_date: datetime | None = None,
    end_date:   datetime | None = None,
    seed:       int   = 42,
    initial_price: float = 1.2000,
) -> pd.DataFrame:
    """Generate realistic synthetic EURUSD.s H1 OHLCV data.

    Parameters
    ----------
    start_date:    Inclusive start (default 2021-01-04 UTC)
    end_date:      Exclusive end   (default 2025-01-01 UTC)
    seed:          NumPy RNG seed for reproducibility
    initial_price: Opening price (default 1.2000)

    Returns
    -------
    DataFrame indexed by tz-aware UTC DatetimeIndex with columns:
    open, high, low, close, volume
    """
    start = start_date or datetime(2021, 1, 4,  tzinfo=timezone.utc)
    end   = end_date   or datetime(2025, 1, 1,  tzinfo=timezone.utc)

    rng = np.random.default_rng(seed)

    # Enumerate trading days Mon–Fri
    trading_days: list[datetime] = []
    cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur < end:
        if cur.weekday() < 5:
            trading_days.append(cur)
        cur += timedelta(days=1)

    records: list[dict[str, Any]] = []
    prev_close = initial_price

    for day in trading_days:
        q_target  = _quarter_target(day)

        # ── Day regime ─────────────────────────────────────────────
        roll = rng.random()
        if   roll < 0.05:  day_type = "news"
        elif roll < 0.62:  day_type = "trending"
        else:              day_type = "ranging"

        # Bias direction toward quarterly target; add some noise
        bias = 0.62 if prev_close < q_target else 0.38
        direction = 1 if rng.random() < bias else -1

        # ── Asian session ──────────────────────────────────────────
        asian_range_pips = float(rng.normal(26.0, 10.0))
        asian_range_pips = float(np.clip(asian_range_pips, 8.0, 55.0))
        a_bars = _asian_bars(rng, prev_close, asian_range_pips)
        a_close = a_bars[-1]["close"]
        a_high  = max(b["high"] for b in a_bars)
        a_low   = min(b["low"]  for b in a_bars)

        # ── London session ─────────────────────────────────────────
        move_pips = float(rng.uniform(22.0, 58.0))
        if   day_type == "trending":
            l_bars = _london_trend_bars(rng, a_close, a_high, a_low,
                                        direction, move_pips)
        elif day_type == "news":
            l_bars = _london_news_bars(rng, a_close, direction)
        else:
            l_bars = _london_range_bars(rng, a_close, a_high, a_low)
        l_close = l_bars[-1]["close"]

        # ── Day close target ───────────────────────────────────────
        trend_add = direction * float(rng.uniform(5.0, 30.0)) * PIP
        noise_add = float(rng.normal(0.0, 15.0)) * PIP
        day_target = q_target + noise_add + (
            trend_add if day_type == "trending" else noise_add * 0.5
        )

        # ── Rest of day ────────────────────────────────────────────
        r_bars = _rest_of_day_bars(rng, l_close, day_target, n_bars=14)

        # ── Assemble bars with timestamps ──────────────────────────
        hour = 0
        for bar in a_bars + l_bars + r_bars:
            ts = day.replace(hour=hour, tzinfo=timezone.utc)
            records.append({
                "datetime": ts,
                "open":   round(bar["open"],  5),
                "high":   round(max(bar["open"], bar["high"]),  5),
                "low":    round(min(bar["open"], bar["low"]),   5),
                "close":  round(bar["close"], 5),
                "volume": bar["volume"],
            })
            hour += 1

        prev_close = r_bars[-1]["close"]

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    return df
