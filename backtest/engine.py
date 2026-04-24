"""Backtesting engine — replays the London Breakout strategy on H1 MT5 data."""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

from config import settings

# Costs baked in per-trade (in pips)
SPREAD_PIPS = 1.2
SLIPPAGE_PIPS = 0.5
TOTAL_COST_PIPS = SPREAD_PIPS + SLIPPAGE_PIPS

PIP = 0.0001  # EURUSD


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Pure-numpy RSI for the backtest engine."""
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    alpha = 1.0 / period
    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()

    for i in range(period + 1, len(closes)):
        avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gain[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + loss[i] * alpha

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    return rsi


class Backtester:
    """
    Vectorised London Breakout backtester.

    Downloads H1 EURUSD data from MT5 for 2021-01-01 → 2024-12-31.
    Simulates the same breakout logic as the live LondonBreakoutStrategy
    but operates on the full dataset at once.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        start: datetime | None = None,
        end: datetime | None = None,
        initial_balance: float = 10_000.0,
    ) -> None:
        self._symbol = symbol
        self._start = start or datetime(2021, 1, 1, tzinfo=timezone.utc)
        self._end = end or datetime(2024, 12, 31, tzinfo=timezone.utc)
        self._initial_balance = initial_balance

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_from_mt5(self) -> pd.DataFrame:
        """Fetch H1 OHLCV from a live MT5 terminal."""
        if not _MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed")

        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

        bars = mt5.copy_rates_range(
            self._symbol,
            mt5.TIMEFRAME_H1,
            self._start,
            self._end,
        )
        mt5.shutdown()

        if bars is None or len(bars) == 0:
            raise RuntimeError("MT5 returned no bars for the requested range")

        df = pd.DataFrame(bars)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("datetime")
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        logger.info("Loaded {} H1 bars from MT5 ({} → {})", len(df), self._start.date(), self._end.date())
        return df

    def _load_data(self) -> pd.DataFrame:
        """Load H1 data — from MT5 if available, else raise."""
        return self._load_from_mt5()

    # ------------------------------------------------------------------
    # Core logic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _asian_session_range(day_bars: pd.DataFrame) -> tuple[float, float]:
        """Return (high, low) of bars in 00:00–07:00 UTC."""
        asian = day_bars[
            (day_bars.index.hour >= settings.asian_session_start_utc)  # type: ignore[attr-defined]
            & (day_bars.index.hour < settings.asian_session_end_utc)  # type: ignore[attr-defined]
        ]
        if asian.empty:
            return 0.0, 0.0
        return float(asian["high"].max()), float(asian["low"].min())

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the backtest and return a results dict."""
        logger.info("Backtest started | {} {} → {}", self._symbol, self._start.date(), self._end.date())
        df = self._load_data()

        # Pre-compute RSI on the full series
        rsi_values = _rsi(df["close"].values, period=settings.rsi_period)
        df["rsi"] = rsi_values

        equity_curve: list[float] = [self._initial_balance]
        balance = self._initial_balance
        trades: list[dict[str, Any]] = []

        unique_dates = sorted(set(df.index.date))  # type: ignore[attr-defined]

        for date in unique_dates:
            day_bars = df[df.index.date == date]  # type: ignore[attr-defined]

            # Compute Asian range
            a_high, a_low = self._asian_session_range(day_bars)
            if a_high == 0.0:
                continue
            range_pips = (a_high - a_low) / PIP
            if not (settings.min_range_pips <= range_pips <= settings.max_range_pips):
                continue

            buffer = settings.breakout_buffer_pips * PIP
            buy_trigger = a_high + buffer
            sell_trigger = a_low - buffer

            # London session bars
            london = day_bars[
                (day_bars.index.hour >= settings.london_session_start_utc)  # type: ignore[attr-defined]
                & (day_bars.index.hour < settings.london_session_end_utc)  # type: ignore[attr-defined]
            ]
            if london.empty:
                continue

            trade_taken_today = False
            for ts, bar in london.iterrows():
                if trade_taken_today:
                    break

                rsi_val = df.loc[ts, "rsi"]  # type: ignore[index]

                # Check breakout
                direction: str | None = None
                if bar["high"] >= buy_trigger and rsi_val > 50:
                    direction = "BUY"
                elif bar["low"] <= sell_trigger and rsi_val < 50:
                    direction = "SELL"

                if direction is None:
                    continue

                # Entry price (with cost)
                cost = TOTAL_COST_PIPS * PIP
                if direction == "BUY":
                    entry = buy_trigger + cost
                    sl = a_low - buffer
                    risk_dist = entry - sl
                    tp = entry + risk_dist * settings.rr_ratio
                else:
                    entry = sell_trigger - cost
                    sl = a_high + buffer
                    risk_dist = sl - entry
                    tp = entry - risk_dist * settings.rr_ratio

                sl_pips = abs(entry - sl) / PIP
                if sl_pips <= 0:
                    continue

                # Fixed fractional sizing (1% risk)
                risk_amount = balance * settings.risk_per_trade
                pip_value = 10.0  # $10/pip for 1 lot EURUSD standard account
                lot = min(
                    round(risk_amount / (sl_pips * pip_value), 2),
                    settings.max_lot_size,
                )
                lot = max(lot, 0.01)

                # Simulate outcome using remaining bars of the day
                remaining = day_bars[day_bars.index > ts]  # type: ignore[operator]
                outcome = "open"
                exit_price = 0.0

                for _, future_bar in remaining.iterrows():
                    if direction == "BUY":
                        if future_bar["low"] <= sl:
                            outcome = "loss"
                            exit_price = sl
                            break
                        if future_bar["high"] >= tp:
                            outcome = "win"
                            exit_price = tp
                            break
                    else:
                        if future_bar["high"] >= sl:
                            outcome = "loss"
                            exit_price = sl
                            break
                        if future_bar["low"] <= tp:
                            outcome = "win"
                            exit_price = tp
                            break

                if outcome == "open":
                    # Close at end of day
                    last_bar = remaining.iloc[-1] if not remaining.empty else bar
                    exit_price = float(last_bar["close"])
                    pnl_pips = (
                        (exit_price - entry) / PIP if direction == "BUY"
                        else (entry - exit_price) / PIP
                    )
                    pnl = pnl_pips * pip_value * lot
                    outcome = "win" if pnl > 0 else "loss"
                elif outcome == "win":
                    pnl = risk_amount * settings.rr_ratio
                else:
                    pnl = -risk_amount

                balance += pnl
                equity_curve.append(balance)
                trade_taken_today = True

                trades.append(
                    {
                        "date": str(date),
                        "direction": direction,
                        "entry": round(entry, 5),
                        "exit": round(exit_price, 5),
                        "lot": lot,
                        "sl_pips": round(sl_pips, 1),
                        "pnl": round(pnl, 2),
                        "outcome": outcome,
                        "rsi": round(rsi_val, 2),
                        "range_pips": round(range_pips, 1),
                    }
                )

        # ---- Compute metrics ----
        return self._compute_metrics(trades, equity_curve)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        trades: list[dict[str, Any]],
        equity_curve: list[float],
    ) -> dict[str, Any]:
        if not trades:
            logger.warning("Backtest produced 0 trades")
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "monthly_pnl": {},
                "trades": [],
                "equity_curve": equity_curve,
                "initial_balance": self._initial_balance,
                "final_balance": self._initial_balance,
            }

        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df["outcome"] == "win"]
        losses = trades_df[trades_df["outcome"] == "loss"]

        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
        profit_factor = (
            round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
        )

        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak
        max_drawdown = round(float(drawdowns.max()), 4)

        # Sharpe (annualised, using daily PnL)
        trades_df["date_dt"] = pd.to_datetime(trades_df["date"])
        daily_pnl = trades_df.groupby("date_dt")["pnl"].sum()
        if len(daily_pnl) > 1:
            mean_daily = daily_pnl.mean()
            std_daily = daily_pnl.std()
            sharpe = round(float(mean_daily / std_daily * np.sqrt(252)), 2) if std_daily > 0 else 0.0
        else:
            sharpe = 0.0

        final_balance = equity_curve[-1]
        total_return = round((final_balance - self._initial_balance) / self._initial_balance, 4)

        # Monthly PnL
        trades_df["month"] = trades_df["date_dt"].dt.to_period("M").astype(str)
        monthly_pnl: dict[str, float] = {
            k: round(v, 2)
            for k, v in trades_df.groupby("month")["pnl"].sum().to_dict().items()
        }

        results: dict[str, Any] = {
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(trades), 4),
            "wins": len(wins),
            "losses": len(losses),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": profit_factor,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "initial_balance": self._initial_balance,
            "final_balance": round(final_balance, 2),
            "monthly_pnl": monthly_pnl,
            "equity_curve": equity_curve,
            "trades": trades,
        }
        logger.info(
            "Backtest complete | trades={} win_rate={:.1%} return={:.1%} "
            "sharpe={} max_dd={:.1%} pf={}",
            results["total_trades"],
            results["win_rate"],
            results["total_return"],
            results["sharpe"],
            results["max_drawdown"],
            results["profit_factor"],
        )
        return results
