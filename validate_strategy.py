"""
Comprehensive strategy validation:
  Step 1 — Synthetic EURUSD H1 data generation (3 years, seed=42)
  Step 2 — Full backtest with metrics
  Step 3 — HTML report (dark theme)
  Step 4 — Walk-forward analysis (6 × 6-month windows)
  Step 5 — Parameter optimisation (243 grid combinations, Sharpe objective)
  Step 6 — FTMO challenge simulation (100 × 30-day windows)

Run:  python validate_strategy.py
"""
from __future__ import annotations

import itertools
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Console formatting helpers
# ──────────────────────────────────────────────────────────────────────
_W = 78

def _hdr(text: str) -> None:
    print(f"\n{'═' * _W}")
    print(f"  {text}")
    print(f"{'═' * _W}")

def _sec(text: str) -> None:
    print(f"\n{'─' * _W}")
    print(f"  {text}")
    print(f"{'─' * _W}")

def _row(label: str, value: str, width: int = 22) -> None:
    print(f"  {label:<{width}} {value}")

def _blank() -> None:
    print()


# ──────────────────────────────────────────────────────────────────────
# Core RSI  (pure numpy, same as backtest engine)
# ──────────────────────────────────────────────────────────────────────

def _rsi_numpy(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta    = np.diff(closes, prepend=closes[0])
    gain     = np.where(delta > 0,  delta, 0.0)
    loss     = np.where(delta < 0, -delta, 0.0)
    alpha    = 1.0 / period
    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)

    if len(closes) > period:
        avg_gain[period] = gain[1 : period + 1].mean()
        avg_loss[period] = loss[1 : period + 1].mean()
        for i in range(period + 1, len(closes)):
            avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gain[i] * alpha
            avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + loss[i] * alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    rsi[:period] = 50.0   # neutral for warm-up period
    return rsi


# ──────────────────────────────────────────────────────────────────────
# Self-contained London Breakout backtest function
# ──────────────────────────────────────────────────────────────────────

def _backtest(
    df: pd.DataFrame,
    *,
    asian_end:     int   = 7,
    min_range:     float = 10.0,
    max_range:     float = 50.0,
    buffer_pips:   float = 2.0,
    rr:            float = 2.0,
    rsi_thresh:    float = 50.0,
    cost_pips:     float = 1.7,      # spread 1.2 + slippage 0.5
    risk_pct:      float = 0.01,
    initial_bal:   float = 10_000.0,
    rsi_arr:       np.ndarray | None = None,
    start_date:    datetime | None = None,
    end_date:      datetime | None = None,
) -> dict[str, Any]:
    """Run London Breakout on *df* with explicit parameters.

    Supports partial date ranges (for walk-forward / FTMO slicing).
    RSI is computed on the FULL df when provided as rsi_arr so warmup
    history is preserved even when start_date restricts the sim window.
    """
    PIP       = 0.0001
    PIP_VALUE = 10.0      # $10 per pip per standard lot (EURUSD)

    # Pre-compute RSI once on the full series
    if rsi_arr is None:
        rsi_arr = _rsi_numpy(df["close"].values, period=14)

    df_rsi = df.copy()
    df_rsi["rsi"] = rsi_arr

    # Filter simulation window
    sim = df_rsi
    if start_date:
        ts_start = pd.Timestamp(start_date).tz_localize("UTC") if start_date.tzinfo is None else pd.Timestamp(start_date).tz_convert("UTC")
        sim = sim[sim.index >= ts_start]
    if end_date:
        ts_end = pd.Timestamp(end_date).tz_localize("UTC") if end_date.tzinfo is None else pd.Timestamp(end_date).tz_convert("UTC")
        sim = sim[sim.index < ts_end]

    balance     : float              = initial_bal
    equity_curve: list[float]        = [balance]
    trades      : list[dict[str, Any]] = []

    # Group by date once — avoids O(n) scan per day inside the loop
    sim_copy = sim.copy()
    sim_copy["_date"] = sim_copy.index.date  # type: ignore[attr-defined]
    sim_copy["_hour"] = sim_copy.index.hour  # type: ignore[attr-defined]
    grouped = {d: g for d, g in sim_copy.groupby("_date")}

    buf  = buffer_pips * PIP
    cost = cost_pips   * PIP

    for date, day in sorted(grouped.items()):
        if len(day) < 10:
            continue

        # ── Asian session ──────────────────────────────────────────
        asian = day[day["_hour"] < asian_end]
        if len(asian) < 3:
            continue

        a_high = float(asian["high"].max())
        a_low  = float(asian["low"].min())
        rng_pips = (a_high - a_low) / PIP
        if not (min_range <= rng_pips <= max_range):
            continue

        # ── London session ─────────────────────────────────────────
        london_start = max(asian_end, 7)
        london = day[(day["_hour"] >= london_start) & (day["_hour"] < 10)]
        if london.empty:
            continue

        buy_trig  = a_high + buf
        sell_trig = a_low  - buf

        direction: str | None = None
        entry_ts              = None
        entry: float          = 0.0

        for ts, bar in london.iterrows():
            rsi_val = float(bar["rsi"])
            if bar["high"] >= buy_trig and rsi_val > rsi_thresh:
                direction = "BUY"
                entry     = buy_trig + cost
                entry_ts  = ts
                break
            if bar["low"] <= sell_trig and rsi_val < (100.0 - rsi_thresh):
                direction = "SELL"
                entry     = sell_trig - cost
                entry_ts  = ts
                break

        if direction is None:
            continue

        # ── Stop-loss / take-profit ────────────────────────────────
        if direction == "BUY":
            sl   = a_low  - buf
            risk = entry  - sl
            tp   = entry  + risk * rr
        else:
            sl   = a_high + buf
            risk = sl     - entry
            tp   = entry  - risk * rr

        sl_pips = risk / PIP
        if sl_pips <= 0:
            continue

        # ── Position sizing  ───────────────────────────────────────
        risk_amt = balance * risk_pct
        lot      = max(0.01, min(round(risk_amt / (sl_pips * PIP_VALUE), 2), 5.0))

        # ── Simulate outcome ───────────────────────────────────────
        remaining = day[day.index > entry_ts]
        outcome:     str   = "open"
        exit_price:  float = 0.0

        for _, future in remaining.iterrows():
            if direction == "BUY":
                if future["low"]  <= sl:
                    outcome = "loss"; exit_price = sl; break
                if future["high"] >= tp:
                    outcome = "win";  exit_price = tp; break
            else:
                if future["high"] >= sl:
                    outcome = "loss"; exit_price = sl; break
                if future["low"]  <= tp:
                    outcome = "win";  exit_price = tp; break

        if outcome == "open":
            last        = remaining.iloc[-1] if not remaining.empty else day.iloc[-1]
            exit_price  = float(last["close"])
            pnl_pips    = ((exit_price - entry) if direction == "BUY"
                           else (entry - exit_price)) / PIP
            pnl         = pnl_pips * PIP_VALUE * lot
            outcome     = "win" if pnl > 0 else "loss"
        elif outcome == "win":
            pnl = risk_amt * rr
        else:
            pnl = -risk_amt

        balance += pnl
        equity_curve.append(balance)

        trades.append({
            "date":      str(date),
            "direction": direction,
            "entry":     round(entry,      5),
            "exit":      round(exit_price, 5),
            "sl":        round(sl, 5),
            "tp":        round(tp, 5),
            "lot":       lot,
            "sl_pips":   round(sl_pips, 1),
            "pnl":       round(pnl, 2),
            "outcome":   outcome,
            "rsi":       round(float(london.iloc[0]["rsi"]), 2),
        })

    return _calc_metrics(trades, equity_curve, initial_bal, rr)


def _calc_metrics(
    trades: list[dict[str, Any]],
    equity_curve: list[float],
    initial_bal: float,
    rr: float = 2.0,
) -> dict[str, Any]:
    """Compute standard performance metrics."""
    _empty: dict[str, Any] = {
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "total_return": 0.0, "sharpe": 0.0,
        "max_drawdown": 0.0, "profit_factor": 0.0,
        "gross_profit": 0.0, "gross_loss": 0.0,
        "avg_win_pips": 0.0, "avg_loss_pips": 0.0,
        "monthly_pnl": {}, "equity_curve": equity_curve,
        "trades": [], "initial_balance": initial_bal,
        "final_balance": initial_bal,
    }
    if not trades:
        return _empty

    df_t = pd.DataFrame(trades)
    wins   = df_t[df_t["outcome"] == "win"]
    losses = df_t[df_t["outcome"] == "loss"]

    gross_profit = float(wins["pnl"].sum())   if not wins.empty   else 0.0
    gross_loss   = float(losses["pnl"].abs().sum()) if not losses.empty else 0.0
    pf           = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.max())

    df_t["date_dt"] = pd.to_datetime(df_t["date"])
    daily_pnl = df_t.groupby("date_dt")["pnl"].sum()
    sharpe = (
        float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
        if len(daily_pnl) > 1 and daily_pnl.std() > 0
        else 0.0
    )

    final_bal    = equity_curve[-1]
    total_return = (final_bal - initial_bal) / initial_bal

    df_t["month"] = df_t["date_dt"].dt.to_period("M").astype(str)
    monthly_pnl   = {k: round(float(v), 2)
                     for k, v in df_t.groupby("month")["pnl"].sum().items()}

    avg_win_pips  = float(wins["sl_pips"].mean()   * rr) if not wins.empty   else 0.0
    avg_loss_pips = float(losses["sl_pips"].mean())       if not losses.empty else 0.0

    return {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / len(trades),
        "total_return":   total_return,
        "sharpe":         round(sharpe, 2),
        "max_drawdown":   max_dd,
        "profit_factor":  round(pf, 2),
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss, 2),
        "avg_win_pips":   round(avg_win_pips, 1),
        "avg_loss_pips":  round(avg_loss_pips, 1),
        "monthly_pnl":    monthly_pnl,
        "equity_curve":   equity_curve,
        "trades":         trades,
        "initial_balance": initial_bal,
        "final_balance":  round(final_bal, 2),
    }


# ──────────────────────────────────────────────────────────────────────
# Step 1 — Synthetic data generation
# ──────────────────────────────────────────────────────────────────────

def step1_generate_data() -> pd.DataFrame:
    _sec("STEP 1 — SYNTHETIC DATA GENERATION")
    from backtest.synthetic_data import generate_eurusd_h1

    t0 = time.perf_counter()
    df = generate_eurusd_h1(seed=42, initial_price=1.2000)
    elapsed = time.perf_counter() - t0

    daily_ranges: list[float] = []
    for date in sorted(set(df.index.date)):  # type: ignore[attr-defined]
        day = df[df.index.date == date]  # type: ignore[attr-defined]
        daily_ranges.append((day["high"].max() - day["low"].min()) / 0.0001)

    dr = np.array(daily_ranges)
    _blank()
    _row("Bars generated",   f"{len(df):,}")
    _row("Trading days",     f"{len(daily_ranges):,}")
    _row("Period",           f"{df.index.min().date()} → {df.index.max().date()}")
    _row("Open price",       f"{df['open'].iloc[0]:.5f}")
    _row("Last close",       f"{df['close'].iloc[-1]:.5f}")
    _row("All-time high",    f"{df['high'].max():.5f}")
    _row("All-time low",     f"{df['low'].min():.5f}")
    _row("Avg daily range",  f"{dr.mean():.1f} pips")
    _row("Generated in",     f"{elapsed:.2f}s")
    return df


# ──────────────────────────────────────────────────────────────────────
# Step 2 — Full backtest
# ──────────────────────────────────────────────────────────────────────

def step2_backtest(df: pd.DataFrame, rsi_arr: np.ndarray) -> dict[str, Any]:
    _sec("STEP 2 — FULL BACKTEST  (2021 – 2024)")

    t0 = time.perf_counter()
    r  = _backtest(df, rsi_arr=rsi_arr, initial_bal=10_000.0)
    elapsed = time.perf_counter() - t0

    _blank()
    _row("Starting capital",   "$10,000.00")
    _row("Final balance",      f"${r['final_balance']:,.2f}")
    _row("Total Return",       f"{r['total_return']:+.2%}")
    _row("Sharpe Ratio",       f"{r['sharpe']:.2f}")
    _row("Max Drawdown",       f"{r['max_drawdown']:.2%}")
    print()
    _row("Total Trades",       str(r["total_trades"]))
    _row("Win Rate",           f"{r['win_rate']:.2%}")
    _row("Profit Factor",      str(r["profit_factor"]))
    _row("Avg Win",            f"{r['avg_win_pips']:.1f} pips")
    _row("Avg Loss",           f"{r['avg_loss_pips']:.1f} pips")
    _row("Gross Profit",       f"${r['gross_profit']:,.2f}")
    _row("Gross Loss",         f"${r['gross_loss']:,.2f}")

    # Best / worst months
    if r["monthly_pnl"]:
        mp     = r["monthly_pnl"]
        init   = r["initial_balance"]
        best_m = max(mp, key=lambda k: mp[k])
        wrst_m = min(mp, key=lambda k: mp[k])
        _blank()
        _row("Best Month",  f"{best_m}  {mp[best_m]:+.2f} ({mp[best_m]/init:+.2%})")
        _row("Worst Month", f"{wrst_m}  {mp[wrst_m]:+.2f} ({mp[wrst_m]/init:+.2%})")

    # Monthly P&L table
    _blank()
    print("  Monthly P&L Table:")
    print(f"  {'Month':<10} {'P&L ($)':>10} {'P&L (%)':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    init = r["initial_balance"]
    for month in sorted(r["monthly_pnl"]):
        v  = r["monthly_pnl"][month]
        pct = v / init * 100
        bar = "█" * min(abs(int(pct * 3)), 15)
        sign = "+" if v >= 0 else ""
        print(f"  {month:<10} {sign}{v:>9.2f}  {sign}{pct:.2f}%  {bar}")

    _row("Computed in", f"{elapsed:.2f}s")
    return r


# ──────────────────────────────────────────────────────────────────────
# Step 3 — HTML report
# ──────────────────────────────────────────────────────────────────────

def step3_html_report(results: dict[str, Any], path: str = "logs/backtest_report.html") -> None:
    _sec("STEP 3 — HTML REPORT")
    from backtest.report import generate_report
    generate_report(results, output_path=path)
    print(f"\n  Report saved → {os.path.abspath(path)}")


# ──────────────────────────────────────────────────────────────────────
# Step 4 — Walk-forward analysis
# ──────────────────────────────────────────────────────────────────────

def step4_walk_forward(df: pd.DataFrame, rsi_arr: np.ndarray) -> None:
    _sec("STEP 4 — WALK-FORWARD ANALYSIS  (6 × 6-month windows)")
    print()
    print("  Each window: 4 months in-sample + 2 months out-of-sample.")
    print("  No parameter re-fitting — same defaults throughout.")
    print()

    windows = [
        ("W1", datetime(2021, 1, 1, tzinfo=timezone.utc),
                datetime(2021, 5, 1, tzinfo=timezone.utc),
                datetime(2021, 7, 1, tzinfo=timezone.utc)),
        ("W2", datetime(2021, 7, 1, tzinfo=timezone.utc),
                datetime(2021, 11, 1, tzinfo=timezone.utc),
                datetime(2022, 1, 1, tzinfo=timezone.utc)),
        ("W3", datetime(2022, 1, 1, tzinfo=timezone.utc),
                datetime(2022, 5, 1, tzinfo=timezone.utc),
                datetime(2022, 7, 1, tzinfo=timezone.utc)),
        ("W4", datetime(2022, 7, 1, tzinfo=timezone.utc),
                datetime(2022, 11, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 1, tzinfo=timezone.utc)),
        ("W5", datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 5, 1, tzinfo=timezone.utc),
                datetime(2023, 7, 1, tzinfo=timezone.utc)),
        ("W6", datetime(2023, 7, 1, tzinfo=timezone.utc),
                datetime(2023, 11, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, tzinfo=timezone.utc)),
    ]

    hdr_fmt  = f"  {'Win':<5} {'Period (IS → OOS)':<30} {'IS Trades':>10} {'IS WR':>8} {'OOS Trades':>11} {'OOS WR':>8} {'OOS Return':>11} {'OOS Sharpe':>11}"
    print(hdr_fmt)
    print(f"  {'─'*5} {'─'*30} {'─'*10} {'─'*8} {'─'*11} {'─'*8} {'─'*11} {'─'*11}")

    oos_returns = []
    for name, w_start, oos_start, w_end in windows:
        period_str = (
            f"{w_start.strftime('%b %Y')} – {oos_start.strftime('%b %Y')}"
            f" → {oos_start.strftime('%b %Y')} – {(oos_start if oos_start == w_end else w_end).strftime('%b %Y')}"
        )
        is_r  = _backtest(df, rsi_arr=rsi_arr, start_date=w_start,  end_date=oos_start)
        oos_r = _backtest(df, rsi_arr=rsi_arr, start_date=oos_start, end_date=w_end)
        oos_returns.append(oos_r["total_return"])
        print(
            f"  {name:<5} {period_str:<30}"
            f" {is_r['total_trades']:>10} {is_r['win_rate']:>8.1%}"
            f" {oos_r['total_trades']:>11} {oos_r['win_rate']:>8.1%}"
            f" {oos_r['total_return']:>+10.2%}"
            f" {oos_r['sharpe']:>11.2f}"
        )

    _blank()
    avg_oos = np.mean(oos_returns)
    pos_oos = sum(1 for r in oos_returns if r > 0)
    print(f"  Avg OOS Return  : {avg_oos:+.2%}")
    print(f"  Positive OOS    : {pos_oos}/6 windows")
    if avg_oos > 0 and pos_oos >= 4:
        print("  ✓ Strategy shows consistent out-of-sample profitability — no overfitting detected.")
    else:
        print("  ✗ OOS results inconsistent — strategy may be overfitted to training period.")


# ──────────────────────────────────────────────────────────────────────
# Step 5 — Parameter optimisation
# ──────────────────────────────────────────────────────────────────────

def step5_optimize(df: pd.DataFrame, rsi_arr: np.ndarray) -> dict[str, Any]:
    _sec("STEP 5 — PARAMETER OPTIMISATION  (243 combinations)")

    grid = {
        "asian_end":  [6, 7, 8],
        "min_range":  [8.0, 10.0, 15.0],
        "max_range":  [40.0, 50.0, 60.0],
        "rr":         [1.5, 2.0, 2.5],
        "rsi_thresh": [45.0, 50.0, 55.0],
    }
    combos   = list(itertools.product(*grid.values()))
    keys     = list(grid.keys())
    total    = len(combos)
    results  = []

    print(f"\n  Running {total} combinations… ", end="", flush=True)
    t0 = time.perf_counter()

    for combo in combos:
        params = dict(zip(keys, combo))
        r = _backtest(df, rsi_arr=rsi_arr, **params)  # type: ignore[arg-type]
        results.append({**params, **{
            "sharpe":       r["sharpe"],
            "win_rate":     r["win_rate"],
            "total_return": r["total_return"],
            "total_trades": r["total_trades"],
            "max_drawdown": r["max_drawdown"],
            "profit_factor": r["profit_factor"],
        }})

    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    best = results[0]

    # ── Top 10 table ───────────────────────────────────────────────────
    _blank()
    print(f"  Top 10 parameter combinations by Sharpe Ratio:")
    print()
    h = (f"  {'#':<4} {'AsianEnd':>8} {'MinR':>5} {'MaxR':>5} {'RR':>5} "
         f"{'RSI':>5} {'Sharpe':>8} {'WinRate':>8} {'Return':>8} {'Trades':>8} {'MaxDD':>7}")
    print(h)
    print(f"  {'─'*4} {'─'*8} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for i, r in enumerate(results[:10], 1):
        print(
            f"  {i:<4} {r['asian_end']:>8} {r['min_range']:>5.0f} {r['max_range']:>5.0f}"
            f" {r['rr']:>5.1f} {r['rsi_thresh']:>5.0f}"
            f" {r['sharpe']:>8.2f} {r['win_rate']:>8.1%}"
            f" {r['total_return']:>+7.2%} {r['total_trades']:>8}"
            f" {r['max_drawdown']:>7.2%}"
        )

    # ── Parameter sensitivity ─────────────────────────────────────────
    _blank()
    print("  Parameter sensitivity (avg Sharpe at each level):")
    _blank()
    df_res = pd.DataFrame(results)
    for param, values in grid.items():
        row = "  " + f"{param:<12}"
        for v in values:
            avg = df_res[df_res[param] == v]["sharpe"].mean()
            row += f"  {v}→{avg:.2f}"
        print(row)

    # ── 2D heatmap table: RR × RSI threshold ─────────────────────────
    _blank()
    print("  Sharpe heatmap — RR ratio vs RSI threshold  (best asian/range):")
    rr_vals  = grid["rr"]
    rsi_vals = grid["rsi_thresh"]
    header_lbl = "RR \\ RSI"
    print(f"\n  {header_lbl:<10}", end="")
    for rt in rsi_vals:
        print(f"  {rt:>8.0f}", end="")
    print()
    print(f"  {'─'*10}", end="")
    for _ in rsi_vals:
        print(f"  {'─'*8}", end="")
    print()
    for rv in rr_vals:
        print(f"  {rv:<10.1f}", end="")
        for rt in rsi_vals:
            sub = df_res[(df_res["rr"] == rv) & (df_res["rsi_thresh"] == rt)]
            best_sh = sub["sharpe"].max() if not sub.empty else 0.0
            print(f"  {best_sh:>8.2f}", end="")
        print()

    # ── Best parameters ────────────────────────────────────────────────
    _blank()
    print("  ★  Best parameter set:")
    for k in keys:
        _row(f"    {k}", str(best[k]))
    _row("    Sharpe",       f"{best['sharpe']:.2f}")
    _row("    Win Rate",     f"{best['win_rate']:.2%}")
    _row("    Total Return", f"{best['total_return']:+.2%}")

    return best


# ──────────────────────────────────────────────────────────────────────
# Step 6 — FTMO challenge simulation
# ──────────────────────────────────────────────────────────────────────

def step6_ftmo_simulation(
    df: pd.DataFrame,
    rsi_arr: np.ndarray,
    n_windows: int = 100,
    seed: int = 42,
) -> None:
    _sec("STEP 6 — FTMO CHALLENGE SIMULATION  (100 × 30-day windows)")

    # FTMO rules
    INITIAL     = 10_000.0
    TARGET      =  1_000.0   # +10%
    MAX_DAILY   =    500.0   # -5% per day
    MAX_TOTAL   =  1_000.0   # -10% total
    DAYS_LIMIT  =     30     # trading days

    print(f"\n  Account         : ${INITIAL:,.0f}")
    print(f"  Profit target   : +${TARGET:,.0f}  ({TARGET/INITIAL:.0%})")
    print(f"  Max daily loss  : -${MAX_DAILY:,.0f} ({MAX_DAILY/INITIAL:.0%})")
    print(f"  Max total DD    : -${MAX_TOTAL:,.0f} ({MAX_TOTAL/INITIAL:.0%})")
    print(f"  Time limit      : {DAYS_LIMIT} trading days")
    print(f"  Windows tested  : {n_windows}")

    # Run full backtest to get trade list
    full_r = _backtest(df, rsi_arr=rsi_arr, initial_bal=INITIAL)
    trades = full_r["trades"]
    if not trades:
        print("\n  No trades found — simulation skipped.")
        return

    df_t = pd.DataFrame(trades)
    df_t["date_dt"] = pd.to_datetime(df_t["date"])
    df_t = df_t.sort_values("date_dt")
    unique_days = sorted(df_t["date_dt"].unique())

    if len(unique_days) < DAYS_LIMIT:
        print(f"\n  Insufficient trade days ({len(unique_days)}) for simulation.")
        return

    rng = np.random.default_rng(seed)
    max_start = len(unique_days) - DAYS_LIMIT

    passes         = 0
    days_to_pass_  = []
    fail_daily     = 0
    fail_total_dd  = 0
    fail_time      = 0

    for _ in range(n_windows):
        start_idx    = int(rng.integers(0, max_start + 1))
        win_days     = unique_days[start_idx : start_idx + DAYS_LIMIT]
        win_trades   = df_t[df_t["date_dt"].isin(win_days)]

        balance      = INITIAL
        peak_bal     = INITIAL
        passed       = False
        failed       = False
        fail_reason  = ""

        for day_num, day in enumerate(win_days, 1):
            day_trades_rows = win_trades[win_trades["date_dt"] == day]
            day_open_bal    = balance

            for _, tr in day_trades_rows.iterrows():
                balance  += tr["pnl"]
                peak_bal  = max(peak_bal, balance)

                # Check daily loss
                if day_open_bal - balance >= MAX_DAILY:
                    failed      = True
                    fail_reason = "daily"
                    break

                # Check total drawdown
                if peak_bal - balance >= MAX_TOTAL:
                    failed      = True
                    fail_reason = "total_dd"
                    break

                # Check profit target
                if balance >= INITIAL + TARGET:
                    passed = True
                    days_to_pass_.append(day_num)
                    passes += 1
                    break

            if failed or passed:
                break

        if not passed and not failed:
            fail_time += 1
        elif failed:
            if fail_reason == "daily":
                fail_daily += 1
            else:
                fail_total_dd += 1

    pass_rate  = passes / n_windows
    avg_days   = float(np.mean(days_to_pass_)) if days_to_pass_ else 0.0
    fail_total = fail_daily + fail_total_dd + fail_time

    _blank()
    print(f"  {'─'*50}")
    print(f"  PASS RATE          : {pass_rate:.0%}  ({passes}/{n_windows} windows)")
    if days_to_pass_:
        print(f"  Avg days to pass   : {avg_days:.1f}")
        print(f"  Min days to pass   : {min(days_to_pass_)}")
        print(f"  Max days to pass   : {max(days_to_pass_)}")
    print(f"  {'─'*50}")
    print(f"  FAILURE BREAKDOWN  : {fail_total} failures")
    print(f"    Daily loss limit  : {fail_daily:>3}  ({fail_daily/n_windows:.0%})")
    print(f"    Total drawdown    : {fail_total_dd:>3}  ({fail_total_dd/n_windows:.0%})")
    print(f"    Time limit        : {fail_time:>3}  ({fail_time/n_windows:.0%})")
    print(f"  {'─'*50}")

    if pass_rate >= 0.65:
        verdict = "★  HIGH confidence — strategy passes the FTMO challenge consistently."
    elif pass_rate >= 0.45:
        verdict = "◑  MODERATE confidence — passes most windows; review position sizing."
    else:
        verdict = "✗  LOW pass rate — strategy needs improvement before live trading."
    print(f"\n  {verdict}")


# ──────────────────────────────────────────────────────────────────────
# Update config.py with best parameters
# ──────────────────────────────────────────────────────────────────────

def update_config(best: dict[str, Any]) -> None:
    _sec("UPDATING config.py WITH BEST PARAMETERS")
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    if not os.path.exists(config_path):
        print(f"  config.py not found at {config_path}")
        return

    with open(config_path, "r") as f:
        src = f.read()

    replacements = {
        "asian_session_end_utc": f"asian_session_end_utc: int = {int(best['asian_end'])}",
        "min_range_pips":        f"min_range_pips: float = {best['min_range']}",
        "max_range_pips":        f"max_range_pips: float = {best['max_range']}",
        "rr_ratio":              f"rr_ratio: float = {best['rr']}",
    }

    import re
    changed: list[str] = []
    for key, new_line in replacements.items():
        pattern = rf"({re.escape(key)}\s*:\s*\w+\s*=\s*[\d.]+)"
        new_src  = re.sub(pattern, new_line, src)
        if new_src != src:
            src = new_src
            changed.append(key)

    with open(config_path, "w") as f:
        f.write(src)

    if changed:
        for c in changed:
            _row(f"  Updated", f"{c} → {replacements[c].split('=')[-1].strip()}")
    else:
        print("  No changes needed — config already matches best parameters.")
    print(f"  config.py updated at: {os.path.abspath(config_path)}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _hdr(
        f"EUR/USD LONDON BREAKOUT — COMPREHENSIVE STRATEGY VALIDATION\n"
        f"  {'Date':<14}: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"  {'Data':<14}: Synthetic EURUSD H1 (seed=42, 2021-2024)\n"
        f"  {'Strategy':<14}: London Session Breakout + RSI(14) filter + 1:2 R:R"
    )

    # Step 1 — data
    df = step1_generate_data()

    # Pre-compute RSI once on the full series (shared across all steps)
    rsi_arr = _rsi_numpy(df["close"].values, period=14)

    # Step 2 — full backtest
    results = step2_backtest(df, rsi_arr)

    # Step 3 — HTML report
    step3_html_report(results)

    # Step 4 — walk-forward
    step4_walk_forward(df, rsi_arr)

    # Step 5 — optimisation
    best = step5_optimize(df, rsi_arr)

    # Step 6 — FTMO simulation
    step6_ftmo_simulation(df, rsi_arr)

    # Update config.py with optimised parameters
    update_config(best)

    _hdr("VALIDATION COMPLETE")
    print(f"  HTML report : logs/backtest_report.html")
    print(f"  config.py   : updated with best parameters")
    print(f"  Next step   : connect MT5, run  python main.py paper  then  python main.py live")
    print()


if __name__ == "__main__":
    main()
