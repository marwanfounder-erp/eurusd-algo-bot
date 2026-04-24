"""Seed Neon PostgreSQL with 10 realistic paper trades + 10 account snapshots.

Usage:
    python scripts/seed_test_data.py

Inserts:
  - 10 closed trades (7 wins, 3 losses) spread over last 10 trading days
  - 10 account snapshots for the equity curve
  - Updates bot_stats with aggregate totals
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from bot.database import get_db  # noqa: E402 — after sys.path

# ── Realistic trade data ───────────────────────────────────────────────────
# 7 wins, 3 losses across 10 trading days (Mon–Fri, skipping weekend)
# All entries are London-session breakouts around 07:00–10:00 UTC

def _trading_days_back(n: int) -> list[date]:
    """Return last n trading days (Mon–Fri), newest first."""
    days = []
    d = date.today()
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon=0 … Fri=4
            days.append(d)
    return days


TRADES = [
    # (direction, entry, sl, tp, rsi, outcome, pnl_usd, pips)
    # 1 trade per day; entries chosen as realistic EURUSD breakout levels
    ("BUY",  1.08432, 1.08182, 1.08932,  58.4, "win",  +100.00, +25.0),  # day 1
    ("SELL", 1.08765, 1.09015, 1.08265,  41.2, "win",  +100.00, +25.0),  # day 2
    ("BUY",  1.08920, 1.08670, 1.09420,  62.1, "win",  +100.00, +25.0),  # day 3
    ("SELL", 1.09340, 1.09090, 1.08840,  39.8, "loss", -100.00, -25.0),  # day 4  loss
    ("BUY",  1.09015, 1.08765, 1.09515,  55.7, "win",  +100.00, +25.0),  # day 5
    ("BUY",  1.08680, 1.08430, 1.09180,  60.3, "win",  +100.00, +25.0),  # day 6
    ("SELL", 1.09125, 1.08875, 1.08625,  42.9, "loss", -100.00, -25.0),  # day 7  loss
    ("BUY",  1.08940, 1.08690, 1.09440,  57.6, "win",  +100.00, +25.0),  # day 8
    ("SELL", 1.09260, 1.09010, 1.08760,  38.5, "loss", -100.00, -25.0),  # day 9  loss
    ("BUY",  1.08810, 1.08560, 1.09310,  63.2, "win",  +100.00, +25.0),  # day 10
]


def main() -> None:
    db = get_db()
    if not db._pool:
        print("ERROR: could not connect to database. Check DATABASE_URL.")
        sys.exit(1)

    trading_days = _trading_days_back(10)

    balance = 10_000.0
    gross_profit = 0.0
    gross_loss   = 0.0
    wins = 0
    losses = 0
    snapshots = []

    print(f"\nSeeding {len(TRADES)} trades into Neon...\n")

    for i, (direction, entry, sl, tp, rsi, outcome, pnl, pips) in enumerate(TRADES):
        trade_date = trading_days[i]
        opened_at  = datetime(
            trade_date.year, trade_date.month, trade_date.day,
            7, 30, 0, tzinfo=timezone.utc
        )
        closed_at  = datetime(
            trade_date.year, trade_date.month, trade_date.day,
            9, 45, 0, tzinfo=timezone.utc
        )
        sl_pips = abs(entry - sl) / 0.0001
        lot     = 0.10

        # Compute realistic exit price
        if direction == "BUY":
            exit_price = tp if outcome == "win" else sl
        else:
            exit_price = tp if outcome == "win" else sl

        trade_id = str(uuid.uuid4())
        try:
            db._run(
                """
                INSERT INTO trades
                  (id, direction, entry, sl, tp, lot, sl_pips, rsi,
                   exit_price, pnl, pips, outcome, opened_at, closed_at, date)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    trade_id, direction, entry, sl, tp, lot, sl_pips, rsi,
                    exit_price, pnl, pips, outcome,
                    opened_at.isoformat(), closed_at.isoformat(),
                    trade_date.isoformat(),
                ),
            )
            balance += pnl
            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

            daily_pnl  = pnl
            return_pct = (balance - 10_000.0) / 10_000.0 * 100
            snapshots.append((balance, balance, daily_pnl, return_pct, closed_at.isoformat()))

            marker = "✓ WIN " if outcome == "win" else "✗ LOSS"
            print(f"  [{marker}] {trade_date} | {direction} @ {entry:.5f} → {exit_price:.5f} | P&L: ${pnl:+.2f} | Balance: ${balance:,.2f}")
        except Exception as exc:
            print(f"  ERROR inserting trade {i+1}: {exc}")

    print(f"\nInserting {len(snapshots)} account snapshots...")
    for snap in snapshots:
        try:
            db._run(
                "INSERT INTO account_snapshots (balance, equity, daily_pnl, return_pct, created_at) VALUES (%s,%s,%s,%s,%s)",
                snap,
            )
        except Exception as exc:
            print(f"  ERROR inserting snapshot: {exc}")

    print("\nUpdating bot_stats...")
    total_trades = wins + losses
    try:
        db.update_stats({
            "balance":      round(balance, 2),
            "total_trades": total_trades,
            "wins":         wins,
            "losses":       losses,
            "gross_profit": round(gross_profit, 2),
            "gross_loss":   round(gross_loss, 2),
        })
    except Exception as exc:
        print(f"  ERROR updating bot_stats: {exc}")

    print("\nInserting sample log entries...")
    sample_logs = [
        ("INFO",    "Bot started — mode=PAPER feed=PaperFeed symbol=EURUSD"),
        ("INFO",    "Signal: BUY | entry=1.08432 rsi=58.4"),
        ("INFO",    "PAPER TRADE opened: BUY 0.10lots @ 1.08432"),
        ("INFO",    "Position closed WIN @ 1.08932 pnl=$100.00 balance=$10100.00"),
        ("WARNING", "News filter active | next_event=USD CPI in 28.5min"),
        ("INFO",    "Daily reset complete"),
        ("ERROR",   "yfinance 1m tick failed: connection timeout — falling back to 1h"),
        ("INFO",    "Signal: SELL | entry=1.09260 rsi=38.5"),
        ("INFO",    "PAPER TRADE opened: SELL 0.10lots @ 1.09260"),
        ("INFO",    f"Position closed LOSS @ 1.09010 pnl=-$100.00 balance=${balance:,.2f}"),
    ]
    for level, msg in sample_logs:
        try:
            db.log(level, msg)
        except Exception as exc:
            print(f"  ERROR inserting log: {exc}")

    final_return = (balance - 10_000.0) / 10_000.0 * 100
    win_rate = wins / total_trades * 100 if total_trades else 0

    print("\n" + "="*55)
    print("  SEED COMPLETE")
    print("="*55)
    print(f"  Trades inserted : {total_trades} ({wins}W / {losses}L)")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total P&L       : ${balance - 10_000.0:+,.2f}")
    print(f"  Final balance   : ${balance:,.2f}")
    print(f"  Return          : {final_return:+.2f}%")
    print(f"  Snapshots       : {len(snapshots)}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
