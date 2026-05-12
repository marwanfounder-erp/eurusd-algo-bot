"""Seed Neon PostgreSQL with 10 realistic paper trades + 10 account snapshots.

Usage:
    python scripts/seed_test_data.py

Inserts:
  - 10 closed trades (7 wins, 3 losses) spread over last 10 trading days
  - 10 account snapshots for the equity curve
  - Upserts bot_stats with aggregate totals
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(".env")

from bot.database import get_db  # noqa: E402


def _trading_days_back(n: int) -> list[date]:
    """Return last n trading days (Mon–Fri), oldest first."""
    days: list[date] = []
    d = date.today()
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    days.reverse()
    return days


# 10 trades — 7 wins, 3 losses
# (direction, entry, sl, tp, rsi, result)
TRADES = [
    ("BUY",  1.08432, 1.08182, 1.08932,  58.4, "win"),
    ("SELL", 1.08765, 1.09015, 1.08265,  41.2, "win"),
    ("BUY",  1.08920, 1.08670, 1.09420,  62.1, "win"),
    ("SELL", 1.09340, 1.09590, 1.08840,  39.8, "loss"),
    ("BUY",  1.09015, 1.08765, 1.09515,  55.7, "win"),
    ("BUY",  1.08680, 1.08430, 1.09180,  60.3, "win"),
    ("SELL", 1.09125, 1.09375, 1.08625,  42.9, "loss"),
    ("BUY",  1.08940, 1.08690, 1.09440,  57.6, "win"),
    ("SELL", 1.09260, 1.09510, 1.08760,  38.5, "loss"),
    ("BUY",  1.08810, 1.08560, 1.09310,  63.2, "win"),
]

RISK_PER_TRADE = 100.0   # $100 per trade (1% of $10k)
RR_RATIO       = 1.5     # 1:1.5 R:R → $150 win / $100 loss


def main() -> None:
    db = get_db()
    if not db._pool:
        print("ERROR: could not connect to database. Check DATABASE_URL.")
        sys.exit(1)

    trading_days = _trading_days_back(10)

    balance      = 10_000.0
    wins         = 0
    losses       = 0
    total_trades = 0
    snapshots    = []

    print(f"\nSeeding {len(TRADES)} trades into Neon...\n")

    for i, (direction, entry, sl, tp, rsi, result) in enumerate(TRADES):
        trade_date = trading_days[i]
        opened_at  = datetime(trade_date.year, trade_date.month, trade_date.day,
                              7, 30, 0, tzinfo=timezone.utc)
        closed_at  = datetime(trade_date.year, trade_date.month, trade_date.day,
                              9, 45, 0, tzinfo=timezone.utc)

        pnl  = RISK_PER_TRADE * RR_RATIO if result == "win" else -RISK_PER_TRADE
        pips = 25.0 if result == "win" else -25.0

        # Exit price: use TP on win, SL on loss
        if direction == "BUY":
            exit_price = tp if result == "win" else sl
        else:
            exit_price = tp if result == "win" else sl

        trade_id = str(uuid.uuid4())
        try:
            db._run(
                """
                INSERT INTO trades
                  (id, symbol, direction, entry_price, exit_price, stop_loss,
                   take_profit, lot_size, pips, pnl, status, result, rsi,
                   mode, opened_at, closed_at)
                VALUES (%s,'EURUSD.s',%s,%s,%s,%s,%s,0.10,%s,%s,'closed',%s,%s,'paper',%s,%s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    trade_id, direction, entry, exit_price, sl, tp,
                    pips, pnl, result, rsi,
                    opened_at.isoformat(), closed_at.isoformat(),
                ),
            )
            balance      += pnl
            total_trades += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1

            return_pct = (balance - 10_000.0) / 10_000.0 * 100
            snapshots.append((
                round(balance, 2), round(balance, 2),
                round(pnl, 2), round(return_pct, 2), 0,
                closed_at.isoformat(),
            ))

            marker = "✓ WIN " if result == "win" else "✗ LOSS"
            print(f"  [{marker}] {trade_date} | {direction} @ {entry:.5f} "
                  f"→ {exit_price:.5f} | P&L: ${pnl:+.2f} | Balance: ${balance:,.2f}")
        except Exception as exc:
            print(f"  ERROR trade {i+1}: {exc}")

    print(f"\nInserting {len(snapshots)} account snapshots...")
    for snap in snapshots:
        try:
            db._run(
                """INSERT INTO account_snapshots
                     (balance, equity, daily_pnl, total_return_pct, open_positions, timestamp)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                snap,
            )
        except Exception as exc:
            print(f"  ERROR snapshot: {exc}")

    print("\nUpdating bot_stats...")
    try:
        db.update_stats({
            "current_balance": round(balance, 2),
            "total_trades":    total_trades,
            "wins":            wins,
            "losses":          losses,
        })
    except Exception as exc:
        print(f"  ERROR bot_stats: {exc}")

    print("\nInserting sample log entries...")
    sample_logs = [
        ("INFO",    "Bot started — mode=PAPER feed=PaperFeed symbol=EURUSD.s"),
        ("INFO",    "Signal: BUY | entry=1.08432 rsi=58.4"),
        ("INFO",    "PAPER TRADE opened: BUY 0.10lots @ 1.08432 sl=1.08182 tp=1.08932"),
        ("INFO",    "Position closed WIN @ 1.08932 pnl=$150.00 balance=$10150.00"),
        ("WARNING", "News filter active | next_event=USD CPI in 28.5min — blocking trade"),
        ("INFO",    "Daily reset complete"),
        ("ERROR",   "yfinance 1m tick failed: timeout — falling back to 1h"),
        ("INFO",    "Signal: SELL | entry=1.09260 rsi=38.5"),
        ("INFO",    "PAPER TRADE opened: SELL 0.10lots @ 1.09260 sl=1.09510 tp=1.08760"),
        ("INFO",    f"Position closed LOSS @ 1.09510 pnl=-$100.00 balance=${balance:,.2f}"),
    ]
    for level, msg in sample_logs:
        try:
            db.log(level, msg)
        except Exception as exc:
            print(f"  ERROR log: {exc}")

    net_pnl    = balance - 10_000.0
    return_pct = net_pnl / 10_000.0 * 100
    win_rate   = wins / total_trades * 100 if total_trades else 0

    print("\n" + "=" * 55)
    print("  SEED COMPLETE")
    print("=" * 55)
    print(f"  Trades inserted : {total_trades} ({wins}W / {losses}L)")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total P&L       : ${net_pnl:+,.2f}")
    print(f"  Final balance   : ${balance:,.2f}")
    print(f"  Return          : {return_pct:+.2f}%")
    print(f"  Snapshots       : {len(snapshots)}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
