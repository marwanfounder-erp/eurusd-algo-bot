"""Verify Neon PostgreSQL connection and print table summaries.

Usage:
    python scripts/check_db.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from bot.database import get_db  # noqa: E402


def _divider(title: str = "") -> None:
    line = "─" * 55
    if title:
        pad = (55 - len(title) - 2) // 2
        print(f"{'─'*pad} {title} {'─'*(55-pad-len(title)-2)}")
    else:
        print(line)


def main() -> None:
    db = get_db()

    _divider("CONNECTION")
    if db._pool:
        print("  ✓ Connected to Neon PostgreSQL")
        try:
            row = db._run("SELECT version()", fetch="one")
            ver = list(row.values())[0] if row else "unknown"
            # Shorten the version string
            ver_short = ver.split(",")[0] if ver else "unknown"
            print(f"  Server: {ver_short}")
        except Exception as exc:
            print(f"  Version query failed: {exc}")
    else:
        print("  ✗ NOT connected — check DATABASE_URL")
        sys.exit(1)

    _divider("TABLE ROW COUNTS")
    tables = ["trades", "account_snapshots", "bot_logs", "bot_stats"]
    for table in tables:
        try:
            row = db._run(f"SELECT COUNT(*) AS n FROM {table}", fetch="one")
            count = int(row["n"]) if row else 0
            print(f"  {table:<22} {count:>6} rows")
        except Exception as exc:
            print(f"  {table:<22}  ERROR: {exc}")

    _divider("LAST 3 CLOSED TRADES")
    try:
        trades = db.get_all_trades(limit=3)
        if not trades:
            print("  (no closed trades)")
        for t in trades:
            result = t.get("result", "?")
            marker  = "WIN " if result == "win" else "LOSS"
            date_   = str(t.get("opened_at", ""))[:10]
            dir_    = t.get("direction", "?")
            entry   = float(t.get("entry_price") or 0)
            pnl     = float(t.get("pnl") or 0)
            print(f"  [{marker}] {date_} | {dir_} @ {entry:.5f} | P&L: ${pnl:+.2f}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    _divider("CURRENT BOT STATS")
    try:
        stats = db.get_stats()
        if not stats:
            print("  (no stats row — will be created on first trade)")
        else:
            balance = float(stats.get("current_balance", 10_000))
            wins    = int(stats.get("wins", 0))
            losses  = int(stats.get("losses", 0))
            total   = int(stats.get("total_trades", 0))
            wr      = wins / total * 100 if total else 0.0
            ret_pct = (balance - 10_000) / 10_000 * 100
            print(f"  Balance       : ${balance:,.2f}")
            print(f"  Return        : {ret_pct:+.2f}%")
            print(f"  Total trades  : {total}")
            print(f"  Wins / Losses : {wins} / {losses}")
            print(f"  Win rate      : {wr:.1f}%")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    _divider("LAST 5 LOG ENTRIES")
    try:
        logs = db.get_recent_logs(limit=5)
        if not logs:
            print("  (no logs)")
        for log in logs:
            level = log.get("level", "INFO")
            msg   = log.get("message", "")[:60]
            ts    = str(log.get("created_at", ""))[:19]
            marker = {"INFO": "·", "WARNING": "!", "ERROR": "✗"}.get(level, "·")
            print(f"  {marker} [{level:<7}] {ts}  {msg}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    _divider()
    print("  Database check complete.\n")


if __name__ == "__main__":
    main()
