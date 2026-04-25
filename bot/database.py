"""Neon PostgreSQL persistence layer.

Uses psycopg2 SimpleConnectionPool.  Every public method is wrapped in
try/except so a DB outage never crashes the trading bot.

Actual Neon table schemas (verified):
  trades          — id, symbol, direction, entry_price, exit_price, stop_loss,
                    take_profit, lot_size, pips, pnl, status, result, rsi,
                    confidence, asian_range_pips, asian_high, asian_low,
                    opened_at, closed_at, mode
  account_snapshots — id, balance, equity, daily_pnl, total_return_pct,
                      open_positions, timestamp
  bot_logs        — id, level, message, timestamp
  bot_stats       — id, starting_balance, current_balance, total_trades,
                    wins, losses, today_pnl, last_signal, last_updated

Environment variable required:
    DATABASE_URL = postgresql://user:pass@host/db
    (sslmode=require is added automatically if absent)
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from loguru import logger

_INITIAL_BALANCE = 10_000.0
_IN_MEMORY_LOGS: list[dict[str, Any]] = []   # fallback when DB is down


class Database:
    """Thread-safe PostgreSQL database interface with connection pooling."""

    def __init__(self) -> None:
        self._pool: SimpleConnectionPool | None = None
        self._lock = threading.Lock()
        self._connect()

    # ── Connection management ─────────────────────────────────────────

    def _connect(self) -> None:
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            logger.error("DATABASE_URL not set — DB features disabled")
            return
        if "sslmode" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        try:
            self._pool = SimpleConnectionPool(1, 5, url)
            logger.info("Database pool created (min=1, max=5)")
        except Exception as exc:
            logger.error("Database connection failed: {}", exc)
            self._pool = None

    def get_conn(self):  # type: ignore[return]
        with self._lock:
            if self._pool is None:
                self._connect()
            if self._pool is None:
                raise RuntimeError("No database connection available")
            return self._pool.getconn()

    def release_conn(self, conn: Any, close: bool = False) -> None:
        with self._lock:
            if self._pool:
                try:
                    self._pool.putconn(conn, close=close)
                except Exception:
                    pass

    def _run(
        self,
        sql: str,
        params: tuple | None = None,
        fetch: str | None = None,
    ) -> Any:
        """Execute SQL, commit, return result.  Reconnects once on failure."""
        conn = None
        for attempt in range(2):
            try:
                conn = self.get_conn()
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    if fetch == "one":
                        result = cur.fetchone()
                        result = dict(result) if result else None
                    elif fetch == "all":
                        result = [dict(r) for r in cur.fetchall()]
                    else:
                        result = None
                conn.commit()
                return result
            except psycopg2.OperationalError as exc:
                logger.warning("DB operational error (attempt {}): {}", attempt + 1, exc)
                if conn:
                    self.release_conn(conn, close=True)
                    conn = None
                if attempt == 0:
                    self._connect()
                    continue
                raise
            except Exception:
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn:
                    self.release_conn(conn)

    # ── Trade methods ──────────────────────────────────────────────────

    def save_trade(self, trade: dict[str, Any]) -> str:
        """Insert a new open trade; return the generated trade_id (UUID)."""
        trade_id = str(uuid.uuid4())
        try:
            self._run(
                """
                INSERT INTO trades
                  (id, symbol, direction, entry_price, stop_loss, take_profit,
                   lot_size, rsi, status, mode, opened_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'open',%s,%s)
                """,
                (
                    trade_id,
                    trade.get("symbol", "EURUSD"),
                    trade.get("direction", ""),
                    trade.get("entry", 0.0),
                    trade.get("sl", 0.0),
                    trade.get("tp", 0.0),
                    trade.get("lot", 0.0),
                    trade.get("rsi"),
                    trade.get("mode", "paper"),
                    trade.get("opened_at", datetime.now(tz=timezone.utc).isoformat()),
                ),
            )
            logger.info("DB: trade {} saved", trade_id[:8])
        except Exception as exc:
            logger.error("DB save_trade failed: {}", exc)
        return trade_id

    def update_trade(self, trade_id: str, updates: dict[str, Any]) -> None:
        """Update fields on an open trade (e.g. SL/TP modification)."""
        if not updates:
            return
        cols = list(updates.keys())
        vals = list(updates.values())
        set_clause = ", ".join(f"{c} = %s" for c in cols)
        try:
            self._run(
                f"UPDATE trades SET {set_clause} WHERE id = %s",
                tuple(vals) + (trade_id,),
            )
            logger.info("DB: trade {} updated", trade_id[:8])
        except Exception as exc:
            logger.error("DB update_trade failed: {}", exc)

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pips: float,
        result: str,
    ) -> None:
        """Mark a trade as closed with exit data."""
        try:
            self._run(
                """
                UPDATE trades
                   SET exit_price = %s,
                       pnl        = %s,
                       pips       = %s,
                       result     = %s,
                       status     = 'closed',
                       closed_at  = %s
                 WHERE id = %s
                """,
                (
                    exit_price,
                    pnl,
                    pips,
                    result,
                    datetime.now(tz=timezone.utc).isoformat(),
                    trade_id,
                ),
            )
            logger.info("DB: trade {} closed ({})", trade_id[:8], result)
        except Exception as exc:
            logger.error("DB close_trade failed: {}", exc)

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Return all trades with status='open'."""
        try:
            return self._run(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY opened_at DESC",
                fetch="all",
            ) or []
        except Exception as exc:
            logger.error("DB get_open_trades failed: {}", exc)
            return []

    def get_today_trades(self) -> list[dict[str, Any]]:
        """Return all closed trades from today (UTC)."""
        try:
            return self._run(
                """SELECT * FROM trades
                   WHERE status = 'closed'
                     AND opened_at::date = CURRENT_DATE
                   ORDER BY opened_at DESC""",
                fetch="all",
            ) or []
        except Exception as exc:
            logger.error("DB get_today_trades failed: {}", exc)
            return []

    def get_all_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return most-recent closed trades (newest first)."""
        try:
            return self._run(
                """SELECT * FROM trades WHERE status = 'closed'
                   ORDER BY closed_at DESC LIMIT %s""",
                (limit,),
                fetch="all",
            ) or []
        except Exception as exc:
            logger.error("DB get_all_trades failed: {}", exc)
            return []

    def get_trade_stats(self) -> dict[str, Any]:
        """Compute aggregated performance metrics from closed trades."""
        try:
            row = self._run(
                """
                SELECT
                  COUNT(*)                                          AS total_trades,
                  COUNT(*) FILTER (WHERE result = 'win')           AS wins,
                  COUNT(*) FILTER (WHERE result = 'loss')          AS losses,
                  COALESCE(SUM(pnl) FILTER (WHERE pnl > 0), 0)    AS gross_profit,
                  COALESCE(SUM(ABS(pnl)) FILTER (WHERE pnl < 0), 0) AS gross_loss,
                  COALESCE(SUM(pnl), 0)                            AS total_pnl,
                  COALESCE(AVG(pips) FILTER (WHERE result = 'win'),  0) AS avg_win_pips,
                  COALESCE(AVG(pips) FILTER (WHERE result = 'loss'), 0) AS avg_loss_pips,
                  COALESCE(MAX(pnl), 0)                            AS best_trade_pnl,
                  COALESCE(MIN(pnl), 0)                            AS worst_trade_pnl
                FROM trades
                WHERE status = 'closed'
                """,
                fetch="one",
            )
            if not row:
                return self._empty_stats()

            total    = int(row["total_trades"] or 0)
            wins     = int(row["wins"] or 0)
            gross_p  = float(row["gross_profit"] or 0)
            gross_l  = float(row["gross_loss"] or 0)
            win_rate = wins / total * 100 if total else 0.0
            profit_factor = (
                gross_p / gross_l if gross_l > 0
                else (999.0 if gross_p > 0 else 0.0)
            )
            total_pnl = float(row["total_pnl"] or 0)

            stats_row = self.get_stats()
            balance = float(stats_row.get("current_balance", _INITIAL_BALANCE))
            total_return_pct = total_pnl / _INITIAL_BALANCE * 100

            return {
                "total_trades":    total,
                "wins":            wins,
                "losses":          int(row["losses"] or 0),
                "win_rate":        round(win_rate, 1),
                "profit_factor":   round(profit_factor, 2),
                "gross_profit":    round(gross_p, 2),
                "gross_loss":      round(gross_l, 2),
                "total_pnl":       round(total_pnl, 2),
                "avg_win_pips":    round(float(row["avg_win_pips"] or 0), 1),
                "avg_loss_pips":   round(float(row["avg_loss_pips"] or 0), 1),
                "best_trade_pnl":  round(float(row["best_trade_pnl"] or 0), 2),
                "worst_trade_pnl": round(float(row["worst_trade_pnl"] or 0), 2),
                "total_return_pct": round(total_return_pct, 2),
                "balance":         round(balance, 2),
            }
        except Exception as exc:
            logger.error("DB get_trade_stats failed: {}", exc)
            return self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "total_pnl": 0.0, "avg_win_pips": 0.0, "avg_loss_pips": 0.0,
            "best_trade_pnl": 0.0, "worst_trade_pnl": 0.0,
            "total_return_pct": 0.0, "balance": _INITIAL_BALANCE,
        }

    # ── Account snapshot methods ───────────────────────────────────────

    def save_snapshot(
        self,
        balance: float,
        equity: float,
        daily_pnl: float,
        return_pct: float,
        open_positions: int = 0,
    ) -> None:
        try:
            self._run(
                """
                INSERT INTO account_snapshots
                  (balance, equity, daily_pnl, total_return_pct, open_positions, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (balance, equity, daily_pnl, return_pct, open_positions,
                 datetime.now(tz=timezone.utc).isoformat()),
            )
        except Exception as exc:
            logger.error("DB save_snapshot failed: {}", exc)

    def get_snapshots(self, days: int = 30) -> list[dict[str, Any]]:
        try:
            return self._run(
                """SELECT balance, equity, daily_pnl, total_return_pct, timestamp
                   FROM account_snapshots
                   WHERE timestamp >= NOW() - (%s || ' days')::INTERVAL
                   ORDER BY timestamp ASC""",
                (str(days),),
                fetch="all",
            ) or []
        except Exception as exc:
            logger.error("DB get_snapshots failed: {}", exc)
            return []

    def update_stats(self, updates: dict[str, Any]) -> None:
        """Upsert into bot_stats (single-row table, id=1).

        Incoming key aliases supported:
          balance       → current_balance
          updated_at    → last_updated
        """
        if not updates:
            return
        # Translate any convenience aliases to actual column names
        mapped: dict[str, Any] = {}
        for k, v in updates.items():
            if k == "balance":
                mapped["current_balance"] = v
            elif k == "updated_at":
                mapped["last_updated"] = v
            else:
                mapped[k] = v
        mapped["last_updated"] = datetime.now(tz=timezone.utc).isoformat()

        cols   = list(mapped.keys())
        vals   = list(mapped.values())
        insert_cols = ", ".join(["id"] + cols)
        insert_phs  = ", ".join(["%s"] * (len(cols) + 1))
        set_clause  = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
        try:
            self._run(
                f"""
                INSERT INTO bot_stats ({insert_cols})
                VALUES ({insert_phs})
                ON CONFLICT (id) DO UPDATE SET {set_clause}
                """,
                tuple([1] + vals),
            )
        except Exception as exc:
            logger.error("DB update_stats failed: {}", exc)

    def get_stats(self) -> dict[str, Any]:
        try:
            row = self._run("SELECT * FROM bot_stats WHERE id = 1", fetch="one")
            return dict(row) if row else {}
        except Exception as exc:
            logger.error("DB get_stats failed: {}", exc)
            return {}

    # ── Log methods ────────────────────────────────────────────────────

    def log(self, level: str, message: str) -> None:
        """Insert a log entry; falls back to in-memory list on DB failure."""
        entry: dict[str, Any] = {
            "level":   level.upper(),
            "message": message,
            "ts":      datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            self._run(
                "INSERT INTO bot_logs (level, message, timestamp) VALUES (%s, %s, %s)",
                (entry["level"], entry["message"], entry["ts"]),
            )
        except Exception as exc:
            logger.warning("DB log failed (using in-memory): {}", exc)
            _IN_MEMORY_LOGS.append(entry)
            if len(_IN_MEMORY_LOGS) > 500:
                del _IN_MEMORY_LOGS[:100]

    def get_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            rows = self._run(
                """SELECT level, message, timestamp AS created_at
                   FROM bot_logs ORDER BY timestamp DESC LIMIT %s""",
                (limit,),
                fetch="all",
            )
            return rows or []
        except Exception as exc:
            logger.error("DB get_recent_logs failed: {}", exc)
            return list(reversed(_IN_MEMORY_LOGS[-limit:]))


# ── Module-level singleton ─────────────────────────────────────────────────
_db_instance: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    """Return the module-level Database singleton."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = Database()
    return _db_instance
