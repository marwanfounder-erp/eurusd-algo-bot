"""Entry point — TradingBot with live, paper, and backtest modes.

Platform detection
------------------
* Windows with MetaTrader5 installed → uses DataFeed (MT5)
* Linux / Railway / any platform without MT5 → uses PaperFeed (yfinance)
  Paper mode is forced automatically; --live is not available without MT5.
"""
from __future__ import annotations

import platform
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import click
from dotenv import load_dotenv
from loguru import logger

import os

# Load .env into os.environ BEFORE any bot modules read DATABASE_URL etc.
load_dotenv()

from config import settings

# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=False,
)
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG",
    enqueue=True,
)


# ──────────────────────────────────────────────────────────────────────
# Feed selection — MT5 on Windows, yfinance PaperFeed everywhere else
# ──────────────────────────────────────────────────────────────────────

def _mt5_available() -> bool:
    """Return True only when the MetaTrader5 package can be imported."""
    try:
        import MetaTrader5  # noqa: F401
        return True
    except ImportError:
        return False


def _make_feed(paper: bool):  # type: ignore[return]
    """Return the appropriate feed instance for the current platform."""
    if _mt5_available() and not paper:
        # Live mode on Windows with MT5
        from bot.data_feed import DataFeed
        return DataFeed(), False   # feed, is_paper_feed

    if _mt5_available() and paper:
        # Paper mode but MT5 is present — use DataFeed for data, skip execution
        from bot.data_feed import DataFeed
        return DataFeed(), False

    # MT5 unavailable (Linux / Railway / macOS without MT5)
    from bot.paper_feed import PaperFeed
    logger.info(
        "MetaTrader5 not available on {} — using PaperFeed (yfinance)",
        platform.system(),
    )
    return PaperFeed(), True      # feed, is_paper_feed


# ──────────────────────────────────────────────────────────────────────
# TradingBot
# ──────────────────────────────────────────────────────────────────────

class TradingBot:
    """Orchestrates the full trading pipeline."""

    _LOOP_INTERVAL = 60  # seconds

    def __init__(self, paper: bool = False) -> None:
        self._paper = paper
        self._running = False
        self._daily_stats: dict[str, Any] = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
        }
        self._last_reset_day: int = -1

        self._feed, self._is_paper_feed = _make_feed(paper)

        # For paper feeds, force paper mode regardless of CLI flag
        if self._is_paper_feed:
            self._paper = True

        logger.info(
            "Initialising TradingBot | paper={} feed={}",
            self._paper,
            type(self._feed).__name__,
        )

        mt5_ok = self._feed.initialize_mt5()
        if not mt5_ok:
            if not self._paper:
                logger.critical("MT5 initialisation failed — cannot run in LIVE mode")
                sys.exit(1)
            else:
                logger.warning(
                    "MT5 initialisation failed — running in paper/demo mode "
                    "with no live MT5 data"
                )

        from bot.executor import OrderExecutor
        from bot.news_filter import NewsFilter
        from bot.notifier import TelegramNotifier
        from bot.risk_manager import RiskManager
        from bot.strategy import LondonBreakoutStrategy

        self._executor = OrderExecutor(self._feed)
        self._risk: RiskManager | None = (
            RiskManager(self._feed, self._executor) if mt5_ok else None
        )
        self._strategy = LondonBreakoutStrategy(self._feed)
        self._news = NewsFilter()
        self._notifier = TelegramNotifier()

        from bot.database import get_db
        self._db = get_db()

        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        # Start dashboard server in background thread
        from bot.dashboard_server import start_dashboard, update_bot_state
        self._update_bot_state = update_bot_state
        update_bot_state(
            running=False,
            mode="paper" if self._paper else "live",
            feed_type=type(self._feed).__name__,
        )
        start_dashboard(feed_ref=self._feed)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown_handler(self, signum: int, frame: Any) -> None:
        logger.warning("Shutdown signal received — closing positions and exiting")
        self._running = False
        self._update_bot_state(running=False)
        try:
            if not self._paper:
                self._executor.close_all_positions(settings.symbol)
                logger.info("All positions closed on shutdown")
        except Exception as exc:
            logger.error("Error closing positions on shutdown: {}", exc)
        finally:
            self._db.log("INFO", "Bot stopped")
            self._feed.shutdown()
            sys.exit(0)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        now = datetime.now(tz=timezone.utc)
        if now.day != self._last_reset_day:
            if self._last_reset_day != -1:
                self._notifier.send_daily_summary(self._daily_stats)
            if self._risk is not None:
                self._risk.reset_daily_balance()
            self._daily_stats = {"total_trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0}
            self._last_reset_day = now.day
            logger.info("Daily reset complete | day={}", now.date())

    # ------------------------------------------------------------------
    # Friday close check
    # ------------------------------------------------------------------

    @staticmethod
    def _is_friday_close() -> bool:
        now = datetime.now(tz=timezone.utc)
        return now.weekday() == 4 and now.hour >= settings.friday_close_hour_utc

    # ------------------------------------------------------------------
    # Main loop iteration
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self._maybe_reset_daily()

        # a. Risk checks
        if self._risk is not None:
            if not self._risk.is_safe_to_trade(settings.symbol):
                report = self._risk.get_risk_report()
                self._notifier.send_risk_alert(report)
                logger.warning("Risk limits hit — skipping tick")
                return
        elif not self._is_paper_feed:
            logger.debug("Risk manager unavailable (no MT5) — skipping risk check")

        # b. News filter
        if self._news.high_impact_soon():
            nxt = self._news.get_next_event()
            logger.info("News filter active | next_event={}", nxt)
            return

        # c. Weekend close
        if self._is_friday_close():
            logger.info("Friday close time — closing all positions")
            if not self._paper:
                self._executor.close_all_positions(settings.symbol)
            return

        # d. Skip if already in a trade
        if self._is_paper_feed:
            from bot.paper_feed import PaperFeed
            assert isinstance(self._feed, PaperFeed)
            if self._feed.get_open_position() is not None:
                logger.debug("Paper position already open — skipping signal")
                return
        else:
            open_pos = self._executor.get_open_positions(settings.symbol)
            if open_pos:
                logger.debug("Position already open ({}) — skipping signal", len(open_pos))
                return

        # e. Hourly snapshot (paper feed only)
        if self._is_paper_feed:
            from bot.paper_feed import PaperFeed
            assert isinstance(self._feed, PaperFeed)
            self._feed.maybe_save_snapshot()

        # f. Get signal
        trade_signal = self._strategy.get_signal(settings.symbol)
        if trade_signal["direction"] == "NONE":
            logger.debug("No signal this tick")
            return

        now_ts = datetime.now(tz=timezone.utc).isoformat()
        self._update_bot_state(last_signal_ts=now_ts)
        self._db.log(
            "INFO",
            f"Signal: {trade_signal['direction']} | entry={trade_signal.get('entry')} "
            f"sl={trade_signal.get('stop_loss')} tp={trade_signal.get('take_profit')} "
            f"rsi={trade_signal.get('rsi', 'n/a')}"
        )

        # g. Lot size
        entry  = float(trade_signal["entry"])
        sl     = float(trade_signal["stop_loss"])
        pip    = 0.01 if settings.symbol.upper().endswith("JPY") else 0.0001
        sl_pips = abs(entry - sl) / pip
        lot = (
            self._risk.calculate_lot_size(settings.symbol, sl_pips)
            if self._risk is not None
            else round(min(settings.risk_per_trade * 10, settings.max_lot_size), 2)
        )

        # h. Paper mode — simulate and notify
        if self._paper:
            direction = str(trade_signal["direction"])
            tp = float(trade_signal["take_profit"])

            if self._is_paper_feed:
                from bot.paper_feed import PaperFeed
                assert isinstance(self._feed, PaperFeed)
                self._feed.open_position(
                    direction=direction,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    lot=lot,
                    sl_pips=sl_pips,
                    rsi=float(trade_signal.get("rsi", 0)),
                )
                summary = self._feed.account_summary()
            else:
                summary = f"balance=${settings.risk_per_trade * 100_000:,.0f} (MT5 paper)"

            msg = (
                f"<b>[PAPER] {direction}</b> {lot:.2f} lots\n"
                f"Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f}\n"
                f"Range: {trade_signal.get('range_pips', 0):.1f}pips | "
                f"RSI: {trade_signal.get('rsi', 50):.1f} | "
                f"Conf: {trade_signal.get('confidence', 'n/a')}\n"
                f"{summary}"
            )
            logger.info(
                "PAPER TRADE | {} {:.2f}lots entry={:.5f} sl={:.5f} tp={:.5f}",
                direction, lot, entry, sl, tp,
            )
            self._notifier.send(msg)
            self._db.log(
                "INFO",
                f"PAPER TRADE opened: {direction} {lot:.2f}lots @ {entry:.5f} "
                f"sl={sl:.5f} tp={tp:.5f}"
            )

            self._daily_stats["total_trades"] += 1
            self._daily_stats["net_pnl"] += 0.0  # resolved when position closes
            return

        # i. Live execution
        try:
            result = self._executor.place_order(
                symbol=settings.symbol,
                direction=str(trade_signal["direction"]),
                lot=lot,
                sl=float(trade_signal["stop_loss"]),
                tp=float(trade_signal["take_profit"]),
            )
            self._daily_stats["total_trades"] += 1
            self._notifier.send_trade_alert(trade_signal, lot)  # type: ignore[arg-type]
            logger.info("Trade executed | result={}", result)
            self._db.log("INFO", f"LIVE TRADE executed: {result}")
        except Exception as exc:
            logger.error("Order execution failed: {}", exc)
            self._notifier.send_error(exc)
            self._db.log("ERROR", f"Order execution failed: {exc}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        mode = "PAPER" if self._paper else "LIVE"
        feed_name = type(self._feed).__name__
        port = int(os.environ.get("PORT", 8080))
        railway_url = os.environ.get("RAILWAY_STATIC_URL") or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
        if railway_url:
            dashboard_url = f"https://{railway_url}"
        else:
            dashboard_url = f"http://localhost:{port}"

        logger.info(
            "TradingBot starting | mode={} feed={} symbol={} dashboard={}",
            mode, feed_name, settings.symbol, dashboard_url,
        )
        self._update_bot_state(running=True)
        self._db.log("INFO", f"Bot started — mode={mode} feed={feed_name} symbol={settings.symbol}")
        self._notifier.send(
            f"<b>Bot started</b> | mode={mode} | feed={feed_name} | symbol={settings.symbol}\n"
            f"Dashboard: {dashboard_url}"
        )

        while self._running:
            # Weekend — forex closed, no need to poll every 60 s
            if not self._strategy.is_market_open():
                logger.info("Weekend — bot sleeping 1 hour")
                time.sleep(3600)
                continue

            start = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Unhandled error in tick: {}", exc)
                self._notifier.send_error(exc)
                self._db.log("ERROR", f"Unhandled tick error: {exc}")

            elapsed  = time.monotonic() - start
            sleep_for = max(0.0, self._LOOP_INTERVAL - elapsed)
            logger.debug("Tick done in {:.2f}s — sleeping {:.1f}s", elapsed, sleep_for)
            time.sleep(sleep_for)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--paper",   "mode", flag_value="paper",   help="Run in paper mode")
@click.option("--live",    "mode", flag_value="live",    help="Run in live trading mode")
@click.option("--backtest","mode", flag_value="backtest", help="Run historical backtest")
@click.pass_context
def cli(ctx: click.Context, mode: str | None) -> None:
    """EUR/USD London Breakout algo bot.

    On Linux/Railway (no MT5) paper mode is used automatically.
    """
    if ctx.invoked_subcommand is None and mode:
        if mode == "paper":
            ctx.invoke(cmd_paper)
        elif mode == "live":
            ctx.invoke(cmd_live)
        elif mode == "backtest":
            ctx.invoke(cmd_backtest)
    elif ctx.invoked_subcommand is None and not mode:
        click.echo(ctx.get_help())


@cli.command("live")
def cmd_live() -> None:
    """Run in LIVE trading mode (real orders via MT5 — Windows only)."""
    if not _mt5_available():
        logger.error(
            "MetaTrader5 is not available on this platform. "
            "Live mode requires Windows + MT5 terminal. Use --paper instead."
        )
        sys.exit(1)
    bot = TradingBot(paper=False)
    bot.run()


@cli.command("paper")
def cmd_paper() -> None:
    """Run in PAPER mode — signals logged/notified, orders simulated."""
    bot = TradingBot(paper=True)
    bot.run()


@cli.command("backtest")
@click.option("--output", default="logs/backtest_report.html", help="Output HTML path")
def cmd_backtest(output: str) -> None:
    """Run a historical backtest and generate an HTML report."""
    from backtest.engine import Backtester
    from backtest.report import generate_report

    logger.info("Starting backtest")
    bt = Backtester()
    results = bt.run()
    generate_report(results, output_path=output)
    click.echo(f"Backtest complete. Report saved to: {output}")


if __name__ == "__main__":
    cli()
