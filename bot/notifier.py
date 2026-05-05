"""Telegram notifier — non-blocking, queue-backed."""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from loguru import logger

from config import settings


class TelegramNotifier:
    """Sends Telegram messages from a background thread with an internal queue.

    Messages are enqueued and dispatched asynchronously so the main trading
    loop is never blocked by network I/O.  If Telegram is unreachable the
    queue drains when connectivity is restored.
    """

    _MAX_QUEUE = 100

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue(maxsize=self._MAX_QUEUE)
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.warning(
                "Telegram not configured — notifications disabled "
                "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
            )
            return

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="TelegramNotifier"
        )
        self._thread.start()
        logger.info("TelegramNotifier started")

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Consume the queue and dispatch messages to Telegram."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            try:
                message = self._queue.get(timeout=1)
                loop.run_until_complete(self._dispatch(message))
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("TelegramNotifier worker error: {}", exc)

    async def _dispatch(self, message: str) -> None:
        """Send a single message via the Telegram Bot API."""
        try:
            from telegram import Bot  # type: ignore[import-untyped]
            bot = Bot(token=self._token)
            await bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode="HTML",
            )
            logger.debug("Telegram message sent ({} chars)", len(message))
        except Exception as exc:
            logger.error("Telegram dispatch failed: {}", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: str) -> None:
        """Enqueue a plain-text message (HTML markup allowed)."""
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            logger.warning("Telegram queue full — dropping message")

    def send_trade_alert(self, signal: dict[str, Any], lot: float) -> None:
        """Format and enqueue a trade entry notification."""
        direction = signal.get("direction", "?")
        entry = signal.get("entry", 0.0)
        sl = signal.get("stop_loss", 0.0)
        tp = signal.get("take_profit", 0.0)
        rsi = signal.get("rsi", 0.0)
        confidence = signal.get("confidence", "?")
        range_pips = signal.get("range_pips", 0.0)
        strategy = signal.get("strategy", "LB")

        if strategy == "LB":
            strategy_label = "London Breakout"
            range_label = f"Asian range: {range_pips:.1f} pips"
        else:
            window = signal.get("window", strategy)
            strategy_label = f"Silver Bullet ({window})"
            range_label = f"FVG size: {range_pips:.1f} pips"

        emoji = "\U0001f7e2" if direction == "BUY" else "\U0001f534"
        message = (
            f"{emoji} <b>TRADE EXECUTED</b>\n"
            f"Strategy : <b>{strategy_label}</b>\n"
            f"Symbol : <code>{settings.symbol}</code>\n"
            f"Direction : <b>{direction}</b>\n"
            f"Entry  : <code>{entry:.5f}</code>\n"
            f"SL     : <code>{sl:.5f}</code>\n"
            f"TP     : <code>{tp:.5f}</code>\n"
            f"Lots   : <code>{lot}</code>\n"
            f"RSI    : <code>{rsi:.2f}</code>\n"
            f"Confidence : {confidence}\n"
            f"{range_label}"
        )
        self.send(message)

    def send_risk_alert(self, report: dict[str, Any]) -> None:
        """Format and enqueue a risk-limit warning."""
        daily_pct = float(report.get("daily_loss_pct", 0)) * 100
        dd_pct = float(report.get("total_drawdown_pct", 0)) * 100
        equity = float(report.get("current_equity", 0))
        message = (
            "\u26a0\ufe0f <b>RISK ALERT</b>\n"
            f"Daily loss  : <code>{daily_pct:.2f}%</code> "
            f"(limit {settings.max_daily_loss*100:.0f}%)\n"
            f"Total DD    : <code>{dd_pct:.2f}%</code> "
            f"(limit {settings.max_total_drawdown*100:.0f}%)\n"
            f"Current equity : <code>{equity:.2f}</code>"
        )
        self.send(message)

    def send_daily_summary(self, stats: dict[str, Any]) -> None:
        """Format and enqueue an end-of-day summary."""
        trades = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pnl = float(stats.get("net_pnl", 0.0))
        win_rate = (wins / trades * 100) if trades > 0 else 0.0
        message = (
            "\U0001f4ca <b>DAILY SUMMARY</b>\n"
            f"Trades   : {trades}\n"
            f"Wins     : {wins} | Losses: {losses}\n"
            f"Win rate : {win_rate:.1f}%\n"
            f"Net P&L  : <code>{pnl:+.2f}</code>"
        )
        self.send(message)

    def send_error(self, error: Exception) -> None:
        """Format and enqueue an error notification."""
        message = (
            "\U0001f6a8 <b>BOT ERROR</b>\n"
            f"<code>{type(error).__name__}: {error}</code>"
        )
        self.send(message)
