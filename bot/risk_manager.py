"""Risk management — daily loss, drawdown, and lot-size calculation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from config import settings

if TYPE_CHECKING:
    from bot.data_feed import DataFeed
    from bot.executor import OrderExecutor


class RiskManager:
    """Enforces FTMO/E8 drawdown rules and computes position sizing."""

    def __init__(self, data_feed: "DataFeed", executor: "OrderExecutor") -> None:
        self._feed = data_feed
        self._executor = executor

        account = self._feed.get_account_info()
        self._starting_balance: float = account["balance"]
        self._daily_starting_balance: float = account["balance"]

        logger.info(
            "RiskManager initialised | starting_balance={:.2f} daily_starting_balance={:.2f}",
            self._starting_balance,
            self._daily_starting_balance,
        )

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily_balance(self) -> None:
        """Refresh the daily starting balance — call at 00:00 UTC."""
        account = self._feed.get_account_info()
        self._daily_starting_balance = account["balance"]
        logger.info(
            "Daily balance reset | new_daily_starting_balance={:.2f}",
            self._daily_starting_balance,
        )

    # ------------------------------------------------------------------
    # Limit checks
    # ------------------------------------------------------------------

    def daily_loss_breached(self) -> bool:
        """Return True if today's loss exceeds the configured daily limit."""
        try:
            account = self._feed.get_account_info()
        except Exception as exc:
            logger.error("daily_loss_breached — account fetch failed: {}", exc)
            return True  # safe-mode: block trading

        equity = account["equity"]
        loss_pct = (self._daily_starting_balance - equity) / self._daily_starting_balance

        breached = loss_pct >= settings.max_daily_loss
        logger.debug(
            "Daily loss check | daily_start={:.2f} equity={:.2f} loss={:.2%} limit={:.2%} breached={}",
            self._daily_starting_balance,
            equity,
            loss_pct,
            settings.max_daily_loss,
            breached,
        )
        if breached:
            logger.warning(
                "DAILY LOSS LIMIT BREACHED | loss={:.2%} >= limit={:.2%}",
                loss_pct,
                settings.max_daily_loss,
            )
        return breached

    def drawdown_breached(self) -> bool:
        """Return True if total drawdown from initial balance exceeds limit."""
        try:
            account = self._feed.get_account_info()
        except Exception as exc:
            logger.error("drawdown_breached — account fetch failed: {}", exc)
            return True

        equity = account["equity"]
        dd_pct = (self._starting_balance - equity) / self._starting_balance

        breached = dd_pct >= settings.max_total_drawdown
        logger.debug(
            "Drawdown check | start={:.2f} equity={:.2f} dd={:.2%} limit={:.2%} breached={}",
            self._starting_balance,
            equity,
            dd_pct,
            settings.max_total_drawdown,
            breached,
        )
        if breached:
            logger.warning(
                "TOTAL DRAWDOWN LIMIT BREACHED | dd={:.2%} >= limit={:.2%}",
                dd_pct,
                settings.max_total_drawdown,
            )
        return breached

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        symbol: str,
        stop_loss_pips: float,
    ) -> float:
        """Compute lot size using fixed-fractional 1% risk model.

        Risk amount  = equity × risk_per_trade
        Pip value    = contract_size × point × 10  (for 5-digit broker)
        Lots         = risk_amount / (stop_loss_pips × pip_value_per_lot)

        The result is clamped to [0.01, max_lot_size] and rounded to 2 dp.
        """
        try:
            account = self._feed.get_account_info()
        except Exception as exc:
            logger.error("calculate_lot_size — account fetch failed: {}", exc)
            return 0.01

        equity = account["equity"]
        risk_amount = equity * settings.risk_per_trade

        # Try MT5 for exact pip value; fall back to the standard EURUSD.s constant
        pip_value_per_lot: float = 10.0  # $10 per pip per standard lot (EURUSD.s USD account)
        try:
            import MetaTrader5 as mt5  # type: ignore[import-untyped]
            sym_info = mt5.symbol_info(symbol)
            if sym_info is None:
                raise RuntimeError(f"symbol_info unavailable for {symbol}")
            pip_value_per_lot = sym_info.trade_contract_size * sym_info.point * 10
        except Exception as exc:
            logger.warning(
                "calculate_lot_size — MT5 symbol_info unavailable ({}): "
                "using default pip_value={} for {}",
                exc, pip_value_per_lot, symbol,
            )

        if pip_value_per_lot <= 0 or stop_loss_pips <= 0:
            logger.warning(
                "calculate_lot_size — invalid pip_value={} or sl_pips={}",
                pip_value_per_lot,
                stop_loss_pips,
            )
            return 0.01

        lots = risk_amount / (stop_loss_pips * pip_value_per_lot)
        lots = round(max(0.01, min(lots, settings.max_lot_size)), 2)

        logger.info(
            "Lot size | equity={:.2f} risk={:.2f} sl={:.1f}pips pip_val={:.2f} lots={}",
            equity,
            risk_amount,
            stop_loss_pips,
            pip_value_per_lot,
            lots,
        )
        return lots

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def is_safe_to_trade(self, symbol: str) -> bool:
        """Return True only when all risk conditions are satisfied."""
        if self.daily_loss_breached():
            logger.warning("is_safe_to_trade=False — daily loss limit")
            return False

        if self.drawdown_breached():
            logger.warning("is_safe_to_trade=False — drawdown limit")
            return False

        try:
            positions = self._executor.get_open_positions(symbol)
        except Exception as exc:
            logger.error("is_safe_to_trade — get_open_positions failed: {}", exc)
            return False

        if len(positions) >= settings.max_open_positions:
            logger.warning(
                "is_safe_to_trade=False — {} open positions (max {})",
                len(positions),
                settings.max_open_positions,
            )
            return False

        logger.debug("is_safe_to_trade=True")
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_risk_report(self) -> dict[str, float | str | bool]:
        """Return a snapshot of all current risk metrics."""
        try:
            account = self._feed.get_account_info()
            equity = account["equity"]
        except Exception:
            equity = 0.0

        daily_loss_pct = (
            (self._daily_starting_balance - equity) / self._daily_starting_balance
            if self._daily_starting_balance > 0
            else 0.0
        )
        total_dd_pct = (
            (self._starting_balance - equity) / self._starting_balance
            if self._starting_balance > 0
            else 0.0
        )

        report: dict[str, float | str | bool] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "starting_balance": self._starting_balance,
            "daily_starting_balance": self._daily_starting_balance,
            "current_equity": equity,
            "daily_loss_pct": round(daily_loss_pct, 4),
            "total_drawdown_pct": round(total_dd_pct, 4),
            "daily_loss_limit": settings.max_daily_loss,
            "drawdown_limit": settings.max_total_drawdown,
            "daily_loss_breached": self.daily_loss_breached(),
            "drawdown_breached": self.drawdown_breached(),
        }
        logger.info("Risk report | {}", report)
        return report
