"""MT5 order execution — place, close, and query positions."""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]
except ImportError:
    mt5 = None  # type: ignore[assignment]

from bot.data_feed import DataFeed
from config import settings

# MT5 order type constants (resolved at runtime)
_ORDER_TYPE_BUY = 0
_ORDER_TYPE_SELL = 1
_TRADE_ACTION_DEAL = 1
_TRADE_ACTION_SLTP = 6
_ORDER_FILLING_FOK = 0
_ORDER_FILLING_IOC = 1


def _mt5_const(name: str, default: int = 0) -> int:
    if mt5 is None:
        return default
    return int(getattr(mt5, name, default))


class ExecutionError(RuntimeError):
    """Raised when an order cannot be placed after all retries."""


class OrderExecutor:
    """Sends, monitors, and closes MT5 orders."""

    def __init__(self, data_feed: DataFeed) -> None:
        self._feed = data_feed

    # ------------------------------------------------------------------
    # Slippage guard
    # ------------------------------------------------------------------

    def _check_slippage(self, symbol: str, direction: str, intended_price: float) -> bool:
        """Return True if current market price is within slippage tolerance."""
        try:
            tick = self._feed.get_tick(symbol)
        except Exception as exc:
            logger.error("Slippage check — tick fetch failed: {}", exc)
            return False

        pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
        current = tick["ask"] if direction == "BUY" else tick["bid"]
        slippage_pips = abs(current - intended_price) / pip

        if slippage_pips > settings.max_slippage_pips:
            logger.warning(
                "Slippage too high | intended={} current={} slippage={:.1f}pips (max {})",
                intended_price, current, slippage_pips, settings.max_slippage_pips,
            )
            return False

        logger.debug("Slippage OK | {:.1f}pips", slippage_pips)
        return True

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        direction: str,
        lot: float,
        sl: float,
        tp: float,
        comment: str = "LondonBreakout",
    ) -> dict[str, Any]:
        """Place a market order with stop-loss and take-profit.

        Retries up to ``settings.order_retry_attempts`` times on failure.
        Raises ExecutionError if all attempts fail.

        Returns
        -------
        dict with keys: ticket, price, volume, retcode, comment.
        """
        if mt5 is None:
            raise ExecutionError("MetaTrader5 package not available")

        order_type = _mt5_const("ORDER_TYPE_BUY") if direction == "BUY" else _mt5_const("ORDER_TYPE_SELL")
        filling = _mt5_const("ORDER_FILLING_FOK")

        # Current price for slippage check
        try:
            tick = self._feed.get_tick(symbol)
        except Exception as exc:
            raise ExecutionError(f"Cannot get tick for {symbol}: {exc}") from exc

        price = tick["ask"] if direction == "BUY" else tick["bid"]

        if not self._check_slippage(symbol, direction, price):
            raise ExecutionError(
                f"Slippage guard triggered for {symbol} {direction}"
            )

        request: dict[str, Any] = {
            "action": _mt5_const("TRADE_ACTION_DEAL"),
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": int(settings.max_slippage_pips * 10),  # in points
            "magic": settings.order_magic_id,
            "comment": comment,
            "type_time": _mt5_const("ORDER_TIME_GTC"),
            "type_filling": filling,
        }

        last_result: Any = None
        for attempt in range(1, settings.order_retry_attempts + 1):
            logger.info(
                "Placing order | attempt={} {} {} {} lot={} sl={} tp={}",
                attempt, symbol, direction, price, lot, sl, tp,
            )
            result = mt5.order_send(request)
            last_result = result

            if result is None:
                code, msg = mt5.last_error()
                logger.error("order_send returned None | code={} msg={}", code, msg)
            elif result.retcode == _mt5_const("TRADE_RETCODE_DONE", 10009):
                logger.info(
                    "Order placed | ticket={} price={} volume={}",
                    result.order, result.price, result.volume,
                )
                return {
                    "ticket": result.order,
                    "price": result.price,
                    "volume": result.volume,
                    "retcode": result.retcode,
                    "comment": result.comment,
                }
            else:
                logger.warning(
                    "Order failed | retcode={} comment={}",
                    result.retcode, result.comment,
                )

            if attempt < settings.order_retry_attempts:
                time.sleep(settings.order_retry_delay_seconds)
                # Refresh price for next attempt
                try:
                    tick = self._feed.get_tick(symbol)
                    request["price"] = tick["ask"] if direction == "BUY" else tick["bid"]
                except Exception:
                    pass

        retcode = last_result.retcode if last_result is not None else -1
        raise ExecutionError(
            f"Order failed after {settings.order_retry_attempts} attempts "
            f"(last retcode={retcode})"
        )

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_position(self, ticket: int) -> dict[str, Any]:
        """Close a single position by ticket number."""
        if mt5 is None:
            raise ExecutionError("MetaTrader5 package not available")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning("close_position — ticket {} not found", ticket)
            return {"success": False, "ticket": ticket, "reason": "not found"}

        pos = positions[0]
        # Reverse direction to close
        close_type = (
            _mt5_const("ORDER_TYPE_SELL")
            if pos.type == _mt5_const("ORDER_TYPE_BUY")
            else _mt5_const("ORDER_TYPE_BUY")
        )
        try:
            tick = self._feed.get_tick(pos.symbol)
        except Exception as exc:
            raise ExecutionError(f"Cannot get tick to close {ticket}: {exc}") from exc

        price = tick["bid"] if close_type == _mt5_const("ORDER_TYPE_SELL") else tick["ask"]

        request: dict[str, Any] = {
            "action": _mt5_const("TRADE_ACTION_DEAL"),
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": settings.order_magic_id,
            "comment": "close",
            "type_time": _mt5_const("ORDER_TIME_GTC"),
            "type_filling": _mt5_const("ORDER_FILLING_FOK"),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != _mt5_const("TRADE_RETCODE_DONE", 10009):
            retcode = result.retcode if result else -1
            logger.error("close_position failed | ticket={} retcode={}", ticket, retcode)
            return {"success": False, "ticket": ticket, "retcode": retcode}

        logger.info("Position closed | ticket={} price={}", ticket, result.price)
        return {"success": True, "ticket": ticket, "price": result.price}

    def close_all_positions(self, symbol: str) -> list[dict[str, Any]]:
        """Close every open position for *symbol*."""
        positions = self.get_open_positions(symbol)
        results = []
        for pos in positions:
            try:
                r = self.close_position(pos["ticket"])
                results.append(r)
            except Exception as exc:
                logger.error(
                    "close_all_positions — failed to close ticket {}: {}",
                    pos["ticket"], exc,
                )
                results.append({"success": False, "ticket": pos["ticket"], "error": str(exc)})
        logger.info(
            "close_all_positions | symbol={} closed={}/{}", symbol, len(results), len(positions)
        )
        return results

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_positions(self, symbol: str) -> list[dict[str, Any]]:
        """Return all open positions for *symbol* as a list of dicts."""
        if mt5 is None:
            return []

        self._feed.ensure_connected()
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            code, msg = mt5.last_error()
            logger.error(
                "get_open_positions failed | symbol={} code={} msg={}", symbol, code, msg
            )
            return []

        result = [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == _mt5_const("ORDER_TYPE_BUY") else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "comment": p.comment,
            }
            for p in positions
        ]
        logger.debug("Open positions | symbol={} count={}", symbol, len(result))
        return result
