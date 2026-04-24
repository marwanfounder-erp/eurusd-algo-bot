"""Pytest configuration — mock MetaTrader5 before any import."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _build_mt5_mock() -> ModuleType:
    """Return a MagicMock module that satisfies all MetaTrader5 attribute lookups."""
    mt5 = MagicMock(name="MetaTrader5")

    # Constants used across the codebase
    mt5.TIMEFRAME_M1 = 1
    mt5.TIMEFRAME_M5 = 5
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_M30 = 30
    mt5.TIMEFRAME_H1 = 16385
    mt5.TIMEFRAME_H4 = 16388
    mt5.TIMEFRAME_D1 = 16408

    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.TRADE_ACTION_DEAL = 1
    mt5.TRADE_ACTION_SLTP = 6
    mt5.ORDER_FILLING_FOK = 0
    mt5.ORDER_FILLING_IOC = 1
    mt5.ORDER_TIME_GTC = 0
    mt5.TRADE_RETCODE_DONE = 10009

    # Default sensible returns
    mt5.initialize.return_value = True
    mt5.terminal_info.return_value = MagicMock()
    mt5.last_error.return_value = (0, "")
    mt5.positions_get.return_value = []

    # account_info mock
    account = MagicMock()
    account.login = 123456
    account.balance = 10_000.0
    account.equity = 10_000.0
    account.margin = 0.0
    account.margin_free = 10_000.0
    account.profit = 0.0
    account.margin_level = 0.0
    account.currency = "USD"
    mt5.account_info.return_value = account

    # symbol_info mock
    sym = MagicMock()
    sym.point = 0.00001
    sym.trade_contract_size = 100_000.0
    mt5.symbol_info.return_value = sym

    return mt5  # type: ignore[return-value]


# Install the mock before any production module is imported
_mt5_mock = _build_mt5_mock()
sys.modules.setdefault("MetaTrader5", _mt5_mock)


@pytest.fixture(autouse=True)
def reset_mt5_mock():
    """Reset relevant MT5 mock state between tests."""
    _mt5_mock.initialize.return_value = True
    _mt5_mock.terminal_info.return_value = MagicMock()
    _mt5_mock.last_error.return_value = (0, "")
    _mt5_mock.positions_get.return_value = []
    yield
