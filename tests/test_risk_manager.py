"""Tests for RiskManager — drawdown, daily loss, and lot-size logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk_manager(balance: float = 10_000.0, equity: float | None = None):
    """Return a RiskManager wired to mock feed/executor with the given balances."""
    from bot.risk_manager import RiskManager

    feed = MagicMock()
    executor = MagicMock()
    equity = equity if equity is not None else balance

    feed.get_account_info.return_value = {
        "balance": balance,
        "equity": equity,
        "margin": 0.0,
        "free_margin": equity,
        "profit": equity - balance,
        "margin_level": 0.0,
    }

    rm = RiskManager(data_feed=feed, executor=executor)
    return rm, feed, executor


# ---------------------------------------------------------------------------
# daily_loss_breached
# ---------------------------------------------------------------------------

class TestDailyLossBreach:
    def test_no_breach_when_below_limit(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        # Equity dropped 3% — limit is 4%
        feed.get_account_info.return_value["equity"] = 9_700.0
        assert rm.daily_loss_breached() is False

    def test_breach_at_exact_limit(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        # Exactly 4% loss
        feed.get_account_info.return_value["equity"] = 9_600.0
        assert rm.daily_loss_breached() is True

    def test_breach_above_limit(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_500.0
        assert rm.daily_loss_breached() is True

    def test_safe_mode_on_exception(self):
        rm, feed, _ = _make_risk_manager()
        feed.get_account_info.side_effect = RuntimeError("MT5 error")
        # Should return True (block trading) on exception
        assert rm.daily_loss_breached() is True


# ---------------------------------------------------------------------------
# drawdown_breached
# ---------------------------------------------------------------------------

class TestDrawdownBreach:
    def test_no_breach_below_limit(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_300.0  # 7% dd
        assert rm.drawdown_breached() is False  # limit 8%

    def test_breach_at_limit(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_200.0  # 8% dd
        assert rm.drawdown_breached() is True

    def test_safe_mode_on_exception(self):
        rm, feed, _ = _make_risk_manager()
        feed.get_account_info.side_effect = RuntimeError("MT5 error")
        assert rm.drawdown_breached() is True


# ---------------------------------------------------------------------------
# calculate_lot_size
# ---------------------------------------------------------------------------

class TestLotSize:
    def _make_sym_info(self, contract_size: float = 100_000.0, point: float = 0.00001):
        sym = MagicMock()
        sym.trade_contract_size = contract_size
        sym.point = point
        return sym

    @patch("MetaTrader5.symbol_info")
    def test_lot_size_1pct_risk(self, mock_sym_info):
        mock_sym_info.return_value = self._make_sym_info()
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 10_000.0

        # risk = 10000 * 0.01 = 100
        # pip_value = 100000 * 0.00001 * 10 = 10
        # lots = 100 / (20 * 10) = 0.50
        lot = rm.calculate_lot_size("EURUSD.s", stop_loss_pips=20.0)
        assert lot == pytest.approx(0.50, abs=0.01)

    @patch("MetaTrader5.symbol_info")
    def test_lot_capped_at_max(self, mock_sym_info):
        mock_sym_info.return_value = self._make_sym_info()
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 10_000.0

        # With 1 pip SL, raw lot would be huge → should be capped at max_lot_size
        lot = rm.calculate_lot_size("EURUSD.s", stop_loss_pips=0.1)
        assert lot <= 5.0

    @patch("MetaTrader5.symbol_info")
    def test_lot_minimum_0_01(self, mock_sym_info):
        mock_sym_info.return_value = self._make_sym_info()
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 1.0  # tiny equity

        lot = rm.calculate_lot_size("EURUSD.s", stop_loss_pips=100.0)
        assert lot >= 0.01

    @patch("MetaTrader5.symbol_info")
    def test_invalid_sl_pips_returns_minimum(self, mock_sym_info):
        mock_sym_info.return_value = self._make_sym_info()
        rm, feed, _ = _make_risk_manager(balance=10_000)
        lot = rm.calculate_lot_size("EURUSD.s", stop_loss_pips=0.0)
        assert lot == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# is_safe_to_trade
# ---------------------------------------------------------------------------

class TestIsSafeToTrade:
    def test_safe_when_all_ok(self):
        rm, feed, executor = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 10_000.0
        executor.get_open_positions.return_value = []
        assert rm.is_safe_to_trade("EURUSD.s") is True

    def test_not_safe_daily_loss(self):
        rm, feed, executor = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_500.0  # 5% loss
        executor.get_open_positions.return_value = []
        assert rm.is_safe_to_trade("EURUSD.s") is False

    def test_not_safe_max_positions(self):
        rm, feed, executor = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 10_000.0
        # 3 positions = max_open_positions reached
        executor.get_open_positions.return_value = [MagicMock(), MagicMock(), MagicMock()]
        assert rm.is_safe_to_trade("EURUSD.s") is False

    def test_not_safe_drawdown(self):
        rm, feed, executor = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_100.0  # 9% dd > 8% limit
        executor.get_open_positions.return_value = []
        assert rm.is_safe_to_trade("EURUSD.s") is False


# ---------------------------------------------------------------------------
# reset_daily_balance
# ---------------------------------------------------------------------------

class TestResetDailyBalance:
    def test_reset_updates_balance(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        # Simulate overnight trading — balance grew to 10_200
        feed.get_account_info.return_value["balance"] = 10_200.0
        feed.get_account_info.return_value["equity"] = 10_200.0

        rm.reset_daily_balance()
        assert rm._daily_starting_balance == pytest.approx(10_200.0)


# ---------------------------------------------------------------------------
# get_risk_report
# ---------------------------------------------------------------------------

class TestGetRiskReport:
    def test_report_keys_present(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_800.0
        report = rm.get_risk_report()

        expected_keys = {
            "timestamp",
            "starting_balance",
            "daily_starting_balance",
            "current_equity",
            "daily_loss_pct",
            "total_drawdown_pct",
            "daily_loss_limit",
            "drawdown_limit",
            "daily_loss_breached",
            "drawdown_breached",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_report_values_correct(self):
        rm, feed, _ = _make_risk_manager(balance=10_000)
        feed.get_account_info.return_value["equity"] = 9_800.0
        report = rm.get_risk_report()

        assert report["current_equity"] == pytest.approx(9_800.0)
        assert report["daily_loss_pct"] == pytest.approx(0.02, abs=0.001)
