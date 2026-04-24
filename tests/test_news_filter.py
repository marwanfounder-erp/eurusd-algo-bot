"""Tests for NewsFilter — caching, high-impact detection, safe-mode."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from bot.news_filter import NewsFilter, _CACHE_TTL_SECONDS, _FF_CALENDAR_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    currency: str = "USD",
    impact: str = "High",
    minutes_from_now: float = 60.0,
    title: str = "Non-Farm Payrolls",
) -> dict:
    """Build a synthetic FF calendar event dict."""
    dt = datetime.now(tz=timezone.utc) + timedelta(minutes=minutes_from_now)
    return {
        "title": title,
        "currency": currency,
        "impact": impact,
        "date": dt.isoformat(),
    }


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCaching:
    @patch("requests.get")
    def test_cache_populated_on_first_call(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = [_event()]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        nf._get_calendar()

        mock_get.assert_called_once_with(_FF_CALENDAR_URL, timeout=10)
        assert len(nf._cache) == 1

    @patch("requests.get")
    def test_cache_not_refreshed_within_ttl(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = [_event()]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        nf._get_calendar()  # populates cache
        nf._get_calendar()  # should NOT call requests.get again

        mock_get.assert_called_once()

    @patch("requests.get")
    def test_cache_refreshed_after_ttl(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = [_event()]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        nf._get_calendar()  # first fetch
        # Simulate expired cache
        nf._cache_ts = time.monotonic() - _CACHE_TTL_SECONDS - 1
        nf._get_calendar()  # should refetch

        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# high_impact_soon
# ---------------------------------------------------------------------------

class TestHighImpactSoon:
    @patch("requests.get")
    def test_blocks_when_event_within_before_window(self, mock_get):
        # Event 20 minutes away — within 30 min before window
        resp = MagicMock()
        resp.json.return_value = [_event(currency="USD", impact="High", minutes_from_now=20)]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.high_impact_soon() is True

    @patch("requests.get")
    def test_blocks_when_event_just_passed(self, mock_get):
        # Event happened 30 minutes ago — within 60 min after window
        resp = MagicMock()
        resp.json.return_value = [_event(currency="EUR", impact="High", minutes_from_now=-30)]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.high_impact_soon() is True

    @patch("requests.get")
    def test_allows_when_event_far_away(self, mock_get):
        # Event 120 minutes away — outside window
        resp = MagicMock()
        resp.json.return_value = [_event(currency="USD", impact="High", minutes_from_now=120)]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.high_impact_soon() is False

    @patch("requests.get")
    def test_ignores_medium_impact(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = [_event(currency="USD", impact="Medium", minutes_from_now=10)]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.high_impact_soon() is False

    @patch("requests.get")
    def test_ignores_non_eur_usd_currency(self, mock_get):
        resp = MagicMock()
        resp.json.return_value = [_event(currency="GBP", impact="High", minutes_from_now=10)]
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.high_impact_soon() is False

    @patch("requests.get")
    def test_safe_mode_on_api_failure(self, mock_get):
        """When the API fails and cache is empty → return True (block trading)."""
        mock_get.side_effect = RuntimeError("Network error")

        nf = NewsFilter()
        result = nf.high_impact_soon()
        assert result is True

    @patch("requests.get")
    def test_safe_mode_on_http_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.HTTPError("503")

        nf = NewsFilter()
        result = nf.high_impact_soon()
        assert result is True


# ---------------------------------------------------------------------------
# get_next_event
# ---------------------------------------------------------------------------

class TestGetNextEvent:
    @patch("requests.get")
    def test_returns_closest_upcoming_event(self, mock_get):
        events = [
            _event(currency="USD", impact="High", minutes_from_now=90, title="CPI"),
            _event(currency="EUR", impact="High", minutes_from_now=45, title="ECB Rate"),
            _event(currency="USD", impact="High", minutes_from_now=200, title="FOMC"),
        ]
        resp = MagicMock()
        resp.json.return_value = events
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        next_ev = nf.get_next_event()

        assert next_ev["title"] == "ECB Rate"
        assert next_ev["minutes_until"] == pytest.approx(45.0, abs=1.0)

    @patch("requests.get")
    def test_returns_empty_when_no_upcoming_events(self, mock_get):
        # All events in the past
        events = [
            _event(currency="USD", impact="High", minutes_from_now=-120),
        ]
        resp = MagicMock()
        resp.json.return_value = events
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.get_next_event() == {}

    @patch("requests.get")
    def test_returns_empty_on_api_failure(self, mock_get):
        mock_get.side_effect = RuntimeError("Timeout")

        nf = NewsFilter()
        result = nf.get_next_event()
        assert result == {}

    @patch("requests.get")
    def test_skips_low_and_medium_events(self, mock_get):
        events = [
            _event(currency="USD", impact="Low", minutes_from_now=30),
            _event(currency="EUR", impact="Medium", minutes_from_now=20),
        ]
        resp = MagicMock()
        resp.json.return_value = events
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        nf = NewsFilter()
        assert nf.get_next_event() == {}


# ---------------------------------------------------------------------------
# _is_high_impact_eur_usd
# ---------------------------------------------------------------------------

class TestIsHighImpact:
    @pytest.mark.parametrize("currency,impact,expected", [
        ("USD", "High", True),
        ("EUR", "High", True),
        ("GBP", "High", False),
        ("USD", "Medium", False),
        ("EUR", "Low", False),
        ("usd", "high", True),   # case-insensitive check
    ])
    def test_classification(self, currency, impact, expected):
        nf = NewsFilter()
        event = {"currency": currency, "impact": impact}
        assert nf._is_high_impact_eur_usd(event) is expected
