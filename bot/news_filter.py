"""Forex Factory news filter — blocks trading around high-impact events."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from config import settings

_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_CACHE_TTL_SECONDS = 3600  # 1 hour


class NewsFilter:
    """Fetches the Forex Factory weekly calendar and checks for high-impact events."""

    def __init__(self) -> None:
        self._cache: list[dict[str, Any]] = []
        self._cache_ts: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_calendar(self) -> list[dict[str, Any]]:
        """Download the weekly calendar JSON and update the cache."""
        try:
            resp = requests.get(_FF_CALENDAR_URL, timeout=10)
            resp.raise_for_status()
            data: list[dict[str, Any]] = resp.json()
            self._cache = data
            self._cache_ts = time.monotonic()
            logger.info("FF calendar fetched | {} events", len(data))
            return data
        except Exception as exc:
            logger.error("FF calendar fetch failed: {} — using cached/safe-mode", exc)
            return self._cache  # may be empty → safe-mode in callers

    def _get_calendar(self) -> list[dict[str, Any]]:
        """Return cached data or re-fetch if TTL expired."""
        age = time.monotonic() - self._cache_ts
        if not self._cache or age > _CACHE_TTL_SECONDS:
            return self._fetch_calendar()
        logger.debug("Using cached FF calendar (age={:.0f}s)", age)
        return self._cache

    @staticmethod
    def _parse_event_time(event: dict[str, Any]) -> datetime | None:
        """Parse the event datetime string into a UTC-aware datetime."""
        raw: str = event.get("date", "")
        if not raw:
            return None
        try:
            # FF format: "2024-01-15T08:30:00-05:00" (Eastern time)
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_high_impact_eur_usd(event: dict[str, Any]) -> bool:
        """Return True for High-impact EUR or USD events."""
        impact: str = event.get("impact", "").lower()
        currency: str = event.get("currency", "").upper()
        return impact == "high" and currency in ("EUR", "USD")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def high_impact_soon(self) -> bool:
        """Return True if a high-impact EUR/USD event is within the filter window.

        Blocks trading from ``news_filter_before_minutes`` before the event
        to ``news_filter_after_minutes`` after.  Returns True (safe-mode) on
        any API or parse failure.
        """
        events = self._get_calendar()
        if not events:
            # Empty means fetch failed — block trading (safe-mode)
            logger.warning("News calendar empty — blocking trading (safe-mode)")
            return True

        now = datetime.now(tz=timezone.utc)
        before_secs = settings.news_filter_before_minutes * 60
        after_secs = settings.news_filter_after_minutes * 60

        for event in events:
            if not self._is_high_impact_eur_usd(event):
                continue

            event_time = self._parse_event_time(event)
            if event_time is None:
                continue

            delta = (event_time - now).total_seconds()
            # delta > 0 → event in future; delta < 0 → event in past
            if -after_secs <= delta <= before_secs:
                title = event.get("title", "Unknown")
                currency = event.get("currency", "?")
                minutes_away = delta / 60
                logger.warning(
                    "High-impact event nearby | {} {} in {:.1f}min — blocking trade",
                    currency,
                    title,
                    minutes_away,
                )
                return True

        logger.debug("No high-impact events in filter window")
        return False

    def get_next_event(self) -> dict[str, Any]:
        """Return the next upcoming high-impact EUR/USD event and minutes until it.

        Returns an empty dict if there are no upcoming events or the calendar
        is unavailable.
        """
        events = self._get_calendar()
        if not events:
            return {}

        now = datetime.now(tz=timezone.utc)
        upcoming: list[tuple[float, dict[str, Any]]] = []

        for event in events:
            if not self._is_high_impact_eur_usd(event):
                continue
            event_time = self._parse_event_time(event)
            if event_time is None:
                continue
            delta_seconds = (event_time - now).total_seconds()
            if delta_seconds > 0:
                upcoming.append((delta_seconds, event))

        if not upcoming:
            logger.debug("No upcoming high-impact events found")
            return {}

        upcoming.sort(key=lambda t: t[0])
        seconds_until, next_event = upcoming[0]
        result = {
            "title": next_event.get("title", "Unknown"),
            "currency": next_event.get("currency", "?"),
            "minutes_until": round(seconds_until / 60, 1),
            "event_time_utc": self._parse_event_time(next_event),
        }
        logger.info(
            "Next event | {} {} in {:.1f}min",
            result["currency"],
            result["title"],
            result["minutes_until"],
        )
        return result
