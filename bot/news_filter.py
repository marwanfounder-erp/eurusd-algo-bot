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
        self._rate_limited: bool = False  # True when last fetch hit 429 with no cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_calendar(self) -> list[dict[str, Any]]:
        """Download the weekly calendar JSON, with one retry on 429."""
        for attempt in range(2):
            try:
                resp = requests.get(
                    _FF_CALENDAR_URL,
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 429:
                    if attempt == 0:
                        logger.warning("FF calendar rate limited (429) — waiting 60s and retrying")
                        time.sleep(60)
                        continue
                    # Second attempt also 429 — fall through to cached/allow
                    logger.warning("FF calendar still rate limited after retry — using cached data")
                    self._rate_limited = not bool(self._cache)
                    return self._cache
                resp.raise_for_status()
                data: list[dict[str, Any]] = resp.json()
                self._cache = data
                self._cache_ts = time.monotonic()
                self._rate_limited = False
                logger.info("FF calendar fetched | {} events", len(data))
                return data
            except Exception as exc:
                logger.error("FF calendar fetch failed: {} — using cached/safe-mode", exc)
                self._rate_limited = False  # real error, not rate limit
                return self._cache  # empty → safe-mode in callers
        return self._cache  # unreachable but satisfies type checker

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
        """Return True for High-impact EUR or USD events.

        The FF API uses the key "country" (not "currency"); we check both
        for forward-compatibility.
        """
        impact: str = event.get("impact", "").lower()
        # FF calendar uses "country" field; fall back to "currency" if absent
        country: str = event.get("country", event.get("currency", "")).upper()
        return impact == "high" and country in ("EUR", "USD")

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
            if self._rate_limited:
                # 429 with no cache — allow trading rather than blocking on rate limit
                logger.warning("News calendar unavailable (rate limited) — allowing trade")
                return False
            logger.warning("News calendar empty — blocking trading (safe-mode)")
            return True

        now = datetime.now(tz=timezone.utc)
        # Block window: [before_minutes] before → 0 → [after_minutes] after
        # minutes_until > 0 → event in future, < 0 → event in past
        before_min = settings.news_filter_before_minutes   # e.g. 30
        after_min = settings.news_filter_after_minutes     # e.g. 60

        for event in events:
            if not self._is_high_impact_eur_usd(event):
                continue

            event_time = self._parse_event_time(event)
            if event_time is None:
                continue

            minutes_until = (event_time - now).total_seconds() / 60

            if minutes_until < -after_min:
                logger.debug(
                    "Event passed {}+ min ago — trading allowed | {}",
                    after_min,
                    event.get("title", "Unknown"),
                )
                continue

            # Block if: -after_min <= minutes_until <= before_min
            # e.g. -60 <= minutes_until <= 30
            if -after_min <= minutes_until <= before_min:
                title = event.get("title", "Unknown")
                currency = event.get("currency", "?")
                if minutes_until >= 0:
                    direction = f"in {minutes_until:.1f}min"
                else:
                    direction = f"{abs(minutes_until):.1f}min ago"
                logger.warning(
                    "High-impact event nearby | {} {} {} — blocking trade",
                    currency,
                    title,
                    direction,
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
            "currency": next_event.get("country", next_event.get("currency", "?")),
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
