"""Simple in-process rate and ingestion limits for a local/demo API.

Not distributed; enforce global/IP-level limits in a trusted reverse proxy too.
"""
from __future__ import annotations

from collections import deque
from threading import Lock
import time

from fastapi import HTTPException

MAX_DOCUMENTS_PER_TENANT = 200
MAX_DOCUMENT_BYTES_PER_TENANT = 2_000_000


class SlidingWindowLimiter:
    def __init__(self, limit: int = 120, window_seconds: int = 60) -> None:
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit settings must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._entries: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, tenant: str) -> None:
        now = time.monotonic()
        with self._lock:
            # Bounded by the number of configured tenants.
            entries = self._entries.setdefault(tenant, deque())
            while entries and entries[0] <= now - self.window_seconds:
                entries.popleft()
            if len(entries) >= self.limit:
                wait = max(1, int(entries[0] + self.window_seconds - now) + 1)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(wait)},
                )
            entries.append(now)
