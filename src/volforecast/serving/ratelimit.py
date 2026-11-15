"""In-process token bucket per client IP (no external dependency).

Each bucket holds up to ``burst`` tokens and refills at ``rate`` tokens/second; a request costs one
token. State is per worker process. Idle buckets (full again) are evicted so memory is bounded by
the number of *recently active* clients, with a hard cap as a backstop.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from fastapi import Request

from volforecast.serving.errors import ApiError
from volforecast.serving.settings import ApiSettings


class TokenBucketLimiter:
    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 50_000,
    ) -> None:
        self.rate = rate
        self.burst = float(burst)
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last refill)
        self._lock = threading.Lock()
        self._last_sweep = clock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def acquire(self, key: str) -> float:
        """Take one token. Returns 0.0 if allowed, else the seconds until a token is available."""
        if not self.enabled:
            return 0.0
        now = self._clock()
        with self._lock:
            self._sweep(now)
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1.0 - tokens) / self.rate

    def _sweep(self, now: float) -> None:
        full_after = self.burst / self.rate
        if now - self._last_sweep < max(full_after, 1.0) and len(self._buckets) < self._max_keys:
            return
        self._last_sweep = now
        stale = [k for k, (_, last) in self._buckets.items() if now - last >= full_after]
        for k in stale:
            del self._buckets[k]
        if len(self._buckets) >= self._max_keys:  # backstop: drop the oldest quarter
            oldest = sorted(self._buckets, key=lambda k: self._buckets[k][1])
            for k in oldest[: self._max_keys // 4]:
                del self._buckets[k]

    def __len__(self) -> int:
        return len(self._buckets)


def client_ip(request: Request, trusted_hops: int) -> str:
    """Client address; ``X-Forwarded-For`` is honoured only behind ``trusted_hops`` proxies.

    With N trusted proxies each appending the peer address it saw, the client is the Nth entry
    from the right; anything further left is client-controlled and ignored.
    """
    if trusted_hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
        parts = [p for p in parts if p]
        if len(parts) >= trusted_hops:
            return parts[-trusted_hops]
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(request: Request) -> None:
    """Router dependency: 429 + ``Retry-After`` when the client's bucket is empty.

    Applied to the ``/v1`` router only, so probes and ``/metrics`` are never limited. Because it
    runs after routing, 429 responses carry the right route label in the metrics.
    """
    limiter: TokenBucketLimiter = request.app.state.limiter
    api: ApiSettings = request.app.state.api
    wait = limiter.acquire(client_ip(request, api.trusted_proxy_hops))
    if wait > 0:
        raise ApiError(
            429,
            "rate_limited",
            "rate limit exceeded; retry later",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )
