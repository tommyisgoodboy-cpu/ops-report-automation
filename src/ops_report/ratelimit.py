"""Per-client rate limiting (sliding window, in memory).

Why: a public API can be hammered by anyone. Once an endpoint calls a paid model, every
request costs money, so limiting requests per client also limits the bill.

Limits of this design (on purpose, to keep it small):
- State lives in one process. With several Cloud Run instances each keeps its own count,
  so the real limit is N x instances. The demo runs with --max-instances 1.
  For a shared limit use a gateway (Cloud Armor / API Gateway) or a shared store (Redis).
- The client IP comes from the TCP peer by default. Behind a proxy set
  RATE_LIMIT_TRUST_PROXY=1 to use X-Forwarded-For instead (see client_ip).
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Callable

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit, self.window, self.max_keys, self.clock = limit, window, max_keys, clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns (allowed, seconds_until_retry)."""
        now = self.clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
                if len(self._hits) > self.max_keys:  # bound memory: drop the stalest client
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(key)
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, math.ceil(self.window - (now - hits[0])))
            hits.append(now)
            return True, 0


def client_ip(request: Request, trust_proxy: bool) -> str:
    """Identify the caller.

    X-Forwarded-For can be forged by the caller, so it is ignored unless a trusted proxy is
    known to be in front. Behind Cloud Run the load balancer appends the real client address,
    so the LAST entry is the one the proxy wrote; earlier entries are caller-controlled.
    """
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def make_limit_dependency(limiter: RateLimiter | None, trust_proxy: bool = False):
    """FastAPI dependency: raises 429 when the caller is over the limit. None = disabled."""

    def limit(request: Request) -> None:
        if limiter is None:
            return
        allowed, retry_after = limiter.check(client_ip(request, trust_proxy))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait and try again.",
                headers={"Retry-After": str(retry_after)},
            )

    return limit
