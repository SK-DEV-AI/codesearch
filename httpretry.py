"""Shared, bounded retry helper for transient HTTP failures.

Used by the codesearch lanes (libraries.io, HN Algolia, SOFA, Semantic
Scholar) so a single 429/503/timeout does not collapse a whole lane.

Design (AWS exponential-backoff-and-jitter + Retry-After honouring):
- Retry ONLY transient conditions: 429, 500, 502, 503, 504,
  httpx.TimeoutException, httpx.RequestError. Never 4xx the caller caused.
- At most ``max_extra`` (default 2) extra attempts.
- Exponential backoff with full jitter, capped at 5s, unless the server sends
  ``Retry-After`` (honoured, capped at 5s).
- No shared mutable state: each call is independent (concurrency safe).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_EXTRA = 2
WAIT_CAP_SECONDS = 5.0


def parse_retry_after(headers: Any, cap: float = WAIT_CAP_SECONDS) -> float | None:
    """Return a capped wait (seconds) if Retry-After is present and parseable."""
    raw = None
    for key in ("retry-after", "Retry-After", "RETRY-AFTER"):
        try:
            v = headers.get(key)
        except Exception:
            v = None
        if v is not None:
            raw = v
            break
    if raw is None:
        try:
            for k, v in headers.items():
                if str(k).lower() == "retry-after":
                    raw = v
                    break
        except Exception:
            raw = None
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val < 0:
        val = 0.0
    return min(val, cap)


def _backoff(attempt: int, cap: float = WAIT_CAP_SECONDS) -> float:
    # Full jitter: uniform(0, min(cap, base * 2**attempt)).
    base = 1.0
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(0.0, ceiling)


async def retry_request(
    method: str,
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
    max_extra: int = DEFAULT_MAX_EXTRA,
    **kwargs: Any,
) -> tuple[httpx.Response, int]:
    """Perform an HTTP request with bounded retries on transient failures.

    Returns ``(response, retries_used)``. ``retries_used`` is the number of
    extra attempts actually performed (0 when the first attempt succeeded).
    On exhaustion of network-level retries the last exception is re-raised.
    """
    if client is None:
        from config import get_http_client

        client = get_http_client()

    method = method.upper()
    retries_used = 0
    last_exc: Exception | None = None

    for attempt in range(max_extra + 1):
        try:
            if method == "GET":
                resp = await client.get(url, timeout=timeout, **kwargs)
            elif method == "POST":
                resp = await client.post(url, timeout=timeout, **kwargs)
            elif method == "PUT":
                resp = await client.put(url, timeout=timeout, **kwargs)
            elif method == "DELETE":
                resp = await client.delete(url, timeout=timeout, **kwargs)
            elif method == "PATCH":
                resp = await client.patch(url, timeout=timeout, **kwargs)
            else:
                resp = await client.request(method, url, timeout=timeout, **kwargs)
        except (httpx.TimeoutException, httpx.RequestError) as e:
            last_exc = e
            if attempt < max_extra:
                retries_used += 1
                await asyncio.sleep(_backoff(attempt))
                continue
            raise

        if resp.status_code in TRANSIENT_STATUSES and attempt < max_extra:
            retries_used += 1
            wait = parse_retry_after(resp.headers)
            if wait is None:
                wait = _backoff(attempt)
            await asyncio.sleep(wait)
            continue

        return resp, retries_used

    if last_exc is not None:
        raise last_exc
    # Unreachable in practice; kept for type completeness.
    raise RuntimeError("HTTP request failed after retries")