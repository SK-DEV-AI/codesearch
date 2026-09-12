from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

import httpx

# ── Shared HTTP Client Pool ──────────────────────────────────────────────────
_http_client: httpx.AsyncClient | None = None
_http_client_lock = threading.Lock()


class _PoolStream(httpx.AsyncByteStream):
    """Adapt an httpcore response stream to httpx's stream interface."""
    def __init__(self, stream) -> None:
        self._stream = stream

    async def __aiter__(self):
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _PinningTransport(httpx.AsyncBaseTransport):
    """httpx transport with DNS-rebinding pinning, public APIs only.
    Replaces the old transport._pool poke (private — silently breaks on
    httpx upgrades; the M7 assert only caught it, this removes the need)."""
    def __init__(self, limits: httpx.Limits) -> None:
        from security import PinningNetworkBackend
        import httpcore
        self._pool = httpcore.AsyncConnectionPool(
            network_backend=PinningNetworkBackend(),
            max_keepalive_connections=limits.max_keepalive_connections,
            max_connections=limits.max_connections)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import httpcore
        req = httpcore.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.multi_items(),
            content=request.stream,
            extensions=request.extensions)
        resp = await self._pool.handle_async_request(req)
        return httpx.Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=_PoolStream(resp.stream),
            extensions=resp.extensions)

    async def aclose(self) -> None:
        await self._pool.aclose()


def get_http_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        with _http_client_lock:
            if _http_client is None or _http_client.is_closed:
                limits = httpx.Limits(
                    max_keepalive_connections=10, max_connections=20)
                _http_client = httpx.AsyncClient(
                    timeout=30.0,
                    transport=_PinningTransport(limits),
                )
    return _http_client


async def close_http_client():
    """Shut down the shared HTTP client. Call on server exit."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


class _KeyRotator:
    """Thread-safe round-robin key rotator for API keys."""

    def __init__(self, env_var: str, fallback_var: str = ""):
        raw = os.environ.get(env_var) or os.environ.get(fallback_var, "")
        self._keys: list[str] = [k.strip() for k in raw.split(",") if k.strip()] if raw else []
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str | None:
        if not self._keys:
            return None
        async with self._lock:
            k = self._keys[self._idx % len(self._keys)]
            self._idx = (self._idx + 1) % len(self._keys)
            return k

    @property
    def first(self) -> str:
        return self._keys[0] if self._keys else ""

    @property
    def has_keys(self) -> bool:
        return bool(self._keys)

CONTEXT7_API_KEY = os.environ.get("CONTEXT7_API_KEY", "")
CONTEXT7_SEARCH = "https://context7.com/api/v2/libs/search"
CONTEXT7_CONTEXT = "https://context7.com/api/v2/context"
DEEPWIKI_MCP = "https://mcp.deepwiki.com/mcp"

_gh_rotator = _KeyRotator("GITHUB_TOKEN", "GH_TOKEN")
_next_gh_key = _gh_rotator.next
GH_TOKEN = _gh_rotator.first
GH_SEARCH_CODE = "https://api.github.com/search/code"
GH_SEARCH_REPOS = "https://api.github.com/search/repositories"
GH_SEARCH_ISSUES = "https://api.github.com/search/issues"
GH_API = "https://api.github.com"
NV_KEY = os.environ.get("NV_KEY", "")
NV_BASE = "https://integrate.api.nvidia.com/v1"
NV_EMBED_MODEL = "nvidia/nv-embedcode-7b-v1"
NV_EMBED_DIMS = 4096
SO_API = "https://api.stackexchange.com/2.3"
SE_API_KEY = os.environ.get("SE_API_KEY", "")
SOFA_KEY = os.environ.get("SOFA_KEY", "")
SOFA_BASE = "https://agents.stackoverflow.com/api"

_li_rotator = _KeyRotator("LI_KEY")
_next_li_key = _li_rotator.next
LI_KEY = _li_rotator.first
LI_API = "https://libraries.io/api"

_oss_rotator = _KeyRotator("OSS_TOKEN")
_next_oss_key = _oss_rotator.next
OSS_TOKEN = _oss_rotator.first
OSS_API = "https://api.guide.sonatype.com/api/v3/component-report"


_tv_rotator = _KeyRotator("TAVILY_KEY", "TAVILY_KEYS")
_next_tv_key = _tv_rotator.next
GITHITS_API_TOKEN = os.environ.get("GITHITS_API_TOKEN", "")
GUIDE_API = "https://api.guide.sonatype.com"

REGISTRIES = {
    "npm": "https://registry.npmjs.org/{name}",
    "pypi": "https://pypi.org/pypi/{name}/json",
    "crates": "https://crates.io/api/v1/crates/{name}",
}
CRATES_SEARCH = "https://crates.io/api/v1/crates"
NPM_SEARCH = "https://registry.npmjs.org/-/v1/search"
# NOTE: docs.devdocs.io is a dead host (no DNS) — the API lives on the
# main host (verified live: /docs.json + /{slug}/index.json both 200).
DEVDOCS_API = "https://devdocs.io"
TAVILY_SEARCH = "https://api.tavily.com/search"
HN_API = "https://hn.algolia.com/api/v1"


async def _http_request(method: str, url: str, **kwargs) -> httpx.Response:
    """HTTP request with automatic retry on transient failures, redirect following, and timeout passthrough."""
    retries = kwargs.pop("retries", 2)
    timeout = kwargs.pop("timeout", 30.0)
    kwargs.setdefault("follow_redirects", True)
    last_err: Exception | None = None
    c = get_http_client()
    for attempt in range(retries + 1):
        try:
            if method == "GET":
                resp = await c.get(url, timeout=timeout, **kwargs)
            elif method == "POST":
                resp = await c.post(url, timeout=timeout, **kwargs)
            elif method == "PUT":
                resp = await c.put(url, timeout=timeout, **kwargs)
            else:
                raise ValueError(f"unsupported method {method!r}")  # L1
            if resp.status_code >= 500 and attempt < retries:
                last_err = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp)
                await asyncio.sleep(1 * (attempt + 1))
                continue
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            last_err = e
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
    raise last_err or RuntimeError("HTTP request failed after retries")


def api_error(prefix: str, r: httpx.Response, body_chars: int = 250) -> str:
    """Rich error string: status code + reason + truncated body + rate-limit headers."""
    parts = [f"{prefix} HTTP {r.status_code}"]
    if r.reason_phrase:
        parts.append(r.reason_phrase)
    body = (r.text or "").strip()
    if body:
        body_flat = " ".join(body.split())[:body_chars]
        if body_flat and body_flat != r.reason_phrase:
            parts.append(body_flat)
    if r.status_code in (403, 429):
        remaining = r.headers.get("x-ratelimit-remaining")
        reset = r.headers.get("x-ratelimit-reset")
        if remaining is not None:
            parts.append(f"rate-limit remaining: {remaining}")
        if reset is not None:
            try:
                when = time.strftime("%H:%M:%S UTC", time.gmtime(int(reset)))
                parts.append(f"resets: {when}")
            except (ValueError, OSError):
                parts.append(f"resets: {reset}")
    return " | ".join(parts)


# ── TTL Cache ───────────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}
_MAX_CACHE = 500
_CACHE_TTL = {"gh": 120, "c7": 300, "wiki": 600, "pkg": 60, "readme": 600, "emb": 600}
_cache_lock = asyncio.Lock()


async def _cached(key: str) -> Any | None:
    async with _cache_lock:
        entry = _cache.get(key)
        # gh_lang:/gh_topics:/gh_repo:/gh_rel: keys must hit the gh TTL:
        # split(":")[0] alone yields "gh_lang" (miss -> 300s stale).
        # NOTE: bare split("_")[0] would break "gh:..." keys, so chain both.
        if entry and time.monotonic() - entry[0] < _CACHE_TTL.get(key.split(":")[0].split("_")[0], 300):
            return entry[1]
    return None


async def _set_cache(key: str, val: Any):
    async with _cache_lock:
        _cache[key] = (time.monotonic(), val)
        if len(_cache) > _MAX_CACHE:
            sorted_keys = sorted(_cache, key=lambda k: _cache[k][0])
            evict_count = max(1, len(sorted_keys) // 4)
            for k in sorted_keys[:evict_count]:
                _cache.pop(k, None)
