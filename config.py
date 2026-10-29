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
# NOTE 2026-10-15: nv-embedcode-7b-v1 went 410 Gone (EOL 2026-08-25) like the
# websearch model. Unified on nemotron-3-embed-1b (200 live, general text).
# Code-tuned split is gone until Nvidia ships a code-embed successor.
NV_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
# Reranker model override (parity with websearch `run`). Currently advisory:
# the Rust worker bakes its model at compile time and ignores this env, but
# keeping the knob aligned means a future env-aware worker needs no config
# change on this side.
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
NV_EMBED_DIMS = 2048
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


async def _http_request_capped(method: str, url: str, *, max_bytes: int = 524_288,
                               **kwargs) -> httpx.Response:
    """HTTP request with a hard byte cap on the response body.

    Streams the response and aborts the download as soon as max_bytes are
    read, so a hostile or oversized endpoint can never buffer an unbounded
    body in RAM. Retries on transient failures like _http_request does.

    The returned response is a synthetic httpx.Response carrying the capped
    body — content-encoding/content-length headers are dropped because the
    body is already decoded. Reading .text/.json() on it is safe: the body
    is at most max_bytes long.
    """
    retries = kwargs.pop("retries", 2)
    timeout = kwargs.pop("timeout", 30.0)
    kwargs.setdefault("follow_redirects", True)
    last_err: Exception | None = None
    c = get_http_client()
    for attempt in range(retries + 1):
        try:
            async with c.stream(method, url, timeout=timeout, **kwargs) as resp:
                body = bytearray()
                truncated = False
                async for chunk in resp.aiter_bytes():
                    room = max_bytes - len(body)
                    if room <= 0:
                        truncated = True
                        break
                    if len(chunk) > room:
                        body += chunk[:room]
                        truncated = True
                        break
                    body += chunk
                if resp.status_code >= 500 and attempt < retries:
                    last_err = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp)
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                headers = {k: v for k, v in resp.headers.items()
                           if k.lower() not in ("content-encoding", "content-length")}
                if truncated:
                    headers["x-jcode-body-truncated"] = "1"
                return httpx.Response(
                    status_code=resp.status_code,
                    headers=headers,
                    content=bytes(body),
                    request=resp.request)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            last_err = e
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
    raise last_err or RuntimeError("HTTP request failed after retries")


def body_truncated(r: httpx.Response) -> bool:
    """True when the response body was cut off by the byte cap."""
    return r.headers.get("x-jcode-body-truncated") == "1"


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
# TTL policy per cache "kind". A call site may pass ttl= explicitly
# (preferred); otherwise the kind is inferred from the key prefix.
_CACHE_TTL = {"gh": 120, "c7": 300, "wiki": 600, "pkg": 60, "readme": 600, "emb": 600}
# Default TTL when no kind matches and no explicit ttl is given.
_DEFAULT_CACHE_TTL = 300
_cache: dict[str, tuple[float, float, Any]] = {}  # key -> (stored_at, ttl, value)
# Running total of _approx_bytes over the stored values, so the byte budget is
# checked in O(1) instead of re-measuring every entry on every insert.
_cache_bytes = 0
_MAX_CACHE = 500
# Approximate byte budget for the whole cache. 2048-dim embedding vectors are
# ~16KB each, so 500 of them is ~8MB; bound total at 64MB to leave headroom
# on a laptop-class box (audit rec #4).
_MAX_CACHE_BYTES = 64 * 1024 * 1024


def _approx_bytes(val: Any) -> int:
    """Cheap size estimate for cache eviction weighting."""
    if isinstance(val, list):
        if val and isinstance(val[0], (int, float)):
            return len(val) * 8
        return sum(_approx_bytes(x) for x in val) + 64
    if isinstance(val, dict):
        return sum(_approx_bytes(v) for v in val.values()) + len(val) * 32
    if isinstance(val, str):
        return len(val)
    if isinstance(val, (int, float)):
        return 8
    return 64
_cache_lock = asyncio.Lock()


def _ttl_for_key(key: str) -> float:
    """Infer a TTL from the key prefix (legacy heuristic).

    Compound keys like ``gh_repo:...`` must resolve to the ``gh`` kind, so we
    split on ':' first, then '_' (``gh_repo`` -> ``gh``). A bare ``split("_")``
    would break ``gh:...`` keys, hence the chained split.
    """
    kind = key.split(":")[0].split("_")[0]
    return float(_CACHE_TTL.get(kind, _DEFAULT_CACHE_TTL))


async def _cached(key: str, ttl: float | None = None) -> Any | None:
    """Return a cached value if present and not expired.

    `ttl` (seconds) overrides the inferred policy for this key. Pass it at the
    call site to make the caching policy explicit instead of relying on the
    prefix heuristic.
    """
    async with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        stored_at, entry_ttl, value = entry
        effective_ttl = ttl if ttl is not None else entry_ttl
        if time.monotonic() - stored_at < effective_ttl:
            return value
    return None


async def _set_cache(key: str, val: Any, ttl: float | None = None):
    """Store a value, recording the TTL it was stored with.

    `ttl` (seconds) makes the policy explicit at the call site. When omitted we
    infer from the key prefix (legacy) so existing callers keep working.
    """
    global _cache_bytes
    async with _cache_lock:
        effective_ttl = ttl if ttl is not None else _ttl_for_key(key)
        previous = _cache.get(key)
        if previous is not None:
            _cache_bytes -= _approx_bytes(previous[2])
        _cache[key] = (time.monotonic(), effective_ttl, val)
        _cache_bytes += _approx_bytes(val)
        # Evict oldest entries until both the count cap and the byte budget hold.
        while len(_cache) > _MAX_CACHE or _cache_bytes > _MAX_CACHE_BYTES:
            if not _cache:
                break
            oldest = min(_cache, key=lambda k: _cache[k][0])
            _cache_bytes -= _approx_bytes(_cache.pop(oldest)[2])
