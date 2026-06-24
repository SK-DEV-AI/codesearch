from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx


class _KeyRotator:
    """Thread-safe round-robin key rotator for API keys."""

    def __init__(self, env_var: str, fallback_var: str = ""):
        raw = os.environ.get(env_var, os.environ.get(fallback_var, "")) if fallback_var else os.environ.get(env_var, "")
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

_fc_rotator = _KeyRotator("FIRECRAWL_KEY", "FIRECRAWL_KEYS")
_next_fc_key = _fc_rotator.next

_tv_rotator = _KeyRotator("TAVILY_KEY", "TAVILY_KEYS")
_next_tv_key = _tv_rotator.next
GUIDE_API = "https://api.guide.sonatype.com"

REGISTRIES = {
    "npm": "https://registry.npmjs.org/{name}",
    "pypi": "https://pypi.org/pypi/{name}/json",
    "crates": "https://crates.io/api/v1/crates/{name}",
}
CRATES_SEARCH = "https://crates.io/api/v1/crates"
NPM_SEARCH = "https://registry.npmjs.org/-/v1/search"
DEVDOCS_API = "https://docs.devdocs.io"
FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v2/search"
TAVILY_SEARCH = "https://api.tavily.com/search"
HN_API = "https://hn.algolia.com/api/v1"


async def _http_request(method: str, url: str, **kwargs) -> httpx.Response:
    """HTTP request with automatic retry on transient failures."""
    retries = kwargs.pop("retries", 2)
    timeout = kwargs.pop("timeout", 15)
    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as c:
        for attempt in range(retries + 1):
            try:
                if method == "GET":
                    return await c.get(url, **kwargs)
                elif method == "POST":
                    return await c.post(url, **kwargs)
                elif method == "PUT":
                    return await c.put(url, **kwargs)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(1 * (attempt + 1))
    raise last_err or RuntimeError("HTTP request failed after retries")

# ── TTL Cache ───────────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}
_MAX_CACHE = 500
_CACHE_TTL = {"gh": 120, "c7": 300, "wiki": 600, "pkg": 60, "readme": 600, "emb": 600}


def _cached(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.monotonic() - entry[0] < _CACHE_TTL.get(key.split(":")[0], 300):
        return entry[1]
    return None


def _set_cache(key: str, val: Any):
    _cache[key] = (time.monotonic(), val)
    if len(_cache) > _MAX_CACHE:
        now = time.monotonic()
        # Evict oldest 25% when over capacity
        sorted_keys = sorted(_cache, key=lambda k: _cache[k][0])
        evict_count = max(1, len(sorted_keys) // 4)
        for k in sorted_keys[:evict_count]:
            _cache.pop(k, None)
