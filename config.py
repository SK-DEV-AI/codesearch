from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

CONTEXT7_API_KEY = os.environ.get("CONTEXT7_API_KEY", "")
CONTEXT7_SEARCH = "https://context7.com/api/v2/libs/search"
CONTEXT7_CONTEXT = "https://context7.com/api/v2/context"
DEEPWIKI_MCP = "https://mcp.deepwiki.com/mcp"
_raw_gh = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))
_GH_KEYS = [k.strip() for k in _raw_gh.split(",") if k.strip()] if _raw_gh else []
_gh_idx = 0
_GH_LOCK = asyncio.Lock()


async def _next_gh_key() -> str | None:
    global _gh_idx
    if not _GH_KEYS:
        return None
    async with _GH_LOCK:
        k = _GH_KEYS[_gh_idx % len(_GH_KEYS)]
        _gh_idx = (_gh_idx + 1) % len(_GH_KEYS)
        return k


GH_TOKEN = _GH_KEYS[0] if _GH_KEYS else ""
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
_raw_li = os.environ.get("LI_KEY", "")
_LI_KEYS = [k.strip() for k in _raw_li.split(",") if k.strip()] if _raw_li else []
_li_idx = 0
_LI_LOCK = asyncio.Lock()


async def _next_li_key() -> str | None:
    global _li_idx
    if not _LI_KEYS:
        return None
    async with _LI_LOCK:
        k = _LI_KEYS[_li_idx % len(_LI_KEYS)]
        _li_idx = (_li_idx + 1) % len(_LI_KEYS)
        return k


LI_KEY = _LI_KEYS[0] if _LI_KEYS else ""
LI_API = "https://libraries.io/api"
_raw_oss = os.environ.get("OSS_TOKEN", "")
_OSS_KEYS = [k.strip() for k in _raw_oss.split(",") if k.strip()] if _raw_oss else []
_oss_idx = 0
_OSS_LOCK = asyncio.Lock()


async def _next_oss_key() -> str | None:
    global _oss_idx
    if not _OSS_KEYS:
        return None
    async with _OSS_LOCK:
        k = _OSS_KEYS[_oss_idx % len(_OSS_KEYS)]
        _oss_idx = (_oss_idx + 1) % len(_OSS_KEYS)
        return k


OSS_TOKEN = _OSS_KEYS[0] if _OSS_KEYS else ""
OSS_API = "https://api.guide.sonatype.com/api/v3/component-report"
_raw_fc = os.environ.get("FIRECRAWL_KEY", os.environ.get("FIRECRAWL_KEYS", ""))
_FC_KEYS = [k.strip() for k in _raw_fc.split(",") if k.strip()] if _raw_fc else []
_fc_idx = 0
_FC_LOCK = asyncio.Lock()

async def _next_fc_key() -> str | None:
    global _fc_idx
    if not _FC_KEYS:
        return None
    async with _FC_LOCK:
        k = _FC_KEYS[_fc_idx % len(_FC_KEYS)]
        _fc_idx = (_fc_idx + 1) % len(_FC_KEYS)
        return k

_raw_tv = os.environ.get("TAVILY_KEY", os.environ.get("TAVILY_KEYS", ""))
_TV_KEYS = [k.strip() for k in _raw_tv.split(",") if k.strip()] if _raw_tv else []
_tv_idx = 0
_TV_LOCK = asyncio.Lock()

async def _next_tv_key() -> str | None:
    global _tv_idx
    if not _TV_KEYS:
        return None
    async with _TV_LOCK:
        k = _TV_KEYS[_tv_idx % len(_TV_KEYS)]
        _tv_idx = (_tv_idx + 1) % len(_TV_KEYS)
        return k
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
        stale = [k for k, (t, _) in list(_cache.items()) if now - t > 300]
        for k in stale:
            _cache.pop(k, None)
