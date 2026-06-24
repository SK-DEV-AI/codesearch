from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from config import LI_API, LI_KEY, _cached, _set_cache, _next_li_key


async def search_libraries_io(name: str, platform: str = "") -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    cache_key = f"li:{name}:{platform}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        platforms = [platform] if platform else ["npm", "pypi", "cargo"]
        results = {}
        async with httpx.AsyncClient(timeout=15) as c:
            for p in platforms:
                r = await c.get(f"{LI_API}/{p}/{urllib.parse.quote(name)}", params={"api_key": li_key},
                                headers={"User-Agent": "mcp-codesearch/1.0"})
                if r.status_code == 200:
                    d = r.json()
                    results[p] = {
                        "name": d.get("name", name),
                        "version": d.get("latest_stable_release", d.get("latest_release_number", "")),
                        "license": d.get("licenses", ""),
                        "stars": d.get("stars", 0),
                        "forks": d.get("forks", 0),
                        "dependent_repos": d.get("dependent_repos_count", 0),
                        "dependents": d.get("dependents_count", 0),
                        "rank": d.get("rank", 0),
                        "homepage": d.get("homepage", ""),
                        "repository": d.get("repository_url", ""),
                        "description": d.get("description", "")[:200],
                        "language": d.get("language", ""),
                        "keywords": d.get("keywords", []),
                        "latest_release_date": d.get("latest_stable_release_published_at", ""),
                    }
        if not results:
            return {"success": False, "error": f"'{name}' not found on Libraries.io"}
        result: dict[str, Any] = {"success": True, "results": results}
        _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def libraries_io_search(query: str, platform: str = "", sort: str = "",
                              languages: str = "", licenses: str = "",
                              keywords: str = "", per_page: int = 10) -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    cache_key = f"li_search:{query}:{platform}:{sort}:{languages}:{licenses}:{keywords}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        params: dict[str, Any] = {"api_key": li_key, "q": query, "per_page": min(per_page, 100)}
        if platform:
            params["platforms"] = platform
        if sort:
            params["sort"] = sort
        if languages:
            params["languages"] = languages
        if licenses:
            params["licenses"] = licenses
        if keywords:
            params["keywords"] = keywords
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{LI_API}/search", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Libraries.io search: {r.status_code}"}
        data = r.json()
        results = []
        for item in (data if isinstance(data, list) else []):
            results.append({
                "name": item.get("name", ""),
                "platform": item.get("platform", ""),
                "description": (item.get("description") or "")[:200],
                "stars": item.get("stars", 0),
                "language": item.get("language", ""),
                "latest_version": item.get("latest_release_number", ""),
                "license": item.get("licenses", ""),
                "url": item.get("homepage", ""),
                "forks": item.get("forks", 0),
                "dependent_repos": item.get("dependent_repos_count", 0),
            })
        _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": len(results)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
