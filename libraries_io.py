from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from config import LI_API, LI_KEY, _cached, _set_cache, _next_li_key, get_http_client


async def search_libraries_io(name: str, platform: str = "") -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    cache_key = f"li:{name}:{platform}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        platforms = [platform] if platform else ["npm", "pypi", "cargo"]
        results = {}
        c = get_http_client()
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
        await _set_cache(cache_key, result)
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
    cached = await _cached(cache_key)
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
        c = get_http_client()
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
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": len(results)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_versions(platform: str, name: str) -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    try:
        c = get_http_client()
        r = await c.get(f"{LI_API}/{platform}/{urllib.parse.quote(name)}/versions",
                        params={"api_key": li_key},
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Libraries.io versions: {r.status_code}"}
        data = r.json()
        results = []
        for v in (data if isinstance(data, list) else [])[:50]:
            results.append({
                "number": v.get("number", ""),
                "published_at": v.get("published_at", ""),
                "platform": v.get("platform", ""),
            })
        return {"success": True, "results": results, "total": len(results)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_dependencies(platform: str, name: str, version: str = "") -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    try:
        url = f"{LI_API}/{platform}/{urllib.parse.quote(name)}"
        if version:
            url += f"/{urllib.parse.quote(version)}"
        url += "/dependencies"
        c = get_http_client()
        r = await c.get(url, params={"api_key": li_key},
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Libraries.io dependencies: {r.status_code}"}
        data = r.json()
        deps = []
        for dep in (data.get("dependencies", []) or [])[:50]:
            deps.append({
                "name": dep.get("name", ""),
                "platform": dep.get("platform", ""),
                "requirements": dep.get("requirements", ""),
                "kind": dep.get("kind", ""),
                "latest_version": dep.get("latest_version_number", ""),
            })
        return {"success": True, "results": deps, "total": len(deps)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_dependents(platform: str, name: str) -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    try:
        c = get_http_client()
        r = await c.get(f"{LI_API}/{platform}/{urllib.parse.quote(name)}/dependents",
                        params={"api_key": li_key},
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Libraries.io dependents: {r.status_code}"}
        data = r.json()
        results = []
        for dep in (data if isinstance(data, list) else [])[:50]:
            results.append({
                "name": dep.get("name", ""),
                "platform": dep.get("platform", ""),
                "latest_version": dep.get("latest_version_number", ""),
                "dependent_repos_count": dep.get("dependent_repos_count", 0),
            })
        return {"success": True, "results": results, "total": len(results)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_github_repo(owner: str, repo: str) -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    try:
        c = get_http_client()
        r = await c.get(f"{LI_API}/github/{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}",
                        params={"api_key": li_key},
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Libraries.io github: {r.status_code}"}
        d = r.json()
        return {"success": True, "result": {
            "name": d.get("name", ""),
            "full_name": d.get("full_name", ""),
            "description": (d.get("description") or "")[:300],
            "stars": d.get("stars", 0),
            "forks": d.get("forks", 0),
            "language": d.get("language", ""),
            "license": d.get("license", ""),
            "homepage": d.get("homepage", ""),
            "repository": d.get("repository_url", ""),
            "open_issues": d.get("open_issues", 0),
            "watchers": d.get("watchers", 0),
            "source_rank": d.get("source_rank", 0),
        }}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_github_dependencies(owner: str, repo: str) -> dict:
    if not LI_KEY:
        return {"success": False, "error": "LI_KEY not configured"}
    li_key = await _next_li_key()
    try:
        c = get_http_client()
        r = await c.get(f"{LI_API}/github/{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}/dependencies",
                        params={"api_key": li_key},
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Libraries.io github deps: {r.status_code}"}
        data = r.json()
        deps = []
        for dep in (data.get("dependencies", []) or [])[:50]:
            deps.append({
                "name": dep.get("name", ""),
                "platform": dep.get("platform", ""),
                "requirements": dep.get("requirements", ""),
                "kind": dep.get("kind", ""),
                "latest_version": dep.get("latest_version_number", ""),
            })
        return {"success": True, "results": deps, "total": len(deps)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
