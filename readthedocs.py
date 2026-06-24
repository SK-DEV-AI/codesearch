from __future__ import annotations

from typing import Any

import httpx

from config import _cached, _set_cache

RTD_API = "https://readthedocs.org/api/v3"


async def search_readthedocs(project: str, query: str, version: str = "",
                             page: int = 1, page_size: int = 10) -> dict:
    cache_key = f"rtd:{project}:{query}:{version}:{page}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        q = query
        if project and version:
            q = f"project:{project}/{version} {query}"
        elif project:
            q = f"project:{project} {query}"
        params: dict[str, Any] = {"q": q, "page": page, "page_size": min(page_size, 50)}
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(f"{RTD_API}/search/", params=params,
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"ReadTheDocs: {r.status_code}"}
        data = r.json()
        results = []
        for hit in data.get("results", [])[:page_size]:
            proj = hit.get("project", {})
            ver = hit.get("version", {})
            results.append({
                "title": hit.get("title", ""),
                "url": f"https://{proj.get('slug', project)}.readthedocs.io{hit.get('path', '')}",
                "project": proj.get("slug", project),
                "version": ver.get("slug", ""),
                "content": (hit.get("highlight", "") or "")[:1500],
                "blocks": hit.get("blocks", []),
            })
        _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": data.get("count", len(results)),
                "page": page, "page_size": page_size}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_project_info(project: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(f"{RTD_API}/projects/{project}/",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"ReadTheDocs: {r.status_code}"}
        d = r.json()
        return {
            "success": True,
            "name": d.get("name", ""),
            "slug": d.get("slug", project),
            "description": d.get("description", ""),
            "language": d.get("language", {}).get("code", ""),
            "programming_language": d.get("programming_language", {}).get("code", ""),
            "repository": d.get("repository", {}).get("url", ""),
            "homepage": d.get("homepage", ""),
            "default_version": d.get("default_version", ""),
            "default_branch": d.get("default_branch", ""),
            "created": d.get("created", ""),
            "modified": d.get("modified", ""),
        }
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_versions(project: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(f"{RTD_API}/projects/{project}/versions/",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"ReadTheDocs: {r.status_code}"}
        data = r.json()
        results = []
        for v in data.get("results", [])[:20]:
            results.append({
                "slug": v.get("slug", ""),
                "verbose_name": v.get("verbose_name", ""),
                "active": v.get("active", False),
                "built": v.get("built", False),
                "uploaded": v.get("uploaded", False),
                "hidden": v.get("hidden", False),
            })
        return {"success": True, "results": results, "total": data.get("count", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_translations(project: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(f"{RTD_API}/projects/{project}/translations/",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"ReadTheDocs translations: {r.status_code}"}
        data = r.json()
        results = []
        for t in (data.get("results", []) or [])[:20]:
            results.append({
                "slug": t.get("slug", ""),
                "language": t.get("language", {}).get("code", "") if isinstance(t.get("language"), dict) else t.get("language", ""),
                "url": t.get("url", ""),
                "default_version": t.get("default_version", ""),
            })
        return {"success": True, "results": results, "total": data.get("count", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_subprojects(project: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(f"{RTD_API}/projects/{project}/subprojects/",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"ReadTheDocs subprojects: {r.status_code}"}
        data = r.json()
        results = []
        for sp in (data.get("results", []) or [])[:20]:
            child = sp.get("child", {})
            results.append({
                "alias": sp.get("alias", ""),
                "child_slug": child.get("slug", "") if isinstance(child, dict) else str(child),
                "child_name": child.get("name", "") if isinstance(child, dict) else "",
            })
        return {"success": True, "results": results, "total": data.get("count", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_builds(project: str, limit: int = 10) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(f"{RTD_API}/projects/{project}/builds/",
                            params={"limit": min(limit, 50)},
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"ReadTheDocs builds: {r.status_code}"}
        data = r.json()
        results = []
        for b in (data.get("results", []) or [])[:limit]:
            results.append({
                "id": b.get("id", 0),
                "commit": b.get("commit", ""),
                "status": b.get("status", ""),
                "success": b.get("success"),
                "error": b.get("error", ""),
                "created": b.get("created", ""),
                "finished": b.get("finished", ""),
                "duration": b.get("duration"),
                "version": b.get("version", {}).get("slug", "") if isinstance(b.get("version"), dict) else "",
            })
        return {"success": True, "results": results, "total": data.get("count", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
