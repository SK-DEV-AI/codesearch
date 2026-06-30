from __future__ import annotations

from typing import Any

import httpx

from config import _cached, _set_cache, get_http_client

RTD_API = "https://readthedocs.org/api/v3"


async def search_readthedocs(project: str, query: str, version: str = "",
                             page: int = 1, page_size: int = 10) -> dict:
    cache_key = f"rtd:{project}:{query}:{version}:{page}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        q = query
        if project and version:
            q = f"project:{project}/{version} {query}"
        elif project:
            q = f"project:{project} {query}"
        params: dict[str, Any] = {"q": q, "page": page, "page_size": min(page_size, 50)}
        c = get_http_client()
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
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": data.get("count", len(results)),
                "page": page, "page_size": page_size}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_project_info(project: str) -> dict:
    try:
        c = get_http_client()
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
        c = get_http_client()
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
        c = get_http_client()
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
        c = get_http_client()
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


async def readthedocs_redirects(slug: str) -> dict:
    """List redirects for a ReadTheDocs project."""
    try:
        c = get_http_client()
        r = await c.get(f"{RTD_API}/projects/{slug}/redirects/", headers=await _rtd_headers())
        if r.status_code != 200: return {"success": False, "error": f"RTD redirects: {r.status_code}"}
        data = r.json()
        results = [{"pk": d.get("pk", 0), "from_url": d.get("from_url", ""), "to_url": d.get("to_url", ""),
                     "redirect_type": d.get("redirect_type", ""), "status": d.get("http_status", 302)}
                    for d in (data.get("results", []) or [])]
        return {"success": True, "redirects": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_notifications() -> dict:
    """List notifications for the authenticated user."""
    try:
        c = get_http_client()
        r = await c.get(f"{RTD_API}/notifications/", headers=await _rtd_headers())
        if r.status_code != 200: return {"success": False, "error": f"RTD notifications: {r.status_code}"}
        data = r.json()
        results = [{"id": d.get("id", 0), "message": d.get("message", ""),
                     "resource_uri": d.get("resource_uri", "")}
                    for d in (data.get("results", []) or [])]
        return {"success": True, "notifications": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_remote_repos() -> dict:
    """List remote repositories connected to ReadTheDocs."""
    try:
        c = get_http_client()
        r = await c.get(f"{RTD_API}/remote/repositories/", headers=await _rtd_headers())
        if r.status_code != 200: return {"success": False, "error": f"RTD remote repos: {r.status_code}"}
        data = r.json()
        results = [{"id": d.get("id", 0), "name": d.get("name", ""),
                     "vcs_provider": d.get("vcs_provider", ""), "url": d.get("clone_url", ""),
                     "is_locked": d.get("is_locked", False)}
                    for d in (data.get("results", []) or [])]
        return {"success": True, "remotes": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_remote_orgs() -> dict:
    """List remote organizations connected to ReadTheDocs."""
    try:
        c = get_http_client()
        r = await c.get(f"{RTD_API}/remote/organizations/", headers=await _rtd_headers())
        if r.status_code != 200: return {"success": False, "error": f"RTD remote orgs: {r.status_code}"}
        data = r.json()
        results = [{"id": d.get("id", 0), "name": d.get("name", ""), "slug": d.get("slug", "")}
                    for d in (data.get("results", []) or [])]
        return {"success": True, "organizations": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def readthedocs_builds(project: str, limit: int = 10) -> dict:
    try:
        c = get_http_client()
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
