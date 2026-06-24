from __future__ import annotations

import httpx

from config import DEVDOCS_API


async def devdocs_list_docs() -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{DEVDOCS_API}/docs.json",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"DevDocs: {r.status_code}"}
        docs = r.json()
        return {"success": True, "total": len(docs), "docs": [d.get("name", d.get("slug", "")) for d in docs]}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def devdocs_fetch(slug: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{DEVDOCS_API}/{slug}/index.json",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"DevDocs: HTTP {r.status_code}"}
        data = r.json()
        entries = []
        for entry in (data.get("entries", data.get("types", [data])) if isinstance(data, dict) else data):
            if isinstance(entry, dict):
                entries.append({
                    "name": entry.get("name", ""),
                    "path": entry.get("path", ""),
                    "type": entry.get("type", ""),
                })
        return {"success": True, "slug": slug, "entries": entries[:30], "total": len(entries)}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def devdocs_fetch_content(slug: str, path: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{DEVDOCS_API}/{slug}/db.json",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"DevDocs: HTTP {r.status_code}"}
        db = r.json()
        content = db.get(path, "")
        if not content:
            keys = [k for k in db.keys() if path.lower() in k.lower()]
            if keys:
                content = db[keys[0]]
        return {"success": True, "slug": slug, "path": path, "content": content[:8000] if content else ""}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def devdocs_search(slug: str, query: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{DEVDOCS_API}/{slug}/index.json",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"DevDocs: HTTP {r.status_code}"}
        data = r.json()
        ql = query.lower()
        matched = []
        entries = data.get("entries", []) if isinstance(data, dict) else data
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and (ql in entry.get("name", "").lower() or ql in entry.get("path", "").lower()):
                matched.append({
                    "name": entry.get("name", ""),
                    "path": entry.get("path", ""),
                    "type": entry.get("type", ""),
                    "url": f"https://devdocs.io/{slug}/{entry.get('path', '')}",
                })
        return {"success": True, "slug": slug, "results": matched[:15], "total": len(matched)}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def devdocs_meta(slug: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{DEVDOCS_API}/{slug}/meta.json",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"DevDocs meta: HTTP {r.status_code}"}
        return {"success": True, "slug": slug, "meta": r.json()}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}
