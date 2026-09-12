from __future__ import annotations

import httpx
import json
from typing import Any

from config import DEVDOCS_API, get_http_client, api_error, _cached, _set_cache
from security import SecurityError, safe_stream

# DevDocs layout (verified live against the SPA bundle config + endpoints):
# - docs.json manifest lives on the main host (devdocs.io).
# - per-doc index.json / db.json / meta.json live on the DATA host
#   (documents.devdocs.io, from the bundle's docs_origin), under VERSIONED
#   slugs ("python~3.14" — docs.json lists them; bare "python" 404s).
# - the data host hotlink-protects: requests without a browser
#   Origin/Referer hang until timeout. Public licensed content, same as
#   the web app fetches — sent here for the same reason.
_DEVDOCS_DATA = "https://documents.devdocs.io"
_DD_HEADERS = {"User-Agent": "mcp-codesearch/1.0",
               "Origin": "https://devdocs.io",
               "Referer": "https://devdocs.io/"}


async def _docs_manifest() -> list | None:
    """docs.json, cached 5 min (default TTL) — also the slug resolver base."""
    hit = await _cached("devdocs:docs.json")
    if hit is not None:
        return hit
    data, _ = await _capped_json(f"{DEVDOCS_API}/docs.json", 2_000_000)
    if not isinstance(data, list):
        return None
    await _set_cache("devdocs:docs.json", data)
    return data


async def _resolve_slug(slug: str) -> str | None:
    """Map user slug ("python", "py", "python~3.14") to versioned slug."""
    docs = await _docs_manifest()
    if not docs:
        return None
    slow = (slug or "").lower()
    for d in docs:
        if d.get("slug", "").lower() == slow:
            return d["slug"]
    for d in docs:
        s = d.get("slug", "")
        if s.lower().split("~")[0] == slow or (d.get("alias") or "").lower() == slow:
            return s
    return None


async def _capped_json(url: str, cap: int) -> tuple[Any, str]:
    """GET JSON with a byte cap. Returns (data, "") or (None, error)."""
    try:
        body = await safe_stream(get_http_client(), url,
                                 max_bytes=cap, headers=_DD_HEADERS)
    except httpx.HTTPStatusError as e:
        return None, api_error("DevDocs", e.response)
    except (httpx.HTTPError, SecurityError, ValueError) as e:
        return None, str(e)
    if len(body) >= cap:
        return None, f"response exceeds {cap // 1_000_000} MB cap: {url}"
    try:
        return json.loads(body), ""
    except ValueError as e:
        return None, str(e)


async def devdocs_list_docs() -> dict:
    docs = await _docs_manifest()
    if docs is None:
        return {"success": False, "error": "could not fetch DevDocs docs.json"}
    return {"success": True, "total": len(docs), "docs": [d.get("name", d.get("slug", "")) for d in docs]}


async def devdocs_fetch(slug: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{DEVDOCS_API}/{slug}/index.json",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("DevDocs:", r)}
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
        resolved = await _resolve_slug(slug)
        if not resolved:
            return {"success": False, "error": f"unknown DevDocs docset '{slug}'"}
        # db.json is whole-doc JSON (python: 20 MB) — stream with a cap
        # instead of an unbounded r.json() that OOMs the server.
        db, err = await _capped_json(f"{_DEVDOCS_DATA}/{resolved}/db.json", 32_000_000)
        if db is None:
            return {"success": False, "error": err}
        if not isinstance(db, dict):
            return {"success": False, "error": f"unexpected db.json shape for '{resolved}'"}
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
        resolved = await _resolve_slug(slug)
        if not resolved:
            return {"success": False, "error": f"unknown DevDocs docset '{slug}'"}
        data, err = await _capped_json(f"{_DEVDOCS_DATA}/{resolved}/index.json", 8_000_000)
        if data is None:
            return {"success": False, "error": err}
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


async def devdocs_toc(doc: str, version: str = "") -> dict:
    """Table of contents for a DevDocs docset (entry names by type).
    The old /{doc}/{version}.json endpoint never existed — the TOC is
    derived from index.json, which is the same source the app uses."""
    try:
        slug = f"{doc}~{version}" if version else doc
        resolved = await _resolve_slug(slug)
        if not resolved:
            return {"success": False, "error": f"unknown DevDocs docset '{doc}'"}
        data, err = await _capped_json(f"{_DEVDOCS_DATA}/{resolved}/index.json", 8_000_000)
        if data is None:
            return {"success": False, "error": err}
        entries = data.get("entries", []) if isinstance(data, dict) else []
        toc: dict[str, list[str]] = {}
        for e in entries:
            if isinstance(e, dict):
                toc.setdefault(e.get("type", "Misc"), []).append(e.get("name", ""))
        return {"success": True, "doc": resolved, "types": len(toc),
                "entries": sum(len(v) for v in toc.values()), "toc": toc}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def devdocs_meta(slug: str) -> dict:
    resolved = await _resolve_slug(slug)
    if not resolved:
        return {"success": False, "error": f"unknown DevDocs docset '{slug}'"}
    data, err = await _capped_json(f"{_DEVDOCS_DATA}/{resolved}/meta.json", 200_000)
    if data is None:
        return {"success": False, "error": err}
    return {"success": True, "slug": resolved, "meta": data}
