from __future__ import annotations

import asyncio
import httpx

from config import NV_EMBED_MODEL, SOFA_BASE, SOFA_KEY, _cached, _set_cache, get_http_client, api_error
from httpretry import retry_request

_session_id: str | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> tuple[str | None, int]:
    global _session_id
    if not SOFA_KEY:
        return None, 0
    if _session_id:
        return _session_id, 0
    async with _session_lock:
        if _session_id:
            return _session_id, 0
        r, retries = await retry_request(
            "POST", f"{SOFA_BASE}/sessions",
            headers={
                "Authorization": f"Bearer {SOFA_KEY}",
                "X-Sofa-Client-Name": "mcp-codesearch",
                "X-Sofa-Model-Name": NV_EMBED_MODEL,
            },
        )
        if r.status_code != 201:
            _session_id = None
            return None, retries
        _session_id = r.json()["session_id"]
        return _session_id, retries


async def search_sofa(query: str, count: int = 5, content_type: str = "question",
                      page: int = 1, post_id: str = "",
                      steering: str = "") -> dict:
    if post_id:
        return await _sofa_get_post(post_id)
    cache_key = f"sofa:{query}:{count}:{content_type}:{page}:{steering}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        if not SOFA_KEY:
            return {"success": False, "error": "SOFA_KEY not configured", "retries": 0}
        session_id, retries = await _get_session()
        if not session_id:
            return {"success": False, "error": "SOFA: no session", "retries": retries}
        params: dict[str, str | int] = {
            "search": query, "per_page": min(count, 10),
            "content_type": content_type, "page": page,
        }
        if steering:
            params["steering"] = steering
        r, r_retries = await retry_request(
            "GET", f"{SOFA_BASE}/posts",
            params=params,
            headers={"Authorization": f"Bearer {SOFA_KEY}", "X-Sofa-Session": session_id},
        )
        retries += r_retries
        if r.status_code == 401:
            _session_id = None
            session_id, s_retries = await _get_session()
            retries += s_retries
            if not session_id:
                return {"success": False, "error": "SOFA: session expired", "retries": retries}
            r, r_retries = await retry_request(
                "GET", f"{SOFA_BASE}/posts",
                params=params,
                headers={"Authorization": f"Bearer {SOFA_KEY}", "X-Sofa-Session": session_id},
            )
            retries += r_retries
        if r.status_code != 200:
            return {"success": False, "error": api_error("SOFA search:", r), "retries": retries}
        data = r.json()
        items = data.get("items", [])
        results = []
        for item in items:
            results.append({
                "id": item.get("id"),
                "title": item.get("title", ""),
                "body": (item.get("body_markdown") or item.get("body", ""))[:1000],
                "tags": item.get("tags", []),
                "score": item.get("score"),
                "answer_count": item.get("answer_count"),
                "url": item.get("public_url"),
                "author": item.get("owner", {}).get("display_name") if isinstance(item.get("owner"), dict) else None,
                "content_type": item.get("content_type", content_type),
            })
        result = {
            "success": True,
            "results": results,
            "steering": data.get("steering", ""),
            "total": data.get("total", len(results)),
            "retries": retries,
        }
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e), "retries": 0}


async def _sofa_get_post(post_id: str) -> dict:
    session_id = None
    retries = 0
    try:
        if not SOFA_KEY:
            return {"success": False, "error": "SOFA_KEY not configured", "retries": 0}
        r, rr = await retry_request(
            "POST", f"{SOFA_BASE}/sessions",
            headers={
                "Authorization": f"Bearer {SOFA_KEY}",
                "X-Sofa-Client-Name": "mcp-codesearch",
            },
        )
        retries += rr
        if r.status_code != 201:
            return {"success": False, "error": api_error("SOFA session:", r), "retries": retries}
        session_id = r.json()["session_id"]
        r, rr = await retry_request(
            "GET", f"{SOFA_BASE}/posts/{post_id}",
            headers={"Authorization": f"Bearer {SOFA_KEY}", "X-Sofa-Session": session_id},
        )
        retries += rr
        if r.status_code != 200:
            return {"success": False, "error": api_error("SOFA post:", r), "retries": retries}
        item = r.json()
        return {
            "success": True,
            "id": item.get("id"),
            "title": item.get("title", ""),
            "body": (item.get("body_markdown") or item.get("body", ""))[:3000],
            "tags": item.get("tags", []),
            "score": item.get("score"),
            "content_type": item.get("content_type", ""),
            "url": item.get("public_url"),
            "retries": retries,
        }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e), "retries": retries}
    finally:
        if session_id:
            try:
                c = get_http_client()
                await c.delete(f"{SOFA_BASE}/sessions/{session_id}",
                               headers={"Authorization": f"Bearer {SOFA_KEY}"})
            except (httpx.HTTPError, ValueError):
                pass
