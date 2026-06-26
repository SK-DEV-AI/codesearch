from __future__ import annotations

import httpx

from config import NV_EMBED_MODEL, SOFA_BASE, SOFA_KEY, _cached, _set_cache, get_http_client


async def search_sofa(query: str, count: int = 5, content_type: str = "question",
                      page: int = 1, post_id: str = "",
                      steering: str = "") -> dict:
    if post_id:
        return await _sofa_get_post(post_id)
    cache_key = f"sofa:{query}:{count}:{content_type}:{page}:{steering}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    session_id = None
    try:
        if not SOFA_KEY:
            return {"success": False, "error": "SOFA_KEY not configured"}
        c = get_http_client()
        sess_r = await c.post(f"{SOFA_BASE}/sessions", headers={
            "Authorization": f"Bearer {SOFA_KEY}",
            "X-Sofa-Client-Name": "mcp-codesearch",
            "X-Sofa-Model-Name": NV_EMBED_MODEL,
        })
        if sess_r.status_code != 201:
            return {"success": False, "error": f"SOFA session: HTTP {sess_r.status_code}"}
        session_id = sess_r.json()["session_id"]
        params: dict[str, str | int] = {
            "search": query, "per_page": min(count, 10),
            "content_type": content_type, "page": page,
        }
        if steering:
            params["steering"] = steering
        c = get_http_client()
        r = await c.get(f"{SOFA_BASE}/posts",
                        params=params,
                        headers={"Authorization": f"Bearer {SOFA_KEY}", "X-Sofa-Session": session_id})
        if r.status_code != 200:
            return {"success": False, "error": f"SOFA search: HTTP {r.status_code}"}
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
        }
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}
    finally:
        if session_id:
            try:
                c = get_http_client()
                await c.delete(f"{SOFA_BASE}/sessions/{session_id}",
                               headers={"Authorization": f"Bearer {SOFA_KEY}"})
            except (httpx.HTTPError, ValueError):
                pass


async def _sofa_get_post(post_id: str) -> dict:
    session_id = None
    try:
        c = get_http_client()
        sess_r = await c.post(f"{SOFA_BASE}/sessions", headers={
            "Authorization": f"Bearer {SOFA_KEY}",
            "X-Sofa-Client-Name": "mcp-codesearch",
        })
        if sess_r.status_code != 201:
            return {"success": False, "error": f"SOFA session: HTTP {sess_r.status_code}"}
        session_id = sess_r.json()["session_id"]
        c = get_http_client()
        r = await c.get(f"{SOFA_BASE}/posts/{post_id}",
                        headers={"Authorization": f"Bearer {SOFA_KEY}", "X-Sofa-Session": session_id})
        if r.status_code != 200:
            return {"success": False, "error": f"SOFA post: HTTP {r.status_code}"}
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
        }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}
    finally:
        if session_id:
            try:
                c = get_http_client()
                await c.delete(f"{SOFA_BASE}/sessions/{session_id}",
                               headers={"Authorization": f"Bearer {SOFA_KEY}"})
            except (httpx.HTTPError, ValueError):
                pass
