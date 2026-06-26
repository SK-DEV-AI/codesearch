from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config import _cached, _set_cache, get_http_client

HN_ALGOLIA = "https://hn.algolia.com/api/v1"
HN_FIREBASE = "https://hacker-news.firebaseio.com/v0"


async def search_hn(query: str, count: int = 5, sort_by_date: bool = False,
                    tags: str = "story", min_points: int = 0, min_comments: int = 0,
                    before: int = 0, after: int = 0,
                    page: int = 0, hits_per_page: int = 20) -> dict:
    cache_key = f"hn:{query}:{count}:{sort_by_date}:{tags}:{min_points}:{min_comments}:{before}:{after}:{page}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        endpoint = f"{HN_ALGOLIA}/search_by_date" if sort_by_date else f"{HN_ALGOLIA}/search"
        filters = []
        if min_points > 0 and tags != "comment":
            filters.append(f"points>{min_points}")
        if min_comments > 0 and tags != "comment":
            filters.append(f"num_comments>{min_comments}")
        if before > 0:
            filters.append(f"created_at_i<{before}")
        if after > 0:
            filters.append(f"created_at_i>{after}")
        params: dict[str, Any] = {
            "query": query,
            "hitsPerPage": min(hits_per_page, 500),
            "tags": tags,
            "page": page,
        }
        if filters:
            params["numericFilters"] = ",".join(filters)
        c = get_http_client()
        r = await c.get(endpoint, params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Algolia: {r.status_code}"}
        data = r.json()
        hits = data.get("hits", [])[:count]
        results = []
        for hit in hits:
            results.append({
                "title": hit.get("title", ""),
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                "points": hit.get("points", 0),
                "author": hit.get("author", ""),
                "num_comments": hit.get("num_comments", 0),
                "created_at": hit.get("created_at", ""),
                "object_id": hit.get("objectID", ""),
            })
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": data.get("nbHits", 0),
                "page": data.get("page", 0), "nb_pages": data.get("nbPages", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def hn_get_item(item_id: int) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{HN_ALGOLIA}/items/{item_id}",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Algolia: {r.status_code}"}
        data = r.json()
        return {
            "success": True,
            "title": data.get("title", ""),
            "author": data.get("author", ""),
            "url": data.get("url") or f"https://news.ycombinator.com/item?id={item_id}",
            "points": data.get("points", 0),
            "text": (data.get("text") or "")[:3000],
            "num_comments": data.get("descendants", len(data.get("children", []))),
            "children": [{"author": c.get("author", ""), "text": (c.get("text") or "")[:500]} for c in (data.get("children") or [])[:10]],
        }
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def hn_firebase_stories(story_type: str = "top", count: int = 10) -> dict:
    valid_types = {"top": "topstories", "new": "newstories", "best": "beststories",
                   "ask": "askstories", "show": "showstories"}
    fb_key = valid_types.get(story_type, "topstories")
    try:
        c = get_http_client()
        r = await c.get(f"{HN_FIREBASE}/{fb_key}.json", headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Firebase: {r.status_code}"}
        ids = (r.json() or [])[:count]

        async def _fetch_item(cid: int, client: httpx.AsyncClient) -> dict | None:
            try:
                ir = await client.get(f"{HN_FIREBASE}/item/{cid}.json", headers={"User-Agent": "mcp-codesearch/1.0"})
                if ir.status_code == 200:
                    item = ir.json()
                    if item:
                        return {
                            "id": item.get("id"),
                            "title": item.get("title", ""),
                            "url": item.get("url") or f"https://news.ycombinator.com/item?id={item.get('id')}",
                            "score": item.get("score", 0),
                            "by": item.get("by", ""),
                            "descendants": item.get("descendants", 0),
                            "time": item.get("time", 0),
                        }
            except Exception:
                pass
            return None

        sem = asyncio.Semaphore(5)
        c = get_http_client()
        async def _limited_fetch(cid: int):
            async with sem:
                return await _fetch_item(cid, c)
        fetched = await asyncio.gather(*[_limited_fetch(cid) for cid in ids], return_exceptions=True)
        results = [r for r in fetched if isinstance(r, dict)]
        return {"success": True, "type": story_type, "results": results, "total": len(results)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def hn_get_user(username: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{HN_FIREBASE}/user/{username}.json", headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"Firebase user: {r.status_code}"}
        data = r.json()
        if not data:
            return {"success": False, "error": f"User not found: {username}"}
        return {
            "success": True,
            "user": {
                "id": data.get("id", ""),
                "karma": data.get("karma", 0),
                "created_at": data.get("created", 0),
                "about": (data.get("about") or "")[:2000],
            },
        }
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
