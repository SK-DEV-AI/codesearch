from __future__ import annotations

import httpx

from config import TAVILY_SEARCH, _next_tv_key, get_http_client


async def tavily_search(query: str, count: int = 10) -> dict:
    """Search Tavily and return results. Uses internal key rotation."""
    key = await _next_tv_key()
    if not key:
        return {"success": False, "results": []}
    try:
        c = get_http_client()
        body = {"query": query, "search_depth": "advanced", "max_results": count,
                "include_answer": True, "topic": "general", "include_images": True,
                "auto_parameters": True}
        r = await c.post(TAVILY_SEARCH, json=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "results": []}
        data = r.json()
        results = []
        for item in (data.get("results", []) or []):
            results.append({"title": item.get("title","")[:120], "url": item.get("url",""),
                            "snippet": (item.get("content","") or "")[:300]})
        return {"success": True, "results": results}
    except Exception:
        return {"success": False, "results": []}
