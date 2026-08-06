from __future__ import annotations

import logging

import httpx

from config import TAVILY_SEARCH, _next_tv_key, get_http_client, api_error

logger = logging.getLogger("tavily_search")


async def tavily_search(query: str, count: int = 10,
                        include_domains: list[str] | None = None,
                        exclude_domains: list[str] | None = None) -> dict:
    """Search Tavily and return results. Uses internal key rotation."""
    key = await _next_tv_key()
    if not key:
        return {"success": False, "results": []}
    try:
        c = get_http_client()
        body = {"query": query, "search_depth": "advanced", "max_results": count,
                "include_answer": True, "topic": "general", "include_images": False,
                "auto_parameters": True}
        if include_domains:
            body["include_domains"] = include_domains
        if exclude_domains:
            body["exclude_domains"] = exclude_domains
        r = await c.post(TAVILY_SEARCH, json=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            logger.warning("tavily non-200: %s", api_error("tavily", r))
            return {"success": False, "error": api_error("tavily", r), "results": []}
        data = r.json()
        results = []
        for item in (data.get("results", []) or []):
            results.append({"title": item.get("title","")[:120], "url": item.get("url",""),
                            "snippet": (item.get("content","") or "")[:300]})
        return {"success": True, "results": results}
    except Exception as e:
        logger.warning("tavily search failed: %s", e)
        return {"success": False, "error": str(e), "results": []}
