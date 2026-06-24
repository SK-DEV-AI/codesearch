from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

import httpx

from config import SO_API, _cached, _set_cache


async def _fetch_accepted_answer(c: httpx.AsyncClient, accepted_id: int, site: str) -> str:
    """Fetch a single accepted answer body."""
    try:
        ar = await c.get(f"{SO_API}/answers/{accepted_id}", params={
            "order": "desc", "sort": "votes", "site": site, "pagesize": 1, "filter": "withbody"
        }, headers={"User-Agent": "mcp-codesearch/1.0"})
        if ar.status_code == 200:
            adata = ar.json()
            if adata.get("items"):
                return adata["items"][0].get("body", "")[:2000]
    except Exception:
        pass
    return ""


async def search_so(query: str, count: int = 5, tags: str = "",
                    accepted: bool | None = None, fromdate: str = "",
                    todate: str = "", closed: bool | None = None,
                    sort: str = "relevance", views: int = 0,
                    answers: int = 0, type: str = "search",
                    site: str = "stackoverflow", question_id: int = 0,
                    page: int = 1) -> dict:
    if type == "excerpts":
        return await _so_search_excerpts(query, count, tags, site)
    if type == "faq":
        return await _so_tags_faq(tags, count, site)
    if type == "answers":
        return await _so_question_answers(question_id, count, site=site)
    if type == "similar":
        return await so_similar(query, tags, count, site=site)
    if type == "tags_info":
        return await so_tags_info(tags, site)
    if type == "tags_wikis":
        return await so_tags_wikis(tags, count, site)
    cache_key = f"so:{query}:{tags}:{count}:{accepted}:{fromdate}:{todate}:{closed}:{sort}:{views}:{answers}:{type}:{site}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        params: dict[str, Any] = {
            "order": "desc", "sort": sort, "q": query,
            "site": site, "pagesize": min(count, 50),
            "filter": "withbody",
        }
        if tags:
            params["tagged"] = tags
        if accepted is not None:
            params["accepted"] = "true" if accepted else "false"
        if fromdate:
            params["fromdate"] = fromdate
        if todate:
            params["todate"] = todate
        if closed is not None:
            params["closed"] = "true" if closed else "false"
        if views > 0:
            params["views"] = views
        if answers > 0:
            params["answers"] = answers
        if page > 1:
            params["page"] = page
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/search/advanced", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
            if r.status_code != 200:
                return {"success": False, "error": f"StackExchange API: {r.status_code}"}
            data = r.json()
            items = data.get("items", [])[:count]

            # Batch fetch accepted answers concurrently
            accepted_ids = [item.get("accepted_answer_id") for item in items]
            accepted_answers = {}
            fetch_tasks = []
            fetch_ids = []
            for aid in accepted_ids:
                if aid:
                    fetch_tasks.append(_fetch_accepted_answer(c, aid, site))
                    fetch_ids.append(aid)
            if fetch_tasks:
                results_fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                for aid, res in zip(fetch_ids, results_fetched):
                    if isinstance(res, str):
                        accepted_answers[aid] = res

            results = []
            for item in items:
                accepted_id = item.get("accepted_answer_id")
                top_answer = accepted_answers.get(accepted_id, "") if accepted_id else ""
                results.append({
                    "title": item.get("title", ""),
                    "score": item.get("score", 0),
                    "answer_count": item.get("answer_count", 0),
                    "is_answered": item.get("is_answered", False),
                    "accepted": bool(accepted_id),
                    "tags": item.get("tags", []),
                    "url": item.get("link", ""),
                    "body": (item.get("body", "") or "")[:1000],
                    "top_answer": top_answer,
                    "creation_date": item.get("creation_date", 0),
                    "last_activity_date": item.get("last_activity_date", 0),
                    "view_count": item.get("view_count", 0),
                })
        _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def so_similar(title: str, tags: str = "", count: int = 5, site: str = "stackoverflow") -> dict:
    try:
        params: dict[str, Any] = {
            "order": "desc", "sort": "relevance", "title": title,
            "site": site, "pagesize": min(count, 10),
        }
        if tags:
            params["tagged"] = tags
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/similar", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /similar: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "title": item.get("title", ""),
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "url": item.get("link", ""),
                "tags": item.get("tags", []),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def so_tags_info(tags: str, site: str = "stackoverflow") -> dict:
    try:
        params: dict[str, Any] = {"site": site}
        tag = urllib.parse.quote(tags)
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/tags/{tag}/info", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /tags/info: {r.status_code}"}
        data = r.json()
        items = data.get("items", [])
        if not items:
            return {"success": True, "results": []}
        info = items[0]
        return {"success": True, "results": [{
            "name": info.get("name", tags),
            "has_synonyms": info.get("has_synonyms", False),
            "is_moderator_only": info.get("is_moderator_only", False),
            "is_required": info.get("is_required", False),
            "count": info.get("count", 0),
            "excerpt": info.get("excerpt", ""),
            "link": info.get("link", ""),
        }]}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def so_tags_wikis(tags: str, count: int = 5, site: str = "stackoverflow") -> dict:
    try:
        params: dict[str, Any] = {"site": site, "pagesize": min(count, 50), "filter": "withbody"}
        tag = urllib.parse.quote(tags)
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/tags/{tag}/wikis", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /tags/wikis: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "tag": item.get("tag", tags),
                "link": item.get("link", ""),
                "excerpt": (item.get("excerpt") or "")[:1000],
                "body": (item.get("body") or "")[:3000],
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _so_search_excerpts(query: str, count: int = 5, tags: str = "",
                              site: str = "stackoverflow") -> dict:
    try:
        params: dict[str, Any] = {
            "order": "desc", "sort": "relevance", "q": query,
            "site": site, "pagesize": min(count, 50),
        }
        if tags:
            params["tagged"] = tags
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/search/excerpts", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /search/excerpts: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "title": item.get("title", ""),
                "excerpt": item.get("excerpt", ""),
                "url": item.get("link", ""),
                "score": item.get("score", 0),
                "tags": item.get("tags", []),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _so_tags_faq(tags: str, count: int = 5, site: str = "stackoverflow") -> dict:
    try:
        params: dict[str, Any] = {
            "site": site, "pagesize": min(count, 50),
            "filter": "withbody",
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/tags/{urllib.parse.quote(tags)}/faq",
                            params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /tags/faq: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "title": item.get("title", ""),
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "url": item.get("link", ""),
                "tags": item.get("tags", []),
                "body": (item.get("body", "") or "")[:1000],
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _so_question_answers(question_id: int, count: int = 5, site: str = "stackoverflow") -> dict:
    try:
        params: dict[str, Any] = {
            "order": "desc", "sort": "votes",
            "site": site, "pagesize": min(count, 50),
            "filter": "withbody",
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{SO_API}/questions/{question_id}/answers",
                            params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /questions/answers: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "answer_id": item.get("answer_id", 0),
                "score": item.get("score", 0),
                "is_accepted": item.get("is_accepted", False),
                "body": (item.get("body", "") or "")[:2000],
                "url": item.get("link", ""),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
