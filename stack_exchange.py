from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

import httpx

from config import SO_API, _cached, _set_cache, get_http_client


async def _fetch_accepted_answer(accepted_id: int, site: str) -> str:
    try:
        c = get_http_client()
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
                    page: int = 1, filter: str = "") -> dict:
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
    cache_key = f"so:{query}:{tags}:{count}:{accepted}:{fromdate}:{todate}:{closed}:{sort}:{views}:{answers}:{type}:{site}:{filter}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        params: dict[str, Any] = {
            "order": "desc", "sort": sort, "q": query,
            "site": site, "pagesize": min(count, 50),
            "filter": filter if filter else "withbody",
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
        c = get_http_client()
        r = await c.get(f"{SO_API}/search/advanced", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange API: {r.status_code}"}
        data = r.json()
        items = data.get("items", [])[:count]

        accepted_ids = [item.get("accepted_answer_id") for item in items]
        accepted_answers = {}
        fetch_tasks = []
        fetch_ids = []
        for aid in accepted_ids:
            if aid:
                fetch_tasks.append(_fetch_accepted_answer(aid, site))
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
        await _set_cache(cache_key, results)
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
        c = get_http_client()
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
        c = get_http_client()
        r = await c.get(f"{SO_API}/tags/{tag}/info", params=params,
                        headers={"User-Agent": "mcp-codesearch/1.0"})
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
        c = get_http_client()
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
        c = get_http_client()
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
        c = get_http_client()
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
        c = get_http_client()
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


async def get_questions_by_ids(ids: list[int], site: str = "stackoverflow") -> dict:
    try:
        ids_str = ";".join(str(i) for i in ids[:30])
        params: dict[str, Any] = {"site": site, "filter": "withbody"}
        c = get_http_client()
        r = await c.get(f"{SO_API}/questions/{ids_str}", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /questions: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:len(ids)]:
            results.append({
                "question_id": item.get("question_id", 0),
                "title": item.get("title", ""),
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "tags": item.get("tags", []),
                "url": item.get("link", ""),
                "body": (item.get("body", "") or "")[:1000],
                "creation_date": item.get("creation_date", 0),
                "view_count": item.get("view_count", 0),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_users(query: str, site: str = "stackoverflow", count: int = 10) -> dict:
    try:
        params: dict[str, Any] = {"site": site, "inname": query, "pagesize": min(count, 50)}
        c = get_http_client()
        r = await c.get(f"{SO_API}/users", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /users: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "user_id": item.get("user_id", 0),
                "display_name": item.get("display_name", ""),
                "reputation": item.get("reputation", 0),
                "accept_rate": item.get("accept_rate", 0),
                "profile_url": item.get("link", ""),
                "location": item.get("location", ""),
                "website_url": item.get("website_url", ""),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_tags(query: str, site: str = "stackoverflow", count: int = 10) -> dict:
    try:
        params: dict[str, Any] = {"site": site, "inname": query, "pagesize": min(count, 50)}
        c = get_http_client()
        r = await c.get(f"{SO_API}/tags", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /tags: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "name": item.get("name", ""),
                "has_synonyms": item.get("has_synonyms", False),
                "is_moderator_only": item.get("is_moderator_only", False),
                "is_required": item.get("is_required", False),
                "count": item.get("count", 0),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_question_comments(question_id: int, site: str = "stackoverflow", count: int = 20) -> dict:
    try:
        params: dict[str, Any] = {"site": site, "pagesize": min(count, 50), "filter": "withbody"}
        c = get_http_client()
        r = await c.get(f"{SO_API}/questions/{question_id}/comments", params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"StackExchange /questions/comments: {r.status_code}"}
        data = r.json()
        results = []
        for item in data.get("items", [])[:count]:
            results.append({
                "comment_id": item.get("comment_id", 0),
                "score": item.get("score", 0),
                "body": (item.get("body", "") or "")[:1000],
                "url": item.get("link", ""),
                "creation_date": item.get("creation_date", 0),
                "user_name": item.get("user", {}).get("display_name", ""),
            })
        return {"success": True, "results": results, "total": data.get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
