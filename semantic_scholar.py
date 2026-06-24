from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH = f"{S2_API}/paper/search"
S2_PAPER = f"{S2_API}/paper"
S2_AUTHOR = f"{S2_API}/author"
S2_RECOMMENDATIONS = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"

S2_API_KEY = os.environ.get("S2_API_KEY", "")

FIELDS = (
    "paperId,title,year,abstract,citationCount,url,authors,venue,"
    "publicationDate,externalIds,tldr,isOpenAccess,openAccessPdf,"
    "s2FieldsOfStudy,publicationTypes,referenceCount,influentialCitationCount"
)

_AUTHOR_FIELDS = "authorId,name,affiliations,citationCount,hIndex,paperCount,url"

_S2_RETRIES = 3


def _s2_headers() -> dict[str, str]:
    h: dict[str, str] = {}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


async def _s2_get(url: str, params: dict | None = None, timeout: int = 15) -> httpx.Response:
    for attempt in range(_S2_RETRIES):
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url, params=params, headers=_s2_headers())
        if r.status_code != 429 or attempt == _S2_RETRIES - 1:
            return r
        await asyncio.sleep(2 ** attempt)
    return r


async def _s2_post(url: str, json: dict, params: dict | None = None, timeout: int = 30) -> httpx.Response:
    for attempt in range(_S2_RETRIES):
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(url, json=json, params=params, headers=_s2_headers())
        if r.status_code != 429 or attempt == _S2_RETRIES - 1:
            return r
        await asyncio.sleep(2 ** attempt)
    return r


def _extract_paper(p: dict) -> dict:
    authors = [a.get("name", "") for a in (p.get("authors") or [])]
    tldr = ""
    if p.get("tldr"):
        tldr = p["tldr"].get("text", "") if isinstance(p["tldr"], dict) else str(p["tldr"])
    open_access_pdf = None
    oa = p.get("openAccessPdf")
    if oa and isinstance(oa, dict):
        open_access_pdf = oa.get("url", "")
    elif oa and isinstance(oa, str):
        open_access_pdf = oa
    return {
        "paperId": p.get("paperId", ""),
        "title": p.get("title", ""),
        "year": p.get("year"),
        "abstract": (p.get("abstract", "") or "")[:1000],
        "citationCount": p.get("citationCount", 0),
        "url": p.get("url", ""),
        "venue": p.get("venue", ""),
        "publicationDate": p.get("publicationDate", ""),
        "authors": authors[:10],
        "externalIds": p.get("externalIds", {}),
        "tldr": tldr,
        "isOpenAccess": p.get("isOpenAccess", False),
        "openAccessPdf": open_access_pdf,
        "s2FieldsOfStudy": [f.get("category", "") for f in (p.get("s2FieldsOfStudy") or [])],
        "publicationTypes": p.get("publicationTypes", []),
        "referenceCount": p.get("referenceCount", 0),
        "influentialCitationCount": p.get("influentialCitationCount", 0),
    }


async def search_papers(query: str, limit: int = 10, year: str = "",
                        fields_of_study: str = "", open_access: bool = False,
                        offset: int = 0) -> dict:
    try:
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": FIELDS,
        }
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if open_access:
            params["openAccessPdf"] = ""
        if offset:
            params["offset"] = offset
        r = await _s2_get(S2_SEARCH, params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 API: {r.status_code} {r.text[:200]}"}
        data = r.json()
        papers = [_extract_paper(p) for p in (data.get("data", []) or [])]
        return {
            "success": True,
            "results": papers,
            "total": data.get("total", len(papers)),
            "offset": data.get("offset", 0),
        }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_paper_details(paper_id: str) -> dict:
    try:
        params = {"fields": FIELDS}
        r = await _s2_get(f"{S2_PAPER}/{paper_id}", params)
        if r.status_code == 404:
            return {"success": False, "error": f"Paper not found: {paper_id}"}
        if r.status_code != 200:
            return {"success": False, "error": f"S2 API: {r.status_code} {r.text[:200]}"}
        p = r.json()
        return {"success": True, "paper": _extract_paper(p)}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_papers_batch(paper_ids: list[str]) -> dict:
    try:
        params = {"fields": FIELDS}
        r = await _s2_post(f"{S2_PAPER}/batch", {"ids": paper_ids[:500]}, params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 batch: {r.status_code} {r.text[:200]}"}
        papers = []
        for p in (r.json().get("data", []) or []):
            if p is None:
                continue
            papers.append(_extract_paper(p))
        return {"success": True, "papers": papers}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_paper_citations(paper_id: str, limit: int = 20) -> dict:
    try:
        params = {"fields": FIELDS, "limit": min(limit, 100)}
        r = await _s2_get(f"{S2_PAPER}/{paper_id}/citations", params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 citations: {r.status_code} {r.text[:200]}"}
        cites = []
        for item in (r.json().get("data", []) or [])[:limit]:
            p = item.get("citingPaper", {}) or {}
            cites.append(_extract_paper(p))
        return {"success": True, "citations": cites, "total": r.json().get("total", len(cites))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_paper_references(paper_id: str, limit: int = 20) -> dict:
    try:
        params = {"fields": FIELDS, "limit": min(limit, 100)}
        r = await _s2_get(f"{S2_PAPER}/{paper_id}/references", params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 references: {r.status_code} {r.text[:200]}"}
        refs = []
        for item in (r.json().get("data", []) or [])[:limit]:
            p = item.get("citedPaper", {}) or {}
            refs.append(_extract_paper(p))
        return {"success": True, "references": refs, "total": r.json().get("total", len(refs))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_paper_recommendations(paper_id: str, limit: int = 10) -> dict:
    try:
        params: dict[str, Any] = {"fields": FIELDS, "limit": min(limit, 100)}
        r = await _s2_post(S2_RECOMMENDATIONS, {"paperId": paper_id}, params=params, timeout=20)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 recommendations: {r.status_code} {r.text[:200]}"}
        data = r.json()
        papers = [_extract_paper(p) for p in (data.get("recommendedPapers", []) or [])[:limit]]
        return {"success": True, "recommendations": papers, "total": len(papers)}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def search_authors(query: str, limit: int = 10) -> dict:
    try:
        params: dict[str, Any] = {"query": query, "limit": min(limit, 100), "fields": _AUTHOR_FIELDS}
        r = await _s2_get(f"{S2_AUTHOR}/search", params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 author search: {r.status_code} {r.text[:200]}"}
        data = r.json()
        authors = []
        for a in (data.get("data", []) or [])[:limit]:
            authors.append({
                "authorId": a.get("authorId", ""),
                "name": a.get("name", ""),
                "affiliations": a.get("affiliations", []),
                "citationCount": a.get("citationCount", 0),
                "hIndex": a.get("hIndex", 0),
                "paperCount": a.get("paperCount", 0),
                "url": a.get("url", ""),
            })
        return {"success": True, "results": authors, "total": data.get("total", len(authors))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_author_papers(author_id: str, limit: int = 10) -> dict:
    try:
        params: dict[str, Any] = {"fields": FIELDS, "limit": min(limit, 100)}
        r = await _s2_get(f"{S2_AUTHOR}/{author_id}/papers", params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 author papers: {r.status_code} {r.text[:200]}"}
        data = r.json()
        papers = [_extract_paper(p) for p in (data.get("data", []) or [])[:limit]]
        return {"success": True, "papers": papers, "total": data.get("total", len(papers))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def autocomplete_papers(query: str) -> dict:
    try:
        params: dict[str, Any] = {"query": query}
        r = await _s2_get(f"{S2_PAPER}/autocomplete", params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 autocomplete: {r.status_code} {r.text[:200]}"}
        data = r.json()
        results = []
        for item in (data.get("matches", []) or []):
            results.append({
                "paperId": item.get("paperId", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "year": item.get("year"),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])],
            })
        return {"success": True, "results": results}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}
