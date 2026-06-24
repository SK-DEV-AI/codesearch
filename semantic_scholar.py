from __future__ import annotations

from typing import Any

import httpx

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH = f"{S2_API}/paper/search"
S2_PAPER = f"{S2_API}/paper"

FIELDS = "paperId,title,year,abstract,citationCount,url,authors,venue,publicationDate,externalIds"


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
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(S2_SEARCH, params=params)
        if r.status_code == 429:
            return {"success": False, "error": "Semantic Scholar rate limited (100 req/5min)"}
        if r.status_code != 200:
            return {"success": False, "error": f"S2 API: {r.status_code} {r.text[:200]}"}
        data = r.json()
        papers = []
        for p in (data.get("data", []) or []):
            authors = [a.get("name", "") for a in (p.get("authors", []) or [])]
            papers.append({
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
            })
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
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{S2_PAPER}/{paper_id}", params=params)
        if r.status_code == 404:
            return {"success": False, "error": f"Paper not found: {paper_id}"}
        if r.status_code == 429:
            return {"success": False, "error": "Semantic Scholar rate limited"}
        if r.status_code != 200:
            return {"success": False, "error": f"S2 API: {r.status_code} {r.text[:200]}"}
        p = r.json()
        authors = [a.get("name", "") for a in (p.get("authors", []) or [])]
        return {
            "success": True,
            "paper": {
                "paperId": p.get("paperId", ""),
                "title": p.get("title", ""),
                "year": p.get("year"),
                "abstract": (p.get("abstract", "") or "")[:2000],
                "citationCount": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "venue": p.get("venue", ""),
                "publicationDate": p.get("publicationDate", ""),
                "authors": authors,
                "externalIds": p.get("externalIds", {}),
            },
        }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_papers_batch(paper_ids: list[str]) -> dict:
    """Fetch up to 500 paper details in a single POST call."""
    try:
        params = {"fields": FIELDS}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{S2_PAPER}/batch", params=params,
                             json={"ids": paper_ids[:500]})
        if r.status_code != 200:
            return {"success": False, "error": f"S2 batch: {r.status_code} {r.text[:200]}"}
        papers = []
        for p in (r.json().get("data", []) or []):
            if p is None:
                continue
            authors = [a.get("name", "") for a in (p.get("authors", []) or [])]
            papers.append({
                "paperId": p.get("paperId", ""),
                "title": p.get("title", ""),
                "year": p.get("year"),
                "abstract": (p.get("abstract", "") or "")[:1000],
                "citationCount": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "venue": p.get("venue", ""),
                "authors": authors[:10],
            })
        return {"success": True, "papers": papers}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_paper_citations(paper_id: str, limit: int = 20) -> dict:
    """Get papers that cite this paper (forward citations)."""
    try:
        params = {"fields": FIELDS, "limit": min(limit, 100)}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{S2_PAPER}/{paper_id}/citations", params=params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 citations: {r.status_code} {r.text[:200]}"}
        cites = []
        for item in (r.json().get("data", []) or [])[:limit]:
            p = item.get("citingPaper", {}) or {}
            authors = [a.get("name", "") for a in (p.get("authors", []) or [])]
            cites.append({
                "paperId": p.get("paperId", ""),
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citationCount": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "authors": authors[:5],
            })
        return {"success": True, "citations": cites, "total": r.json().get("total", len(cites))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def get_paper_references(paper_id: str, limit: int = 20) -> dict:
    """Get papers this paper references (backward citations)."""
    try:
        params = {"fields": FIELDS, "limit": min(limit, 100)}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{S2_PAPER}/{paper_id}/references", params=params)
        if r.status_code != 200:
            return {"success": False, "error": f"S2 references: {r.status_code} {r.text[:200]}"}
        refs = []
        for item in (r.json().get("data", []) or [])[:limit]:
            p = item.get("citedPaper", {}) or {}
            authors = [a.get("name", "") for a in (p.get("authors", []) or [])]
            refs.append({
                "paperId": p.get("paperId", ""),
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citationCount": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "authors": authors[:5],
            })
        return {"success": True, "references": refs, "total": r.json().get("total", len(refs))}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}
