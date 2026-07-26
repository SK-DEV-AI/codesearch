from __future__ import annotations

from typing import Any

from config import get_http_client

OPENALEX_API = "https://api.openalex.org"


def _abs_text(inverted: dict | None) -> str:
    if not isinstance(inverted, dict):
        return ""
    words = []
    pos_map: dict[int, str] = {}
    for word, positions in inverted.items():
        for p in (positions if isinstance(positions, list) else [positions]):
            pos_map[p] = word
    for _, word in sorted(pos_map.items()):
        words.append(word)
    return " ".join(words)


async def _openalex_get(endpoint: str, params: dict[str, Any]) -> dict:
    """Generic OpenAlex GET with error handling."""
    try:
        c = get_http_client()
        r = await c.get(
            f"{OPENALEX_API}/{endpoint}",
            params=params,
            headers={"User-Agent": "mcp-codesearch/1.0"},
            timeout=15,
        )
        if r.status_code != 200:
            return {"success": False, "error": f"OpenAlex {endpoint}: {r.status_code}"}
        return {"success": True, "data": r.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def search_openalex(query: str, count: int = 10) -> dict:
    r = await _openalex_get("works", {
        "search": query,
        "per_page": min(count, 50),
        "sort": "relevance_score:desc",
    })
    if not r.get("success"):
        return {"success": False, "results": []}
    data = r["data"]
    results = []
    for work in (data.get("results", []) or [])[:count]:
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (work.get("authorships", []) or [])
        ]
        results.append({
            "title": work.get("title", ""),
            "url": work.get("id", ""),
            "snippet": _abs_text(work.get("abstract_inverted_index"))[:500],
            "authors": authors,
            "year": work.get("publication_year"),
            "citations": work.get("cited_by_count", 0),
            "type": work.get("type", ""),
            "doi": work.get("doi", ""),
        })
    return {
        "success": True,
        "results": results,
        "total": data.get("meta", {}).get("count", 0),
    }


async def search_openalex_authors(query: str, count: int = 10) -> dict:
    """Search OpenAlex for authors (researchers)."""
    r = await _openalex_get("authors", {
        "search": query,
        "per_page": min(count, 50),
        "sort": "relevance_score:desc",
    })
    if not r.get("success"):
        return {"success": False, "error": r.get("error"), "results": []}
    data = r["data"]
    results = []
    for author in (data.get("results", []) or [])[:count]:
        results.append({
            "id": author.get("id", ""),
            "name": author.get("display_name", ""),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "h_index": author.get("summary_stats", {}).get("h_index", 0) if author.get("summary_stats") else 0,
            "2yr_mean_citations": author.get("summary_stats", {}).get("2yr_mean_citedness", 0) if author.get("summary_stats") else 0,
            "last_known_institution": (
                author.get("last_known_institutions", [{}])[0].get("display_name", "")
                if author.get("last_known_institutions") else ""
            ),
            "topics": [
                t.get("display_name", "")
                for t in (author.get("topics", []) or [])[:3]
            ],
            "works_api_url": author.get("works_api_url", ""),
        })
    return {
        "success": True,
        "results": results,
        "total": data.get("meta", {}).get("count", 0),
    }


async def search_openalex_concepts(query: str, count: int = 10) -> dict:
    """Search OpenAlex for research topics (4-level hierarchy: Domain→Field→Subfield→Topic)."""
    r = await _openalex_get("topics", {
        "search": query,
        "per_page": min(count, 50),
    })
    if not r.get("success"):
        return {"success": False, "error": r.get("error"), "results": []}
    data = r["data"]
    results = []
    for topic in (data.get("results", []) or [])[:count]:
        results.append({
            "id": topic.get("id", ""),
            "display_name": topic.get("display_name", ""),
            "description": (topic.get("description", "") or "")[:300],
            "subfield": topic.get("subfield", {}).get("display_name", ""),
            "field": topic.get("field", {}).get("display_name", ""),
            "domain": topic.get("domain", {}).get("display_name", ""),
            "works_count": topic.get("works_count", 0),
            "cited_by_count": topic.get("cited_by_count", 0),
            "works_api_url": topic.get("works_api_url", ""),
        })
    return {
        "success": True,
        "results": results,
        "total": data.get("meta", {}).get("count", 0),
    }


async def search_openalex_institutions(query: str, count: int = 10) -> dict:
    """Search OpenAlex for research institutions (universities, labs)."""
    r = await _openalex_get("institutions", {
        "search": query,
        "per_page": min(count, 50),
        "sort": "relevance_score:desc",
    })
    if not r.get("success"):
        return {"success": False, "error": r.get("error"), "results": []}
    data = r["data"]
    results = []
    for inst in (data.get("results", []) or [])[:count]:
        results.append({
            "id": inst.get("id", ""),
            "name": inst.get("display_name", ""),
            "country": inst.get("country_code", ""),
            "type": inst.get("type", ""),
            "works_count": inst.get("works_count", 0),
            "cited_by_count": inst.get("cited_by_count", 0),
            "2yr_mean_citations": inst.get("summary_stats", {}).get("2yr_mean_citedness", 0) if inst.get("summary_stats") else 0,
            "homepage": inst.get("homepage_url", ""),
            "image_url": inst.get("image_url", ""),
            "works_api_url": inst.get("works_api_url", ""),
        })
    return {
        "success": True,
        "results": results,
        "total": data.get("meta", {}).get("count", 0),
    }


async def search_openalex_sources(query: str, count: int = 10) -> dict:
    """Search OpenAlex for sources (journals, conferences, repositories)."""
    r = await _openalex_get("sources", {
        "search": query,
        "per_page": min(count, 50),
        "sort": "relevance_score:desc",
    })
    if not r.get("success"):
        return {"success": False, "error": r.get("error"), "results": []}
    data = r["data"]
    results = []
    for src in (data.get("results", []) or [])[:count]:
        results.append({
            "id": src.get("id", ""),
            "display_name": src.get("display_name", ""),
            "type": src.get("type", ""),
            "host_organization": src.get("host_organization_name", "") or "",
            "works_count": src.get("works_count", 0),
            "cited_by_count": src.get("cited_by_count", 0),
            "is_in_doaj": src.get("is_in_doaj", False),
            "is_oa": src.get("is_oa", False),
            "homepage_url": src.get("homepage_url", ""),
            "works_api_url": src.get("works_api_url", ""),
        })
    return {
        "success": True,
        "results": results,
        "total": data.get("meta", {}).get("count", 0),
    }
