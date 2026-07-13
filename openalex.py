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


async def search_openalex(query: str, count: int = 10) -> dict:
    try:
        params: dict[str, Any] = {
            "search": query,
            "per_page": min(count, 50),
            "sort": "relevance:desc",
        }
        c = get_http_client()
        r = await c.get(
            f"{OPENALEX_API}/works",
            params=params,
            headers={"User-Agent": "mcp-codesearch/1.0"},
            timeout=15,
        )
        if r.status_code != 200:
            return {"success": False, "results": []}
        data = r.json()
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
    except Exception:
        return {"success": False, "results": []}
