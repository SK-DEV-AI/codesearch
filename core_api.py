from __future__ import annotations

import json
import os
from typing import Any

import httpx

from config import get_http_client

CORE_API = "https://api.core.ac.uk/v3"
CORE_SEARCH_WORKS = f"{CORE_API}/search/works/"
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")
CORE_API_AVAILABLE = bool(CORE_API_KEY.strip())

FIELDS = "id,title,yearPublished,authors,doi,abstract,downloadUrl,fullText,documentType,subjects,publisher,language,identifiers,links"


def _extract_work(w: dict) -> dict:
    return {
        "id": w.get("id", ""),
        "title": w.get("title", "") or "",
        "year": w.get("yearPublished", ""),
        "authors": [a.get("name", "") for a in (w.get("authors") or [])],
        "doi": w.get("doi", ""),
        "abstract": (w.get("abstract", "") or "")[:1000],
        "downloadUrl": w.get("downloadUrl", ""),
        "hasFullText": bool(w.get("fullText")),
        "documentType": w.get("documentType", []),
        "subjects": w.get("subjects", []),
        "publisher": w.get("publisher", ""),
        "language": w.get("language", {}).get("name", "") if isinstance(w.get("language"), dict) else "",
        "ids": w.get("identifiers", {}),
    }


async def search_core_works(query: str, limit: int = 10, offset: int = 0) -> dict:
    try:
        params: dict[str, Any] = {"q": query, "limit": min(limit, 100), "offset": max(offset, 0)}
        headers = {"Accept": "application/json"}
        if CORE_API_KEY:
            headers["Authorization"] = f"Bearer {CORE_API_KEY}"
        c = get_http_client()
        r = await c.get(CORE_SEARCH_WORKS, params=params, headers=headers, timeout=15)
        if r.status_code == 301:
            follow_url = r.headers.get("location", "")
            if follow_url:
                r = await c.get(follow_url, headers=headers, timeout=15)
        if r.status_code != 200:
            return {"success": False, "error": f"CORE API: {r.status_code}", "results": []}
        data = r.json()
        works = [_extract_work(w) for w in (data.get("results", []) or [])]
        return {
            "success": True,
            "results": works,
            "totalHits": data.get("totalHits", 0),
            "searchId": data.get("searchId", ""),
        }
    except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
        return {"success": False, "error": f"CORE: {e}", "results": []}
