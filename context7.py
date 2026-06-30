from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import Any

import httpx

from config import CONTEXT7_API_KEY, CONTEXT7_CONTEXT, CONTEXT7_SEARCH, _cached, _KeyRotator, get_http_client

_c7_rotator = _KeyRotator("CONTEXT7_API_KEY")
_next_c7_key = _c7_rotator.next


async def _context7_headers() -> dict[str, str]:
    h = {"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"}
    k = await _next_c7_key()
    if k:
        h["Authorization"] = f"Bearer {k}"
    return h


async def context7_search_lib(query: str, fast: bool = False, page: int = 1, limit: int = 5) -> list[dict]:
    try:
        params: dict[str, Any] = {"libraryName": query, "query": query, "page": page, "limit": limit}
        if fast:
            params["fast"] = "true"
        c = get_http_client()
        r = await c.get(CONTEXT7_SEARCH, params=params, headers=await _context7_headers())
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for lib in (data if isinstance(data, list) else data.get("results", [])):
            results.append({
                "id": lib.get("id", ""),
                "name": lib.get("name", lib.get("title", "")),
                "description": lib.get("description", ""),
                "trust_score": lib.get("trustScore", lib.get("trust_score", 0)),
                "benchmark_score": lib.get("benchmarkScore", lib.get("benchmark_score")),
                "total_tokens": lib.get("totalTokens", lib.get("total_tokens")),
                "total_snippets": lib.get("totalSnippets", lib.get("total_snippets")),
                "versions": lib.get("versions", []),
                "state": lib.get("state", ""),
                "url": lib.get("url", ""),
            })
        return results
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return []


async def context7_fetch_docs(library_id: str, query: str, fast: bool = False) -> dict:
    if not library_id or not query:
        return {"success": False, "error": "both library_id and query are required"}
    try:
        params: dict[str, Any] = {"libraryId": library_id, "query": query, "type": "json"}
        if fast:
            params["fast"] = "true"
        c = get_http_client()
        r = await c.get(CONTEXT7_CONTEXT, params=params, headers=await _context7_headers())
        if r.status_code == 202:
            return {"success": False, "error": "library still processing, retry later"}
        if r.status_code == 301:
            return {"success": False, "error": "library not found", "hint": "check library ID format"}
        if r.status_code != 200:
            return {"success": False, "error": f"context7 returned {r.status_code}"}
        data = r.json()
        snippets = []
        code_snippets = []
        for s in data.get("infoSnippets", []):
            snippets.append({
                "title": s.get("breadcrumb", ""),
                "content": s.get("content", ""),
                "url": s.get("pageId", ""),
            })
        for s in data.get("codeSnippets", []):
            code = ""
            for cl in s.get("codeList", []):
                code += cl.get("code", "")
            code_snippets.append({
                "title": s.get("codeTitle", ""),
                "description": s.get("codeDescription", ""),
                "language": s.get("codeLanguage", ""),
                "code": code,
            })
        return {
            "success": True,
            "snippets": snippets,
            "code_snippets": code_snippets,
            "total_tokens": data.get("totalTokens", data.get("total_tokens")),
            "total_snippets": data.get("totalSnippets", data.get("total_snippets")),
        }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def context7_resolve(query: str, version: str = "", fast: bool = False, library_id: str = "") -> dict:
    if library_id:
        lib_id = library_id
        if version:
            lib_id = f"{lib_id}@{version}"
        docs = await context7_fetch_docs(lib_id, query, fast=fast)
        return {
            "success": docs.get("success", False),
            "library": {"id": library_id, "name": library_id},
            "docs": docs,
        }
    libs = await context7_search_lib(query, fast=fast)
    if not libs:
        return {"success": False, "error": "no matching library found"}
    ql = query.lower().strip()
    exact = [l for l in libs if l["name"].lower().strip() == ql]
    if exact:
        best = exact[0]
    else:
        best = libs[0]
    lib_id = best["id"]
    if version:
        lib_id = f"{lib_id}@{version}"
    docs = await context7_fetch_docs(lib_id, query, fast=fast)
    if not docs.get("success") and len(libs) > 1:
        return {
            "success": False,
            "error": f"docs fetch failed for '{best['name']}', {len(libs)} candidates found",
            "candidates": [{"id": l["id"], "name": l["name"], "description": l["description"][:100]} for l in libs[:5]],
            "hint": "Retry with library_id parameter to pick the right library",
        }
    return {
        "success": True,
        "library": best,
        "docs": docs,
        "candidates": [{"id": l["id"], "name": l["name"]} for l in libs[:5]] if len(libs) > 1 else None,
    }


async def fetch_llms_txt(url: str) -> dict:
    try:
        if not url.endswith("llms.txt"):
            url = url.rstrip("/") + "/llms.txt"
        c = get_http_client()
        r = await c.get(url, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"HTTP {r.status_code}"}
        text = r.text
        links = []
        lines = text.split("\n")
        current_section = ""
        for line in lines:
            stripped = line.strip()
            md_link = re.match(r"^\s*[-*]\s*\[([^\]]+)\]\(([^)]+)\)\s*(?::\s*(.+))?$", stripped)
            if md_link:
                links.append({
                    "title": md_link.group(1),
                    "url": urllib.parse.urljoin(url, md_link.group(2)),
                    "description": (md_link.group(3) or "").strip(),
                    "section": current_section,
                })
            elif stripped.startswith("## "):
                current_section = stripped.lstrip("# ")
        return {"success": True, "llms_txt": text[:3000], "links": links}
    except (httpx.HTTPError, ValueError, ImportError) as e:
        return {"success": False, "error": str(e)}


async def fetch_doc_url(url: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(url, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": f"HTTP {r.status_code}"}
        content = r.text
        try:
            import markdownify
            md = markdownify.markdownify(content, heading_style="ATX")
        except ImportError:
            md = content[:10000]
        return {"success": True, "url": url, "content": md[:10000]}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_llms_txt(domain: str, query: str) -> dict:
    url = domain if domain.startswith("http") else f"https://{domain}"
    parsed = await fetch_llms_txt(url)
    if not parsed.get("success"):
        return {"success": False, "error": parsed.get("error", "failed to fetch llms.txt")}
    links = parsed["links"]
    ql = query.lower()
    matched = [l for l in links if ql in l["title"].lower() or ql in l["description"].lower()]
    if matched:
        results = []
        for link in matched[:3]:
            doc = await fetch_doc_url(link["url"])
            if doc.get("success"):
                results.append({
                    "title": link["title"],
                    "url": link["url"],
                    "content": doc["content"],
                    "section": link["section"],
                })
        return {"success": True, "results": results}
    return {"success": True, "results": [], "note": "no matching links found in llms.txt"}


async def context7_add_repo(provider: str, repo_url: str) -> dict:
    """Submit a repository for Context7 indexing."""
    try:
        c = get_http_client()
        r = await c.post(f"https://context7.com/api/v2/add/repo/{provider}",
            json={"repo_url": repo_url}, headers=await _context7_headers(), timeout=30)
        if r.status_code != 200:
            return {"success": False, "error": f"Context7 add_repo: {r.status_code}"}
        d = r.json()
        return {"success": True, "message": d.get("message", "repo submitted"), "id": d.get("id", "")}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
