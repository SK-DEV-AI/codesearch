"""SearchCode — per-repo code analysis via api.searchcode.com.

SearchCode's old global code search API (codesearch_I/) is retired. The
new product provides per-repo analysis: repo overview, file tree browsing,
code search within a repo, file reading, and code quality findings.

All endpoints are POST with ?client= parameter. Free, no auth, no API key.
Rate limit: unknown but generous (no 429 observed).

Unique tools vs PkgSeer:
- code_analyze: instant repo overview (languages, complexity, file count,
  tech stack, code quality findings, credential scanning) — nothing else
  in our server does this
- code_get_findings: code quality issues by category/severity — unique
"""

from __future__ import annotations

from typing import Any

import httpx

from config import get_http_client

_API = "https://api.searchcode.com/api/v1"
_CLIENT = "opencode-codesearch"
_TIMEOUT = 30


async def _call(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to a SearchCode API endpoint and return the JSON response."""
    url = f"{_API}/{endpoint}?client={_CLIENT}"
    try:
        resp = await get_http_client().post(url, json=body, timeout=_TIMEOUT)
    except httpx.TimeoutException:
        return {"success": False, "error": f"SearchCode {endpoint} timed out"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"SearchCode request failed: {e}"}
    if resp.status_code == 404:
        return {"success": False, "error": "Repository or file not found"}
    if resp.status_code == 429:
        return {"success": False, "error": "SearchCode rate limited (429)"}
    if resp.status_code >= 400:
        body = " ".join((resp.text or "").split())[:250]
        return {"success": False, "error": f"SearchCode error ({resp.status_code}) {body}".rstrip()}
    try:
        data = resp.json()
    except Exception as e:
        return {"success": False, "error": f"SearchCode JSON parse error: {e}"}
    return {"success": True, "data": data}


async def analyze(repository: str, language: str = "", path: str = "", detail_level: str = "summary") -> dict[str, Any]:
    """Instant repo overview: languages, complexity, file count, tech stack,
    code quality findings, credential scanning.

    Use as your first call to understand any remote public codebase.
    Supports language filtering and path scoping (for large monorepos).

    Returns structured pre-computed data:
    - languages with SLOC percentages
    - file counts and directory structure
    - tech stack detection
    - code quality findings (credentials, complexity hotspots)
    - license and metadata
    """
    body: dict[str, Any] = {"repository": repository, "detail_level": detail_level}
    if language:
        body["language"] = language
    if path:
        body["path"] = path
    return await _call("code_analyze", body)


async def search(repository: str, query: str, max_results: int = 10, context_lines: int = 2, case_sensitive: bool = False) -> dict[str, Any]:
    """Fast code search across any public git repo.

    Returns file paths, line numbers, and code snippets with context.
    Supports regex, boolean queries, fuzzy matching, and structural filters.
    Dependency/build directories excluded by default.

    Differs from PkgSeer code_grep: takes GitHub URLs not package specs,
    no token needed, handles orphan/generated repos better.
    """
    body: dict[str, Any] = {
        "repository": repository, "query": query,
        "max_results": min(max_results, 100), "context_lines": min(context_lines, 20),
        "case_sensitive": case_sensitive,
    }
    return await _call("code_search", body)


async def file_tree(repository: str, path_filter: str = "", query: str = "", max_depth: int = 0) -> dict[str, Any]:
    """List files and directories in any public git repo.

    Supports fuzzy file search (query parameter), language/path filtering,
    and depth control. Returns hierarchical file tree or fuzzy matches.
    Dependency/build directories excluded by default.
    """
    body: dict[str, Any] = {"repository": repository}
    if path_filter:
        body["path_filter"] = path_filter
    if query:
        body["query"] = query
    if max_depth > 0:
        body["max_depth"] = max_depth
    return await _call("code_file_tree", body)


async def get_file(repository: str, path: str, symbol_name: str = "", start_line: int = 1, end_line: int = 0, max_lines: int = 500) -> dict[str, Any]:
    """Get code from a remote public git repository.

    PREFERRED: use symbol_name to extract just a function/class declaration
    (avoids fetching entire files). Alternative: provide line range.
    Max 2000 lines per file.
    """
    body: dict[str, Any] = {"repository": repository, "path": path, "max_lines": min(max_lines, 2000)}
    if symbol_name:
        body["symbol_name"] = symbol_name
    else:
        body["start_line"] = start_line
        if end_line > 0:
            body["end_line"] = end_line
    return await _call("code_get_file", body)


async def findings(repository: str, path: str = "", severity: str = "", category: str = "", max_results: int = 50) -> dict[str, Any]:
    """Get code quality findings from a remote public git repository.

    Returns rule IDs, line numbers, severity, category, descriptions, and
    source snippets. Supports filtering by file path, severity (error/warning/info),
    and category (security, deprecated, safety, correctness, etc.).
    """
    body: dict[str, Any] = {"repository": repository, "max_results": min(max_results, 200)}
    if path:
        body["path"] = path
    if severity:
        body["severity"] = severity
    if category:
        body["category"] = category
    return await _call("code_get_findings", body)
