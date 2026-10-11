"""Sourcegraph public-repo code search lane.

POST sourcegraph.com/.api/graphql, no auth, public repos only.
Verified live: resultCount + FileMatch (path/url/lineMatches) shape.
Complements PkgSeer (package-scoped) and searchcode (per-repo analysis):
Sourcegraph is the global public-code lane.
"""

from __future__ import annotations

from typing import Any

import httpx

from config import get_http_client

_SG_GQL = "https://sourcegraph.com/.api/graphql"
_TIMEOUT = 30

_SEARCH_QUERY = """query($q:String!){
search(query:$q,version:V3){
results{
resultCount
results{
__typename
... on FileMatch{
file{path url}
repository{name}
lineMatches{preview lineNumber}
}
... on CommitSearchResult{
commit{oid message url}
}
}
}
}
}"""


def _build_query(query: str, repo: str = "", lang: str = "",
                 count: int = 10) -> str:
    # NOTE: no `type:code` filter — it suppresses results on the public
    # endpoint (verified: same query 0 with / 3 without). FileMatch-only
    # extraction keeps the lane code-typed regardless.
    parts = ["context:global", f"count:{min(max(count, 1), 50)}"]
    if repo:
        r = repo.strip().removesuffix(".git").rstrip("/")
        if "github.com/" in r:
            r = r.split("github.com/", 1)[1]
        # Sourcegraph repo: filters need the github.com/ prefix to match.
        # NOTE: do NOT re.escape — escaped dots/slashes match nothing.
        if "/" in r:
            if not r.startswith("github.com/"):
                r = f"github.com/{r}"
            parts.append(f"repo:^{r}$")
        else:
            parts.append(f"repo:{r}")
    if lang:
        parts.append(f"lang:{lang}")
    parts.append(query)
    return " ".join(parts)


def _extract(data: dict) -> list[dict]:
    out = []
    try:
        res = data["data"]["search"]["results"]
    except (KeyError, TypeError):
        return out
    total = res.get("resultCount", 0)
    for item in (res.get("results", []) or []):
        if item.get("__typename") == "FileMatch":
            f = item.get("file", {}) or {}
            repo = (item.get("repository", {}) or {}).get("name", "")
            url = f.get("url", "")
            if url and url.startswith("/"):
                url = f"https://sourcegraph.com{url}"
            matches = []
            for lm in (item.get("lineMatches", []) or [])[:5]:
                matches.append({"preview": lm.get("preview", "")[:300],
                                "line": lm.get("lineNumber", 0)})
            out.append({"source": "sourcegraph",
                        "repo": repo.replace("github.com/", ""),
                        "file": f.get("path", ""),
                        "url": url,
                        "matches": matches,
                        "title": f.get("path", ""),
                        "text": " ".join(m.get("preview", "")
                                         for m in matches)[:500]})
        elif item.get("__typename") == "CommitSearchResult":
            c = item.get("commit", {}) or {}
            out.append({"source": "sourcegraph",
                        "repo": "",
                        "commit": c.get("oid", "")[:12],
                        "message": (c.get("message", "") or "")[:300],
                        "url": c.get("url", ""),
                        "title": (c.get("message", "") or "").split("\n")[0][:120],
                        "text": (c.get("message", "") or "")[:500]})
    return out


async def search_sourcegraph(query: str, repo: str = "", lang: str = "",
                             count: int = 10) -> dict[str, Any]:
    """Global code search over public repos. Keyless. Returns file matches."""
    if not query.strip():
        return {"success": False, "error": "query required"}
    q = _build_query(query, repo, lang, count)
    try:
        c = get_http_client()
        r = await c.post(_SG_GQL,
                         json={"query": _SEARCH_QUERY,
                               "variables": {"q": q}},
                         headers={"Content-Type": "application/json",
                                  "User-Agent": "mcp-codesearch/1.0"},
                         timeout=_TIMEOUT)
        if r.status_code != 200:
            return {"success": False,
                    "error": f"Sourcegraph HTTP {r.status_code} {r.text[:200]}"}
        try:
            data = r.json()
        except ValueError as e:
            return {"success": False,
                    "error": f"Sourcegraph JSON parse error: {e}"}
        if isinstance(data, dict) and data.get("errors"):
            msgs = [e.get("message", str(e))
                    for e in data["errors"]][:3]
            return {"success": False,
                    "error": f"Sourcegraph: {'; '.join(msgs)}"}
        results = _extract(data)
        try:
            total = data["data"]["search"]["results"].get("resultCount", 0)
        except (KeyError, TypeError):
            total = len(results)
        return {"success": True, "results": results,
                "total": total, "query": q}
    except httpx.TimeoutException:
        return {"success": False, "error": "Sourcegraph timed out"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"Sourcegraph request failed: {e}"}
