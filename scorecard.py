"""OpenSSF Scorecard project health.

GET api.securityscorecards.dev/projects/github.com/{owner}/{repo}.
Keyless, generous. Verified live (expressjs/express: 200, score 8.9).
Used in pkg info + analyze as a supply-chain health signal.
"""

from __future__ import annotations

from typing import Any

import httpx

from config import _cached, _set_cache, get_http_client

_SCORECARD_API = "https://api.securityscorecards.dev/projects"
_TIMEOUT = 15


async def scorecard_score(owner: str, repo: str) -> dict[str, Any]:
    """Fetch OpenSSF Scorecard score + check summary for a GitHub repo."""
    owner, repo = (owner or "").strip(), (repo or "").strip()
    if not owner or not repo:
        return {"success": False, "error": "owner and repo required"}
    cache_key = f"scorecard:{owner}/{repo}"
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "scorecard": cached, "cached": True}
    try:
        c = get_http_client()
        r = await c.get(f"{_SCORECARD_API}/github.com/{owner}/{repo}",
                        timeout=_TIMEOUT)
        if r.status_code == 404:
            return {"success": False,
                    "error": f"no Scorecard data for {owner}/{repo}"}
        if r.status_code != 200:
            return {"success": False,
                    "error": f"Scorecard HTTP {r.status_code} {r.text[:200]}"}
        d = r.json()
        sc = d.get("scorecard", {}) or {}
        checks = []
        for ch in (sc.get("checks", []) or []):
            checks.append({"name": ch.get("name", ""),
                           "score": ch.get("score", 0),
                           "reason": (ch.get("reason", "") or "")[:200]})
        result = {"repo": (d.get("repo", {}) or {}).get("name", f"{owner}/{repo}"),
                  "score": d.get("score", 0),
                  "date": d.get("date", ""),
                  "commit": (d.get("repo", {}) or {}).get("commit", ""),
                  "checks": checks}
        await _set_cache(cache_key, result)
        return {"success": True, "scorecard": result}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
