"""GitHub Security Advisory (GHSA) enrichment.

GET api.github.com/advisories using the existing GITHUB_TOKEN (core bucket,
5000/hr). Enriches vulns detail (CVE/GHSA text) and pkg info severity.
No new key needed.
"""

from __future__ import annotations

from typing import Any

import httpx

from config import GH_API, _cached, _set_cache, _next_gh_key, get_http_client

_TIMEOUT = 15


async def _ghsa_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        key = await _next_gh_key()
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "mcp-codesearch/1.0",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        c = get_http_client()
        r = await c.get(f"{GH_API}{path}", params=params or {},
                        headers=headers, timeout=_TIMEOUT)
        if r.status_code == 404:
            return {"success": False, "error": "Advisory not found"}
        if r.status_code == 401:
            return {"success": False, "error": "GitHub token rejected (401)"}
        if r.status_code == 403 and "rate limit" in (r.text or "").lower():
            return {"success": False, "error": "GitHub API rate limited (403)"}
        if r.status_code != 200:
            return {"success": False,
                    "error": f"GitHub Advisory HTTP {r.status_code} {r.text[:200]}"}
        return {"success": True, "data": r.json()}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


def _extract(a: dict) -> dict[str, Any]:
    cvss = a.get("cvss") or {}
    cwes = [c.get("cwe_id", "") for c in (a.get("cwes", []) or [])
            if c.get("cwe_id")]
    vuln_pkgs = []
    for v in (a.get("vulnerabilities", []) or [])[:10]:
        pkg = v.get("package", {}) or {}
        vuln_pkgs.append({
            "ecosystem": pkg.get("ecosystem", ""),
            "name": pkg.get("name", ""),
            "patched_versions": v.get("patched_versions", ""),
            "vulnerable_range": v.get("vulnerable_version_range", ""),
            "first_patched": (v.get("first_patched_version", "") or ""),
        })
    return {
        "ghsa_id": a.get("ghsa_id", ""),
        "cve_id": a.get("cve_id", ""),
        "summary": a.get("summary", ""),
        "description": (a.get("description", "") or "")[:2000],
        "severity": a.get("severity", ""),
        "cvss_score": (cvss.get("score", 0) or 0),
        "cvss_vector": cvss.get("vector_string", ""),
        "cwes": cwes,
        "published_at": a.get("published_at", ""),
        "updated_at": a.get("updated_at", ""),
        "withdrawn": bool(a.get("withdrawn_at")),
        "html_url": a.get("html_url", ""),
        "references": [r for r in (a.get("references", []) or [])],
        "vulnerable_packages": vuln_pkgs,
    }


async def ghsa_detail(ghsa_id: str) -> dict[str, Any]:
    """Fetch one advisory by GHSA id (also accepts CVE id via search fallback)."""
    if not ghsa_id:
        return {"success": False, "error": "ghsa_id required"}
    gid = ghsa_id.strip()
    cache_key = f"ghsa:{gid}"
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "advisory": cached, "cached": True}
    if gid.startswith("CVE-"):
        r = await _ghsa_get("/advisories", {"cve_id": gid, "per_page": 1})
        if not r.get("success"):
            return r
        items = r["data"] if isinstance(r["data"], list) else []
        if not items:
            return {"success": False,
                    "error": f"no GitHub advisory found for {gid}"}
        adv = _extract(items[0])
    else:
        r = await _ghsa_get(f"/advisories/{gid}")
        if not r.get("success"):
            return r
        adv = _extract(r["data"])
    await _set_cache(cache_key, adv)
    return {"success": True, "advisory": adv}


async def ghsa_search(ecosystem: str = "", severity: str = "",
                      cve_id: str = "", count: int = 10) -> dict[str, Any]:
    """List advisories, optionally filtered by ecosystem/severity/CVE."""
    params: dict[str, Any] = {"per_page": min(max(count, 1), 100)}
    if ecosystem:
        params["ecosystem"] = ecosystem.lower()
    if severity:
        params["severity"] = severity.lower()
    if cve_id:
        params["cve_id"] = cve_id
    r = await _ghsa_get("/advisories", params)
    if not r.get("success"):
        return r
    items = r["data"] if isinstance(r["data"], list) else []
    return {"success": True, "count": len(items),
            "advisories": [_extract(a) for a in items[:count]]}
