"""OSV.dev fallback for vulnerability scanning.

Keyless, no published rate limits, batch up to 1000 queries per call.
Used when Sonatype Guide is unavailable (401/429/credit-exhausted) or
when OSS_TOKEN is not configured at all.

Response shape is kept compatible with oss_index.scan_vulnerabilities
(reports -> vulnerabilities with id/title/cvss_score/severity/cve/cwe/
description/references), plus an OSV detail fetcher for vuln enrichment.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

import httpx

from config import get_http_client

_OSV_QUERYBATCH = "https://api.osv.dev/v1/querybatch"
_OSV_VULN = "https://api.osv.dev/v1/vulns"

# purl type -> OSV ecosystem name
_ECOSYSTEMS = {
    "npm": "npm",
    "pypi": "PyPI",
    "crates": "crates.io",
    "cargo": "crates.io",
    "maven": "Maven",
    "golang": "Go",
    "go": "Go",
    "rubygems": "RubyGems",
    "packagist": "Packagist",
    "nuget": "NuGet",
    "pub": "Pub",
    "hex": "Hex",
    "swift": "SwiftURL",
    "cocoapods": "CocoaPods",
    "cran": "CRAN",
    "conan": "ConanCenter",
    "alpine": "Alpine",
    "debian": "Debian",
}

_SEV_MAP = {"LOW": "low", "MODERATE": "medium", "MEDIUM": "medium",
            "HIGH": "high", "CRITICAL": "critical"}

_MAX_DETAIL_FETCH = 10  # bound per-purl detail fan-out


def _parse_purl(purl: str) -> tuple[str, str, str]:
    """Split 'pkg:type/[namespace/]name@version' -> (name, ecosystem, version)."""
    s = purl.strip()
    if s.startswith("pkg:"):
        s = s[4:]
    ptype = "npm"
    rest = s
    if "/" in s:
        maybe_type, rest = s.split("/", 1)
        if "@" not in maybe_type and "." not in maybe_type:
            ptype = maybe_type.lower()
    name, version = rest, ""
    if "@" in rest:
        name, version = rest.rsplit("@", 1)
    name = urllib.parse.unquote(name)
    return name, _ECOSYSTEMS.get(ptype, ""), version


def _extract_vuln(v: dict) -> dict[str, Any]:
    db = v.get("database_specific", {}) or {}
    sev_raw = str(db.get("severity", "")).upper()
    severity = _SEV_MAP.get(sev_raw, "unknown")
    aliases = v.get("aliases", []) or []
    cve = next((a for a in aliases if a.startswith("CVE-")), "")
    cwes = db.get("cwe_ids", []) or []
    refs = [r.get("url", "") for r in (v.get("references", []) or [])
            if r.get("url")]
    affected = []
    for a in (v.get("affected", []) or [])[:5]:
        pkg = a.get("package", {}) or {}
        ranges = []
        for rg in (a.get("ranges", []) or []):
            evs = [f"{e.get('introduced', '?')}->{e.get('fixed', '?')}"
                   for e in (rg.get("events", []) or [])]
            ranges.append(f"{rg.get('type', '?')}: {', '.join(evs)}")
        affected.append({"package": pkg.get("name", ""),
                         "ecosystem": pkg.get("ecosystem", ""),
                         "ranges": ranges})
    return {
        "id": v.get("id", ""),
        "title": v.get("summary", "") or v.get("id", ""),
        "cvss_score": 0,  # OSV ships CVSS vectors, not scores
        "severity": severity,
        "cve": cve,
        "cwe": ", ".join(cwes),
        "description": (v.get("details", "") or "")[:300],
        "references": refs,
        "external_references": [],
        "version_ranges": [r for a in affected for r in a["ranges"]],
        "osv_severity_raw": sev_raw,
    }


async def osv_vuln_detail(vuln_id: str) -> dict[str, Any]:
    """Fetch full OSV record for one vuln id (GHSA-*/CVE-*/OSV-*)."""
    if not vuln_id:
        return {"success": False, "error": "vuln_id required"}
    try:
        c = get_http_client()
        r = await c.get(f"{_OSV_VULN}/{urllib.parse.quote(vuln_id, safe='')}",
                        timeout=15)
        if r.status_code == 404:
            return {"success": False, "error": "Vulnerability not found"}
        if r.status_code != 200:
            return {"success": False,
                    "error": f"OSV.dev HTTP {r.status_code} {r.text[:200]}"}
        return {"success": True, "vulnerability": _extract_vuln(r.json())}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def osv_scan_purls(purls: list[str]) -> dict[str, Any]:
    """Keyless OSV.dev scan for a list of purls. Sonatype-compatible shape."""
    queries = []
    for p in purls:
        name, eco, ver = _parse_purl(p)
        if not name:
            continue
        pkg: dict[str, Any] = {"name": name}
        if eco:
            pkg["ecosystem"] = eco
        q: dict[str, Any] = {"package": pkg}
        if ver:
            q["version"] = ver
        queries.append(q)
    if not queries:
        return {"success": False, "error": "no parseable purls"}
    try:
        c = get_http_client()
        r = await c.post(_OSV_QUERYBATCH, json={"queries": queries}, timeout=30)
        if r.status_code != 200:
            return {"success": False,
                    "error": f"OSV.dev HTTP {r.status_code} {r.text[:200]}"}
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
    results = data.get("results", []) or []
    sem = asyncio.Semaphore(5)

    async def _details(idx: int) -> tuple[int, list[dict]]:
        vulns = (results[idx].get("vulns", []) or [])[:_MAX_DETAIL_FETCH] \
            if idx < len(results) else []
        out = []
        for v in vulns:
            vid = v.get("id", "")
            if not vid:
                continue
            async with sem:
                d = await osv_vuln_detail(vid)
            if d.get("success"):
                out.append(d["vulnerability"])
            else:
                out.append({"id": vid, "title": vid, "cvss_score": 0,
                            "severity": "unknown", "cve": "", "cwe": "",
                            "description": "", "references": [],
                            "external_references": [], "version_ranges": []})
        return idx, out

    details = await asyncio.gather(
        *[_details(i) for i in range(len(queries))])
    reports = []
    for i, (_, vulns) in enumerate(sorted(details)):
        reports.append({
            "coordinates": purls[i] if i < len(purls) else "",
            "reference": "",
            "vulnerability_count": len(vulns),
            "vulnerabilities": vulns,
            "patched_versions": "",
            "licenses": "",
        })
    return {"success": True, "reports": reports, "source": "osv.dev"}
