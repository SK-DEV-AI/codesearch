from __future__ import annotations

from typing import Any

import httpx

from config import GUIDE_API, OSS_API, OSS_TOKEN, _cached, _set_cache, _next_oss_key


async def scan_vulnerabilities(platform: str, name: str, version: str = "",
                               coordinates: str = "") -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    if coordinates:
        purls = [c.strip() for c in coordinates.split(",") if c.strip()]
    else:
        purl = f"pkg:{platform}/{name}"
        if version:
            purl += f"@{version}"
        purls = [purl]
    cache_key = f"oss:{','.join(purls)}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(OSS_API, json={"coordinates": purls},
                             headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": f"Sonatype Guide: {r.status_code}"}
        data = r.json()
        reports = []
        for comp in (data if isinstance(data, list) else []):
            vulnerabilities = []
            for vuln in comp.get("vulnerabilities", []):
                vulnerabilities.append({
                    "id": vuln.get("id", ""),
                    "title": vuln.get("title", ""),
                    "cvss_score": vuln.get("cvssScore", 0),
                    "severity": vuln.get("severity", "unknown"),
                    "cve": vuln.get("cve", ""),
                    "cwe": vuln.get("cwe", ""),
                    "description": (vuln.get("description", "") or "")[:300],
                    "references": vuln.get("references", []),
                    "external_references": vuln.get("externalReferences", []),
                    "version_ranges": vuln.get("versionRanges", []),
                })
            reports.append({
                "coordinates": comp.get("coordinates", ""),
                "reference": comp.get("reference", ""),
                "vulnerability_count": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "patched_versions": comp.get("patchedVersions", ""),
                "licenses": comp.get("licenses", ""),
            })
        _set_cache(cache_key, reports)
        return {"success": True, "reports": reports}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_vulnerability_detail(vuln_id: str) -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    if not vuln_id:
        return {"success": False, "error": "vuln_id required"}
    cache_key = f"vuln:{vuln_id}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{GUIDE_API}/vulnerabilities/{vuln_id}",
                            headers={"Authorization": f"Bearer {oss_key}"})
        if r.status_code == 404:
            return {"success": False, "error": "Vulnerability not found"}
        if r.status_code != 200:
            return {"success": False, "error": f"Guide API: {r.status_code}"}
        data = r.json()
        result = {
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "cvss_score": data.get("cvssScore", 0),
            "cvss_vector": data.get("cvssVector", ""),
            "severity": data.get("severity", ""),
            "cve": data.get("cve", ""),
            "cwe": data.get("cwe", ""),
            "reference": data.get("reference", ""),
            "external_references": data.get("externalReferences", []),
        }
        _set_cache(cache_key, result)
        return {"success": True, "vulnerability": result}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_component_latest_version(purl: str) -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    if not purl:
        return {"success": False, "error": "purl required"}
    cache_key = f"latest:{purl}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{GUIDE_API}/components/latest-version",
                             json={"purl": purl},
                             headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": f"Guide API: {r.status_code}"}
        data = r.json()
        result = {
            "purl": data.get("purl", ""),
            "latest_version": data.get("latestVersion", ""),
            "package_name": data.get("packageName", ""),
            "ecosystem": data.get("ecosystem", ""),
        }
        _set_cache(cache_key, result)
        return {"success": True, "result": result}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_vulnerabilities(keyword: str, limit: int = 10) -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{GUIDE_API}/security-data/packages",
                            params={"q": keyword, "limit": min(limit, 50)},
                            headers={"Authorization": f"Bearer {oss_key}"})
        if r.status_code != 200:
            return {"success": False, "error": f"Guide security-data: {r.status_code}"}
        data = r.json()
        results = []
        for pkg in (data if isinstance(data, list) else [])[:limit]:
            results.append({
                "coordinates": pkg.get("coordinates", ""),
                "package_name": pkg.get("package", {}).get("name", ""),
                "ecosystem": pkg.get("package", {}).get("ecosystem", ""),
                "latest_version": pkg.get("latestVersion", ""),
            })
        return {"success": True, "results": results, "total": len(results)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def analyze_license(purl: str) -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    if not purl:
        return {"success": False, "error": "purl required"}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{GUIDE_API}/license-analysis",
                             json={"purls": [purl]},
                             headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": f"Guide license-analysis: {r.status_code}"}
        data = r.json()
        results = []
        for comp in (data if isinstance(data, list) else []):
            results.append({
                "coordinates": comp.get("coordinates", ""),
                "licenses": comp.get("licenses", []),
                "declared_license": comp.get("declaredLicense", ""),
                "observed_license": comp.get("observedLicense", ""),
                "license_score": comp.get("licenseScore", 0),
            })
        return {"success": True, "results": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def quick_component_report(purl: str) -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    if not purl:
        return {"success": False, "error": "purl required"}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{OSS_API}/quick",
                             json={"coordinates": [purl]},
                             headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": f"Guide component-report/quick: {r.status_code}"}
        data = r.json()
        reports = []
        for comp in (data if isinstance(data, list) else []):
            vulnerabilities = []
            for vuln in comp.get("vulnerabilities", []):
                vulnerabilities.append({
                    "id": vuln.get("id", ""),
                    "title": vuln.get("title", ""),
                    "cvss_score": vuln.get("cvssScore", 0),
                    "severity": vuln.get("severity", "unknown"),
                })
            reports.append({
                "coordinates": comp.get("coordinates", ""),
                "vulnerability_count": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "licenses": comp.get("licenses", ""),
            })
        return {"success": True, "reports": reports}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
