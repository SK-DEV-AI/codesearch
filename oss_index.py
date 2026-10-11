from __future__ import annotations

from typing import Any

import httpx

from config import GUIDE_API, OSS_API, OSS_TOKEN, _cached, _set_cache, _next_oss_key, get_http_client, api_error


def _sonatype_error(response_status: int, body: str, purls: list[str]) -> str:
    """E6: per-coordinate error surfacing — name which purls failed and why."""
    msg = f"Sonatype Guide HTTP {response_status} for {len(purls)} coordinate(s): " \
        f"{', '.join(purls[:5])}" + (f" (+{len(purls) - 5} more)" if len(purls) > 5 else "")
    flat = " ".join((body or "").split())[:250]
    if flat:
        msg += f" | {flat}"
    if response_status in (401, 403):
        msg += " | quota/auth failure — OSV.dev fallback applies"
    elif response_status == 429:
        msg += " | rate limited — OSV.dev fallback applies"
    return msg


def _should_fallback_osv(status: int, body: str) -> bool:
    if status in (401, 403, 429):
        return True
    low = (body or "").lower()
    return any(s in low for s in ("credit", "quota", "exhausted", "limit"))


def cargo_platform(platform: str) -> str:
    return "cargo" if platform == "crates" else platform


async def scan_vulnerabilities(platform: str, name: str, version: str = "",
                               coordinates: str = "") -> dict:
    if coordinates:
        purls = [c.strip() for c in coordinates.split(",") if c.strip()]
    else:
        purl = f"pkg:{cargo_platform(platform)}/{name}"
        if version:
            purl += f"@{version}"
        purls = [purl]
    cache_key = f"oss:{','.join(purls)}"
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "reports": cached, "cached": True}
    # No Sonatype token at all: go straight to keyless OSV.dev.
    if not OSS_TOKEN:
        try:
            from osv import osv_scan_purls
        except ImportError:
            return {"success": False, "error": "OSS_TOKEN not configured"}
        r = await osv_scan_purls(purls)
        if r.get("success"):
            await _flag_reports_kev(r["reports"])
            await _set_cache(cache_key, r["reports"])
            r["cached"] = False
        return r
    oss_key = await _next_oss_key()
    try:
        c = get_http_client()
        r = await c.post(OSS_API, json={"coordinates": purls},
                         headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            # Quota/auth/rate-limit failure: fall back to keyless OSV.dev.
            if _should_fallback_osv(r.status_code, r.text):
                try:
                    from osv import osv_scan_purls
                except ImportError:
                    return {"success": False,
                            "error": _sonatype_error(r.status_code, r.text, purls)}
                fb = await osv_scan_purls(purls)
                if fb.get("success"):
                    await _flag_reports_kev(fb["reports"])
                    await _set_cache(cache_key, fb["reports"])
                    fb["cached"] = False
                    fb["fallback"] = "osv.dev"
                    fb["sonatype_error"] = _sonatype_error(
                        r.status_code, r.text, purls)
                return fb
            return {"success": False,
                    "error": _sonatype_error(r.status_code, r.text, purls)}
        data = r.json()
        reports = []
        for comp in (data if isinstance(data, list) else []):
            vulnerabilities = []
            for vuln in comp.get("vulnerabilities", []):
                # L12: Guide v3 has no plain `severity` field — derive from cvss
                cvss = vuln.get("cvssScore", 0) or 0
                try: cvss = float(cvss)
                except (TypeError, ValueError): cvss = 0
                sev = "critical" if cvss >= 9 else "high" if cvss >= 7 else \
                      "medium" if cvss >= 4 else "low"
                vulnerabilities.append({
                    "id": vuln.get("id", ""),
                    "title": vuln.get("title", ""),
                    "cvss_score": cvss,
                    "severity": sev,
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
        await _set_cache(cache_key, reports)
        await _flag_reports_kev(reports)
        return {"success": True, "reports": reports}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _flag_reports_kev(reports: list[dict]) -> None:
    """Annotate every vulnerability in reports with CISA KEV exploited flag."""
    try:
        from kev import flag_vulns
    except ImportError:
        return
    for rep in reports:
        vulns = rep.get("vulnerabilities", [])
        if vulns:
            try:
                await flag_vulns(vulns)
            except Exception:
                pass


async def get_vulnerability_detail(vuln_id: str) -> dict:
    if not vuln_id:
        return {"success": False, "error": "vuln_id required"}
    cache_key = f"vuln:{vuln_id}"
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "vulnerability": cached, "cached": True}
    # GHSA/CVE ids: enrich via GitHub Advisory API (existing GITHUB_TOKEN).
    if vuln_id.startswith(("GHSA-", "CVE-")):
        try:
            from ghsa import ghsa_detail
            g = await ghsa_detail(vuln_id)
            if g.get("success"):
                adv = g["advisory"]
                result = {
                    "id": adv.get("ghsa_id", vuln_id),
                    "title": adv.get("summary", ""),
                    "description": adv.get("description", ""),
                    "cvss_score": adv.get("cvss_score", 0),
                    "cvss_vector": adv.get("cvss_vector", ""),
                    "severity": adv.get("severity", ""),
                    "cve": adv.get("cve_id", ""),
                    "cwe": ", ".join(adv.get("cwes", [])),
                    "reference": adv.get("html_url", ""),
                    "external_references": adv.get("references", []),
                    "source": "ghsa",
                }
                try:
                    from kev import flag_vulns
                    await flag_vulns([result])
                except Exception:
                    pass
                await _set_cache(cache_key, result)
                return {"success": True, "vulnerability": result,
                        "source": "ghsa"}
        except ImportError:
            pass
    if not OSS_TOKEN:
        # No Sonatype token: try OSV.dev detail as last resort.
        try:
            from osv import osv_vuln_detail
            o = await osv_vuln_detail(vuln_id)
            if o.get("success"):
                await _set_cache(cache_key, o["vulnerability"])
                o["source"] = "osv.dev"
                return o
        except ImportError:
            pass
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    try:
        c = get_http_client()
        r = await c.get(f"{GUIDE_API}/vulnerabilities/{vuln_id}",
                        headers={"Authorization": f"Bearer {oss_key}"})
        if r.status_code == 404:
            return {"success": False, "error": "Vulnerability not found"}
        if r.status_code != 200:
            return {"success": False, "error": api_error("Guide API", r)}
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
        await _set_cache(cache_key, result)
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
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "result": cached, "cached": True}
    try:
        c = get_http_client()
        r = await c.post(f"{GUIDE_API}/components/latest-version",
                         json={"purl": purl},
                         headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("Guide API", r)}
        data = r.json()
        result = {
            "purl": data.get("purl", ""),
            "latest_version": data.get("latestVersion", ""),
            "package_name": data.get("packageName", ""),
            "ecosystem": data.get("ecosystem", ""),
        }
        await _set_cache(cache_key, result)
        return {"success": True, "result": result}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def analyze_license(purl: str) -> dict:
    if not OSS_TOKEN:
        return {"success": False, "error": "OSS_TOKEN not configured"}
    oss_key = await _next_oss_key()
    if not purl:
        return {"success": False, "error": "purl required"}
    try:
        c = get_http_client()
        r = await c.post(f"{GUIDE_API}/license-analysis",
                         json={"purls": [purl]},
                         headers={"Authorization": f"Bearer {oss_key}", "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("Guide license-analysis", r)}
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


