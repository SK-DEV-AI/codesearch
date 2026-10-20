"""NVD (NIST National Vulnerability Database) enrichment lane.

NVD API 2.0: GET https://services.nvd.nist.gov/rest/json/cves/2.0
Keyless by default (anonymous tier). An OPTIONAL NVD_API_KEY env var raises
the rate limit; the key is read from the environment at call time and sent as
the `apiKey` query param (no key => no param, no auth failure).

Rate limits (NIST docs + live-verified 2026-10-20):
  - no key:  5 requests / rolling 30s  -> ~6.5s min spacing between calls
  - with key: 50 requests / rolling 30s -> ~0.7s min spacing between calls

Throttle design (validated 2026-10-20): the spacing guard RESERVES a slot at
enqueue time, so the queue depth is known immediately and a caller that would
wait too long is told to retry instead of blocking the dispatch. Reservation
happens outside the request lock (fast), the HTTP call happens inside it
(strictly serialized + minimum gap enforced against the previous real call).

Enrichment-only: this module NEVER raises into the dispatch path. Any failure
(404 / 403 / 429 / 5xx / network / bad JSON / throttled) comes back as
{"success": False, "error": ...} and the caller keeps its base
Sonatype/OSV/GHSA result intact.

Verified live 2026-10-20 from this egress:
  GET ?cveId=CVE-2024-5535 -> 200, vulnerabilities[0].cve.id == "CVE-2024-5535",
  metrics.cvssMetricV31[0].cvssData.baseScore == 9.1, weaknesses[0] ==
  "CWE-125", 28 references, published/lastModified present.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import httpx

from config import _cached, _set_cache, get_http_client

logger = logging.getLogger("nvd")

_NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_TIMEOUT = 20.0
_MAX_BODY = 2_000_000  # hard cap on the response body (a cveId hit is ~16KB)

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)

# NIST: 5 req / 30s anonymous, 50 req / 30s with a key. Keep headroom.
# 6.5s spacing => 4.6 calls per 30s window, safely under the 5/30s ceiling
# (exactly 6.0s would sit exactly at the documented limit).
_SPACING_ANON_S = 6.5
_SPACING_KEY_S = 0.7  # 50/30s = 0.6s -> 0.7s for headroom
# Never block a single tool call longer than this. The codesearch MCP client
# timeout is 300s, but a long block inside the dispatch is bad UX: the base
# Sonatype/OSV/GHSA answer is already available, so the NVD bonus is skipped
# instead of holding the caller. Beyond the skip, the caller is told to retry.
_MAX_WAIT_S = 25.0

_MAX_REFS = 50
_MAX_DESC = 2000

# CVSS version preference: v3.1 is the most widely consumed, then v3.0,
# then v4.0 (newer scale), then v2. All versions are reported anyway.
_CVSS_ORDER = ("cvssMetricV31", "cvssMetricV30", "cvssMetricV40", "cvssMetricV2")

# Sentinel status: the request was NOT made because the throttle queue was
# full. Distinct from 0 (network failure) so it is never negative-cached —
# a skip says nothing about the CVE and must stay retryable.
_SKIPPED = -1

_slot_lock = asyncio.Lock()   # fast: reserve a slot
_req_lock = asyncio.Lock()    # slow: serialize the actual HTTP call
_next_slot_ts = 0.0           # earliest start time of the next reserved slot
_last_call_ts = 0.0           # start time of the last REAL NVD request

# Short-lived negative cache so a rate-limited / missing CVE is not re-hit
# on every call (the config cache would hold a failure for 300s, too long).
_neg_cache: dict[str, tuple[float, dict]] = {}
_NEG_TTL_S = 60.0
_NEG_MAX = 100


def _api_key() -> str:
    """Read the optional NVD API key from the environment at call time."""
    return (os.environ.get("NVD_API_KEY") or "").strip()


def _spacing() -> float:
    return _SPACING_KEY_S if _api_key() else _SPACING_ANON_S


def _neg_get(cve_id: str) -> dict | None:
    entry = _neg_cache.get(cve_id)
    if entry and time.monotonic() - entry[0] < _NEG_TTL_S:
        return entry[1]
    return None


def _neg_put(cve_id: str, res: dict) -> None:
    if len(_neg_cache) >= _NEG_MAX:
        oldest = sorted(_neg_cache, key=lambda k: _neg_cache[k][0])[:_NEG_MAX // 2]
        for k in oldest:
            _neg_cache.pop(k, None)
    _neg_cache[cve_id] = (time.monotonic(), res)


async def _nvd_get(cve_id: str) -> tuple[int, str]:
    """GET one CVE from NVD. Returns (status_code, body_text). Never raises."""
    params: dict[str, Any] = {"cveId": cve_id}
    headers = {"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"}
    key = _api_key()
    if key:
        params["apiKey"] = key
    c = get_http_client()
    try:
        async with c.stream("GET", _NVD_API, params=params, headers=headers,
                            timeout=_TIMEOUT) as r:
            status = r.status_code
            body = b""
            async for chunk in r.aiter_bytes():
                remaining = _MAX_BODY - len(body)
                if remaining <= 0:
                    break
                # Truncate exactly at the cap: appending whole chunks would
                # overshoot it by up to one chunk.
                body += chunk[:remaining]
        return status, body.decode("utf-8", "replace")
    except httpx.HTTPError as e:
        return 0, f"{type(e).__name__}: {e}"


async def _throttled_get(cve_id: str) -> tuple[int, str]:
    """Rate-limited NVD GET. Minimum spacing between live calls, in-process.

    Two phases:
      1. Reserve a slot (fast lock). If the slot is further out than
         _MAX_WAIT_S, return _SKIPPED immediately WITHOUT consuming a slot
         and WITHOUT negative-caching — the caller retries later.
      2. Sleep until the slot, then make the request under the request lock
         so two NVD calls never overlap and the minimum gap is measured
         against the previous REAL call (a slow request still counts).
    """
    global _next_slot_ts, _last_call_ts
    async with _slot_lock:
        now = time.monotonic()
        spacing = _spacing()
        slot = max(now, _next_slot_ts)
        wait = slot - now
        if wait > _MAX_WAIT_S:
            return _SKIPPED, (f"NVD throttle: busy, would wait {wait:.1f}s "
                              f"(max {_MAX_WAIT_S:.0f}s) — skipped, retry shortly")
        # Reserve the slot so the next caller queues behind this one.
        _next_slot_ts = slot + spacing
    if wait > 0:
        await asyncio.sleep(wait)
    async with _req_lock:
        # Guard against a previous request overrunning the spacing.
        gap = _spacing() - (time.monotonic() - _last_call_ts)
        if gap > 0:
            await asyncio.sleep(gap)
        # Stamp BEFORE the request so failures/errors still consume a slot.
        _last_call_ts = time.monotonic()
        return await _nvd_get(cve_id)


def _error_for(status: int, body: str, keyed: bool) -> str:
    flat = " ".join((body or "").split())[:200]
    if status == 0:
        return f"NVD request failed: {flat}" if flat else "NVD request failed"
    if status in (403, 429):
        msg = (f"NVD rate limited (HTTP {status}) — anonymous tier is "
               "5 req/30s; set NVD_API_KEY for 50 req/30s")
        return f"{msg} | {flat}" if flat else msg
    if status == 404:
        if keyed:
            return ("NVD HTTP 404 — invalid NVD_API_KEY (NVD answers 404 with an "
                    "empty body for a rejected key)")
        return "NVD HTTP 404 — CVE id rejected or unknown"
    return f"NVD HTTP {status}" + (f" | {flat}" if flat else "")


def _extract_cwes(cve: dict) -> list[str]:
    out: list[str] = []
    for w in (cve.get("weaknesses") or []):
        for d in (w.get("description") or []):
            val = (d.get("value") or "").strip()
            if not val or val.upper().startswith("NVD-CWE"):
                continue  # NVD-CWE-noinfo / NVD-CWE-Other are placeholders
            m = _CWE_RE.search(val)
            item = m.group(0).upper() if m else val
            if item not in out:
                out.append(item)
    return out


def _cvss_entries(metrics: dict) -> list[dict]:
    """Flatten every CVSS metric NVD ships into a normalized list."""
    out: list[dict] = []
    seen_keys: set[str] = set()
    for key in _CVSS_ORDER:
        seen_keys.add(key)
        for m in (metrics.get(key) or []):
            data = m.get("cvssData") or {}
            out.append({
                "version": data.get("version", "") or key.replace("cvssMetricV", ""),
                "type": m.get("type", ""),
                "source": m.get("source", ""),
                "score": data.get("baseScore", 0) or 0,
                "vector": data.get("vectorString", ""),
                "severity": data.get("baseSeverity", ""),
                "exploitability_score": m.get("exploitabilityScore", 0) or 0,
                "impact_score": m.get("impactScore", 0) or 0,
            })
    # Any version NVD adds later that we do not know about yet.
    for key, val in (metrics or {}).items():
        if key in seen_keys or not key.startswith("cvssMetricV"):
            continue
        for m in (val or []):
            data = m.get("cvssData") or {}
            out.append({
                "version": data.get("version", "") or key.replace("cvssMetricV", ""),
                "type": m.get("type", ""),
                "source": m.get("source", ""),
                "score": data.get("baseScore", 0) or 0,
                "vector": data.get("vectorString", ""),
                "severity": data.get("baseSeverity", ""),
                "exploitability_score": m.get("exploitabilityScore", 0) or 0,
                "impact_score": m.get("impactScore", 0) or 0,
            })
    return out


def _pick_cvss(entries: list[dict]) -> dict:
    """Pick the authoritative metric: NVD 'Primary' first, else the first."""
    for e in entries:
        if e.get("type", "").lower() == "primary":
            return e
    return entries[0] if entries else {}


def _shape(cve: dict) -> dict[str, Any]:
    """Shape one NVD cve object into the enrichment payload."""
    entries = _cvss_entries(cve.get("metrics") or {})
    best = _pick_cvss(entries)
    desc = ""
    for d in (cve.get("descriptions") or []):
        if d.get("lang") == "en":
            desc = d.get("value", "") or ""
            break
    if not desc:
        for d in (cve.get("descriptions") or []):
            if d.get("value"):
                desc = d.get("value", "") or ""
                break
    raw_refs = [r for r in (cve.get("references") or []) if r.get("url")]
    # NVD itself ships the same URL under different sources (e.g. the Intel
    # advisory twice). Dedupe by URL, keep order, report both counts honestly.
    urls: list[str] = []
    for r in raw_refs:
        url = r.get("url", "")
        if url not in urls:
            urls.append(url)
    cwes = _extract_cwes(cve)
    return {
        "id": cve.get("id", ""),
        "source_identifier": cve.get("sourceIdentifier", ""),
        "published": cve.get("published", ""),
        "last_modified": cve.get("lastModified", ""),
        "vuln_status": cve.get("vulnStatus", ""),
        "description": desc[:_MAX_DESC],
        "cvss_score": best.get("score", 0) or 0,
        "cvss_vector": best.get("vector", ""),
        "cvss_version": best.get("version", ""),
        "cvss_severity": best.get("severity", ""),
        "cvss_source": best.get("source", ""),
        "cvss_metrics": entries,
        "cwes": cwes,
        "cwe": ", ".join(cwes),
        "reference_count": len(urls),
        "raw_reference_count": len(raw_refs),
        "references": urls[:_MAX_REFS],
    }


async def nvd_enrich(cve_id: str) -> dict[str, Any]:
    """Fetch NVD enrichment for one CVE id. Never raises.

    Returns {"success": True, "cve": {...}} on success, or
    {"success": False, "error": ...} on any failure.
    """
    try:
        cid = (cve_id or "").strip()
        if not cid:
            return {"success": False, "error": "cve_id required"}
        if not _CVE_RE.match(cid):
            # Guards the "empty cveId" case, which makes NVD answer with the
            # whole 410k-CVE catalogue (~4MB) instead of one record.
            return {"success": False, "error": f"not a CVE id: {cid}"}
        cid = cid.upper()
        cache_key = f"nvd:{cid}"
        cached = await _cached(cache_key)
        if cached is not None:
            return {"success": True, "cve": cached, "cached": True, "source": "nvd"}
        neg = _neg_get(cid)
        if neg is not None:
            return {**neg, "cached": True}
        status, body = await _throttled_get(cid)
        if status == _SKIPPED:
            # Throttled, not an NVD verdict: never negative-cache, so the
            # caller can retry as soon as a slot frees up.
            return {"success": False, "error": body}
        if status != 200:
            res = {"success": False,
                   "error": _error_for(status, body, bool(_api_key()))}
            # Cache only failures that are a verdict ABOUT THIS CVE.
            #  - status 0 (network) is transient: never cache, retry recovers.
            #  - 403/429 is a shared rate-limit, not a per-CVE verdict: never
            #    cache (it would punish every other CVE for one burst).
            #  - a 404 while a key is set is an INVALID KEY, not a CVE verdict:
            #    never cache (fixing the key must recover immediately).
            #  - a 404 with no key IS a per-CVE verdict (unknown CVE): cache.
            #  - 5xx is a server verdict: cache (short TTL).
            cacheable = status >= 500 or (status == 404 and not _api_key())
            if cacheable:
                _neg_put(cid, res)
            return res
        try:
            data = json.loads(body)
        except ValueError as e:
            res = {"success": False, "error": f"NVD invalid JSON: {e}"}
            _neg_put(cid, res)
            return res
        vulns = data.get("vulnerabilities") or []
        if not vulns:
            res = {"success": False, "error": f"{cid} not found in NVD"}
            _neg_put(cid, res)
            return res
        shaped = _shape(vulns[0].get("cve") or {})
        await _set_cache(cache_key, shaped)
        return {"success": True, "cve": shaped, "source": "nvd"}
    except Exception as e:  # never raise into the dispatch path
        logger.debug("NVD enrichment failed for %r: %s", cve_id, e, exc_info=True)
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def _cve_id_for(result: dict, vuln_id: str = "") -> str:
    """Find the CVE id to enrich: the id itself, or the CVE attached to a
    GHSA-/Sonatype-shaped record."""
    for cand in (vuln_id, (result.get("vulnerability") or {}).get("cve", ""),
                 (result.get("vulnerability") or {}).get("id", "")):
        c = str(cand or "").strip()
        if _CVE_RE.match(c):
            return c.upper()
    return ""


def _fill_gaps(vuln: dict, data: dict) -> None:
    """Fill ONLY empty fields of the base vulnerability from NVD. Never
    overwrites a value the base source already provided."""
    if not isinstance(vuln, dict) or not data:
        return
    if not vuln.get("cvss_score") and data.get("cvss_score"):
        vuln["cvss_score"] = data["cvss_score"]
    if not vuln.get("cvss_vector") and data.get("cvss_vector"):
        vuln["cvss_vector"] = data["cvss_vector"]
    if not vuln.get("cwe") and data.get("cwe"):
        vuln["cwe"] = data["cwe"]
    if not vuln.get("description") and data.get("description"):
        vuln["description"] = data["description"]
    if not vuln.get("published") and data.get("published"):
        vuln["published"] = data["published"]
    if not vuln.get("last_modified") and data.get("last_modified"):
        vuln["last_modified"] = data["last_modified"]


async def attach_nvd(result: dict, vuln_id: str = "") -> dict:
    """Attach NVD enrichment to a vulns-detail result. Never raises.

    Adds an `nvd` section and fills empty fields of `vulnerability` from
    NVD. Every pre-existing field of the base result is kept as-is.
    """
    try:
        if not isinstance(result, dict):
            return result
        cid = _cve_id_for(result, vuln_id)
        if not cid:
            return result  # nothing CVE-shaped to enrich
        enrich = await nvd_enrich(cid)
        if not enrich.get("success"):
            result["nvd"] = {"success": False, "cve_id": cid,
                             "error": enrich.get("error", "NVD enrichment failed")}
            return result
        data = enrich.get("cve") or {}
        section = {"success": True, "cve_id": cid}
        section.update(data)
        if enrich.get("cached"):
            section["cached"] = True
        result["nvd"] = section
        _fill_gaps(result.get("vulnerability") or {}, data)
        return result
    except Exception:
        logger.debug("NVD attach failed for %r: %s", vuln_id, exc_info=True)
        return result
