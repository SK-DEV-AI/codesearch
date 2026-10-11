"""CISA Known Exploited Vulnerabilities (KEV) exploited-flag.

Static JSON catalog, keyless, no limits. Cached 24h to a local file
(~1.8MB) so vulns scan + upgrade_review can flag actively-exploited CVEs
without a network call on every scan.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from config import get_http_client

_KEV_URL = ("https://www.cisa.gov/sites/default/files/feeds/"
            "known_exploited_vulnerabilities.json")
_KEV_FILE = os.path.expanduser("~/.cache/codesearch/kev.json")
_KEV_TTL = 24 * 3600

_lock = asyncio.Lock()
_mem: dict[str, Any] | None = None
_mem_ts: float = 0.0


async def _load_kev() -> dict[str, Any]:
    """Return {cve_id: entry}. Memory cache, then file (24h), then network."""
    global _mem, _mem_ts
    async with _lock:
        now = time.time()
        if _mem is not None and now - _mem_ts < _KEV_TTL:
            return _mem
        if os.path.exists(_KEV_FILE):
            try:
                age = now - os.path.getmtime(_KEV_FILE)
                if age < _KEV_TTL:
                    with open(_KEV_FILE, encoding="utf-8") as f:
                        data = json.load(f)
                    _mem = {v.get("cveID", ""): v
                            for v in (data.get("vulnerabilities", []) or [])
                            if v.get("cveID")}
                    _mem_ts = now
                    return _mem
            except (OSError, ValueError):
                pass
        try:
            c = get_http_client()
            r = await c.get(_KEV_URL, timeout=30)
            if r.status_code != 200:
                return _mem or {}
            data = r.json()
            _mem = {v.get("cveID", ""): v
                    for v in (data.get("vulnerabilities", []) or [])
                    if v.get("cveID")}
            _mem_ts = now
            try:
                os.makedirs(os.path.dirname(_KEV_FILE), exist_ok=True)
                tmp = _KEV_FILE + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, _KEV_FILE)
            except OSError:
                pass
            return _mem
        except (httpx.HTTPError, ValueError):
            return _mem or {}


async def is_exploited(cve_id: str) -> dict[str, Any]:
    """Check one CVE against the KEV catalog. Returns entry or exploited=false."""
    if not cve_id:
        return {"success": False, "error": "cve_id required"}
    kev = await _load_kev()
    entry = kev.get(cve_id.strip().upper())
    if not entry:
        return {"success": True, "cve_id": cve_id, "exploited": False}
    return {"success": True, "cve_id": cve_id, "exploited": True,
            "vendor": entry.get("vendorProject", ""),
            "product": entry.get("product", ""),
            "vulnerability_name": entry.get("vulnerabilityName", ""),
            "date_added": entry.get("dateAdded", ""),
            "due_date": entry.get("dueDate", ""),
            "known_ransomware": entry.get("knownRansomwareCampaignUse", "")}


async def flag_vulns(vulns: list[dict]) -> list[dict]:
    """Annotate vuln dicts (with 'cve' and/or 'id' keys) with exploited flag.

    Mutates in place and returns the list. No-op (all False) when the
    catalog cannot be fetched.
    """
    kev = await _load_kev()
    for v in vulns:
        cands = [str(v.get("cve", "") or "").upper()]
        for a in (v.get("aliases", []) or []):
            cands.append(str(a).upper())
        vid = str(v.get("id", "") or "").upper()
        if vid.startswith("CVE-"):
            cands.append(vid)
        hit = next((kev[c] for c in cands if c in kev), None)
        v["exploited"] = hit is not None
        if hit:
            v["kev"] = {"vendor": hit.get("vendorProject", ""),
                        "product": hit.get("product", ""),
                        "date_added": hit.get("dateAdded", ""),
                        "ransomware_use": hit.get("knownRansomwareCampaignUse", "")}
    return vulns
