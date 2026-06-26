from __future__ import annotations

import json
from typing import Any

import httpx

from config import get_http_client

DEPSDEV_API = "https://api.deps.dev/v3"


async def get_resolved_dependencies(system: str, package: str, version: str) -> dict:
    try:
        url = f"{DEPSDEV_API}/systems/{system}/packages/{package}/versions/{version}:dependencies"
        c = get_http_client()
        r = await c.get(url, timeout=15)
        if r.status_code != 200:
            return {"success": False, "error": f"deps.dev: HTTP {r.status_code}"}
        data = r.json()
        nodes: list[dict] = []
        for n in (data.get("nodes", []) or []):
            vk = n.get("versionKey", {}) or {}
            pk = n.get("packageKey", {}) or {}
            nodes.append({
                "package": vk.get("name", pk.get("name", "")),
                "version": vk.get("version", ""),
                "system": vk.get("system", pk.get("system", system)),
                "errors": n.get("errors", []),
                "relation": n.get("relation", ""),
            })
        edges: list[dict] = []
        for e in (data.get("edges", []) or []):
            edges.append({
                "fromNode": e.get("fromNode", 0),
                "toNode": e.get("toNode", 0),
                "requirement": e.get("requirement", ""),
            })
        return {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "total_dependencies": len(nodes),
        }
    except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
        return {"success": False, "error": f"deps.dev: {e}"}


async def get_package_info(system: str, package: str) -> dict:
    try:
        url = f"{DEPSDEV_API}/systems/{system}/packages/{package}"
        c = get_http_client()
        r = await c.get(url, timeout=10)
        if r.status_code != 200:
            return {"success": False, "error": f"deps.dev: HTTP {r.status_code}"}
        data = r.json()
        versions = []
        for v in (data.get("versions", []) or [])[:50]:
            vk = v.get("versionKey", {})
            versions.append({
                "version": vk.get("version", ""),
                "publishedAt": v.get("publishedAt", ""),
                "isDefault": v.get("isDefault", False),
                "isDeprecated": v.get("isDeprecated", False),
            })
        advisory_keys = data.get("advisoryKeys", [])
        licenses = data.get("licenses", [])
        return {
            "success": True,
            "package": package,
            "system": system,
            "versions": versions,
            "total_versions": len(versions),
            "advisory_keys": advisory_keys[:20],
            "licenses": licenses[:5],
        }
    except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as e:
        return {"success": False, "error": f"deps.dev: {e}"}
