from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from config import CRATES_SEARCH, NPM_SEARCH, REGISTRIES, get_http_client, api_error

NPM_DOWNLOADS = "https://api.npmjs.org/downloads/point/last-month"
PYPI_STATS = "https://pypistats.org/api/packages"
CRATES_API = "https://crates.io/api/v1"


async def search_package(name: str, registry: str = "auto", type: str = "") -> dict:
    if type == "npm_dist_tags":
        return await _npm_dist_tags(name)
    if type == "npm_versions":
        return await get_npm_versions(name)
    if type == "npm_time":
        return await get_npm_time(name)
    if type == "crates_downloads":
        return await _crates_downloads(name)
    if type == "crates_reverse_deps":
        return await _crates_reverse_deps(name)
    if type == "crates_owners":
        return await _crates_owners(name)
    if type == "crates_categories":
        return await _crates_categories()
    if type == "crates_keywords":
        return await _crates_keywords()
    if type == "crates_versions":
        return await get_crates_versions(name)
    if type == "npm_search":
        return await npm_search(name, count=10)
    if type == "crates_search":
        return await crates_search(name, count=10)
    registries_to_try = []
    if registry == "auto":
        registries_to_try = list(REGISTRIES.items())
    elif registry in REGISTRIES:
        registries_to_try = [(registry, REGISTRIES[registry])]
    results = {}
    qname = urllib.parse.quote(name, safe="@/")
    for reg_name, url_template in registries_to_try:
        try:
            url = url_template.format(name=qname)
            headers = {"User-Agent": "mcp-codesearch/1.0"}
            if reg_name == "crates":
                headers["Accept"] = "application/json"
            c = get_http_client()
            r = await c.get(url, headers=headers)
            if r.status_code != 200:
                continue
            data = r.json()
            if reg_name == "npm":
                results[reg_name] = {
                    "name": data.get("name", name),
                    "version": data.get("dist-tags", {}).get("latest", ""),
                    "description": data.get("description", ""),
                    "keywords": data.get("keywords", []),
                    "homepage": data.get("homepage", ""),
                    "repository": data.get("repository", {}).get("url", "") if isinstance(data.get("repository"), dict) else data.get("repository", ""),
                    "license": data.get("license", ""),
                    "dependencies": list(data.get("dependencies", {}).keys())[:10],
                    "dev_dependencies": list(data.get("devDependencies", {}).keys())[:5],
                }
                try:
                    c = get_http_client()
                    dls_r = await c.get(f"{NPM_DOWNLOADS}/{urllib.parse.quote(name)}",
                                        headers={"User-Agent": "mcp-codesearch/1.0"})
                    if dls_r.status_code == 200:
                        dls_data = dls_r.json()
                        results[reg_name]["downloads_last_month"] = dls_data.get("downloads", 0)
                except (httpx.HTTPError, ValueError):
                    pass
            elif reg_name == "pypi":
                info = data.get("info", {})
                results[reg_name] = {
                    "name": info.get("name", name),
                    "version": info.get("version", ""),
                    "description": info.get("summary", ""),
                    "homepage": info.get("home_page", ""),
                    "repository": info.get("project_urls", {}).get("Source", "") if isinstance(info.get("project_urls"), dict) else "",
                    "license": info.get("license", "") or (info.get("classifiers", [""])[0] if info.get("classifiers") else ""),
                    "requires_dist": info.get("requires_dist", [])[:10],
                    "python_versions": info.get("requires_python", ""),
                    "classifiers": info.get("classifiers", [])[:10],
                    "project_urls": info.get("project_urls", {}),
                    "yanked": info.get("yanked", False),
                    "yanked_reason": info.get("yanked_reason", ""),
                }
                try:
                    c = get_http_client()
                    dls_r = await c.get(f"{PYPI_STATS}/{urllib.parse.quote(name)}/recent",
                                        headers={"User-Agent": "mcp-codesearch/1.0"})
                    if dls_r.status_code == 200:
                        dls_data = dls_r.json()
                        results[reg_name]["downloads_last_month"] = dls_data.get("data", {}).get("last_month", 0)
                except (httpx.HTTPError, ValueError):
                    pass
            elif reg_name == "crates":
                crate = data.get("crate", {})
                results[reg_name] = {
                    "name": crate.get("name", name),
                    "version": crate.get("max_stable_version", crate.get("max_version", "")),
                    "description": crate.get("description", ""),
                    "homepage": crate.get("homepage", ""),
                    "repository": crate.get("repository", ""),
                    "license": crate.get("license", ""),
                    "keywords": crate.get("keywords", []),
                    "categories": crate.get("categories", []),
                    "downloads": crate.get("downloads", 0),
                    "recent_downloads": crate.get("recent_downloads", 0),
                    "max_version": crate.get("max_version", ""),
                }
        except (httpx.HTTPError, ValueError):
            continue
    if not results:
        return {"success": False, "error": f"package '{name}' not found in any registry"}
    return {"success": True, "registries_checked": list(results.keys()), "results": results}


async def npm_search(query: str, count: int = 10, type: str = "search", name: str = "") -> dict:
    if type == "dist_tags":
        return await _npm_dist_tags(name)
    try:
        params: dict[str, Any] = {"text": query, "size": min(count, 250)}
        c = get_http_client()
        r = await c.get(NPM_SEARCH, params=params, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("npm search", r)}
        data = r.json()
        results = []
        for obj in data.get("objects", [])[:count]:
            pkg = obj.get("package", {})
            score = obj.get("score", {})
            detail = score.get("detail", {})
            results.append({
                "name": pkg.get("name", ""),
                "version": pkg.get("version", ""),
                "description": pkg.get("description", ""),
                "keywords": pkg.get("keywords", []),
                "date": pkg.get("date", ""),
                "links": pkg.get("links", {}),
                "author": pkg.get("author", {}).get("name", "") if isinstance(pkg.get("author"), dict) else str(pkg.get("author", "")),
                "score": {
                    "final": round(score.get("final", 0), 3),
                    "quality": round(detail.get("quality", 0), 3),
                    "popularity": round(detail.get("popularity", 0), 3),
                    "maintenance": round(detail.get("maintenance", 0), 3),
                },
            })
        return {"success": True, "results": results, "total": data.get("total", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _npm_dist_tags(name: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"https://registry.npmjs.org/{urllib.parse.quote(name)}/dist-tags",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("npm dist-tags", r)}
        return {"success": True, "name": name, "dist_tags": r.json()}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_npm_versions(name: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"https://registry.npmjs.org/{urllib.parse.quote(name)}",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("npm versions", r)}
        data = r.json()
        versions = sorted(data.get("versions", {}).keys())
        return {"success": True, "name": name, "versions": versions, "total": len(versions)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_npm_time(name: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"https://registry.npmjs.org/{urllib.parse.quote(name)}",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("npm time", r)}
        data = r.json()
        time_data = data.get("time", {})
        return {"success": True, "name": name, "time": time_data}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_crates_versions(name: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}/versions",
                        headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates versions", r)}
        data = r.json()
        versions = []
        for v in (data.get("versions", []) or [])[:50]:
            versions.append({
                "num": v.get("num", ""),
                "created_at": v.get("created_at", ""),
                "downloads": v.get("downloads", 0),
                "yanked": v.get("yanked", False),
            })
        return {"success": True, "name": name, "versions": versions, "total": len(versions)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_pypi_versions(name: str) -> dict:
    """List all available versions of a PyPI package."""
    try:
        c = get_http_client()
        r = await c.get(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("PyPI", r)}
        data = r.json()
        releases = data.get("releases", {}) or {}
        versions = []
        for ver, files in releases.items():
            upload_times = [f.get("upload_time_iso_8601", "") for f in (files or []) if f.get("upload_time_iso_8601")]
            has_sdist = any(f.get("packagetype") == "sdist" for f in (files or []))
            has_wheel = any(f.get("packagetype") == "bdist_wheel" for f in (files or []))
            versions.append({
                "version": ver,
                "upload_time": max(upload_times) if upload_times else "",
                "has_wheel": has_wheel,
                "has_sdist": has_sdist,
                "file_count": len(files or []),
            })
        info = data.get("info", {})
        try:
            from packaging.version import InvalidVersion, Version
            def _vkey(v: str):
                try:
                    return (0, Version(v))
                except InvalidVersion:
                    return (1, ())  # unparseable versions sort last, stable
        except ImportError:
            def _vkey(v: str):
                return (0, v)
        return {"success": True, "name": info.get("name", name),
                "summary": (info.get("summary") or "")[:300],
                "total_versions": len(versions),
                "versions": sorted(versions, key=lambda v: _vkey(v["version"]), reverse=True)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def get_pypi_version(name: str, version: str) -> dict:
    try:
        c = get_http_client()
        if version.strip().lower() in ("", "latest", "auto", "*"):
            # models ask for "latest" — resolve via the metadata endpoint
            meta = await c.get(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json",
                               headers={"User-Agent": "mcp-codesearch/1.0"})
            if meta.status_code != 200:
                return {"success": False, "error": api_error("PyPI", meta)}
            version = (meta.json().get("info") or {}).get("version", "") or version
        r = await c.get(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/json",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("PyPI version", r)}
        data = r.json()
        info = data.get("info", {})
        urls = data.get("urls", [])
        files = []
        for url_info in (urls or []):
            files.append({
                "filename": url_info.get("filename", ""),
                "url": url_info.get("url", ""),
                "size": url_info.get("size", 0),
                "python_version": url_info.get("python_version", ""),
                "packagetype": url_info.get("packagetype", ""),
                "requires_python": url_info.get("requires_python", ""),
                "sha256": (url_info.get("digests", {}) or {}).get("sha256", ""),
                "upload_time": url_info.get("upload_time_iso_8601", ""),
            })
        yanked = any(f.get("yanked", False) for f in files)
        yanked_reason = next((f.get("yanked_reason", "") for f in files if f.get("yanked")), "")
        return {"success": True, "name": info.get("name", name), "version": info.get("version", version),
                "summary": info.get("summary", ""),
                "license": info.get("license", ""),
                "requires_python": info.get("requires_python", ""),
                "yanked": yanked, "yanked_reason": yanked_reason,
                "files": files}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def crates_search(query: str, count: int = 10,
                        sort: str = "", keywords: str = "", categories: str = "",
                        type: str = "search", name: str = "") -> dict:
    if type == "downloads":
        return await _crates_downloads(name, count)
    if type == "reverse_deps":
        return await _crates_reverse_deps(name, count)
    if type == "owners":
        return await _crates_owners(name)
    if type == "categories":
        return await _crates_categories()
    if type == "keywords":
        return await _crates_keywords()
    if type == "versions":
        return await get_crates_versions(name)
    try:
        params: dict[str, Any] = {"per_page": min(count, 100)}
        if query:
            params["q"] = query
        if sort:
            params["sort"] = sort
        if keywords:
            if params.get("q"):
                params["q"] += f" keywords:{keywords}"
            else:
                params["q"] = f"keywords:{keywords}"
        if categories:
            params["category"] = categories
        c = get_http_client()
        r = await c.get(CRATES_SEARCH, params=params,
                        headers={"User-Agent": "mcp-codesearch/1.0 (mcp-codesearch)", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates.io search", r)}
        data = r.json()
        results = []
        for crate in data.get("crates", [])[:count]:
            results.append({
                "name": crate.get("name", ""),
                "version": crate.get("max_version", ""),
                "description": crate.get("description", ""),
                "downloads": crate.get("downloads", 0),
                "recent_downloads": crate.get("recent_downloads", 0),
                "homepage": crate.get("homepage", ""),
                "repository": crate.get("repository", ""),
                "license": crate.get("license", ""),
                "keywords": crate.get("keywords", []),
                "categories": crate.get("categories", []),
            })
        return {"success": True, "results": results, "total": data.get("meta", {}).get("total", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _crates_downloads(name: str, count: int = 10) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}/downloads",
                        headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates.io downloads", r)}
        data = r.json()
        return {"success": True, "name": name, "data": (data.get("data", []) or [])[:count]}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _crates_reverse_deps(name: str, count: int = 10) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}/reverse_dependencies",
                        params={"per_page": min(count, 100)},
                        headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates.io reverse_deps", r)}
        data = r.json()
        results = []
        for dep in (data.get("reverse_dependencies", []) or [])[:count]:
            results.append({
                "name": dep.get("crate_id", ""),
                "req": dep.get("req", ""),
                "kind": dep.get("kind", ""),
                "version_id": dep.get("version_id", ""),
            })
        return {"success": True, "name": name, "results": results,
                "total": data.get("meta", {}).get("total", len(results))}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _crates_owners(name: str) -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}/owners",
                        headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates.io owners", r)}
        data = r.json()
        results = []
        for owner in (data.get("users", []) or []):
            results.append({
                "login": owner.get("login", ""),
                "name": owner.get("name", ""),
                "avatar": owner.get("avatar", ""),
                "kind": owner.get("kind", ""),
            })
        return {"success": True, "name": name, "owners": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _crates_categories() -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/categories",
                        headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates.io categories", r)}
        data = r.json()
        results = []
        for cat in (data.get("categories", []) or []):
            results.append({
                "id": cat.get("id", ""),
                "slug": cat.get("slug", ""),
                "name": cat.get("name", ""),
                "description": cat.get("description", ""),
                "crates_count": cat.get("crates_cnt", 0),
            })
        return {"success": True, "results": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def npm_get_version(name: str, version: str) -> dict:
    """Get metadata for a specific version of an npm package."""
    try:
        c = get_http_client()
        if version.strip().lower() in ("", "latest", "auto", "*"):
            meta = await c.get(f"https://registry.npmjs.org/{urllib.parse.quote(name)}",
                               headers={"User-Agent": "mcp-codesearch/1.0"})
            if meta.status_code != 200:
                return {"success": False, "error": api_error("npm", meta)}
            version = (meta.json().get("dist-tags") or {}).get("latest", "") or version
        r = await c.get(f"https://registry.npmjs.org/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}",
                        headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200: return {"success": False, "error": api_error("npm version", r)}
        d = r.json()
        return {"success": True, "name": d.get("name",""), "version": d.get("version",""),
            "description": d.get("description",""), "license": d.get("license",""),
            "homepage": d.get("homepage",""), "repository": d.get("repository",{}).get("url","") if isinstance(d.get("repository"),dict) else d.get("repository",""),
            "dependencies": list(d.get("dependencies",{}).keys())[:15],
            "devDependencies": list(d.get("devDependencies",{}).keys())[:10]}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def crates_get_version(name: str, version: str) -> dict:
    """Get metadata for a specific version of a crate."""
    try:
        c = get_http_client()
        if version.strip().lower() in ("", "latest", "auto", "*"):
            meta = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}",
                headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
            if meta.status_code != 200:
                return {"success": False, "error": api_error("crates.io", meta)}
            version = (meta.json().get("crate") or {}).get("max_version", "") or version
        r = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}",
            headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200: return {"success": False, "error": api_error("crates.io", r)}
        found = r.json().get("version", {})
        if not found: return {"success": False, "error": f"version {version} not found"}
        return {"success": True, "name": name, "version": version,
            "license": found.get("license",""), "downloads": found.get("downloads",0),
            "published": found.get("created_at",""), "updated": found.get("updated_at",""),
            "yanked": found.get("yanked",False), "rust_version": found.get("rust_version","")}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def crates_get_readme(name: str, version: str) -> dict:
    """Get README for a specific version of a crate."""
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/crates/{urllib.parse.quote(name)}/versions/{urllib.parse.quote(version)}/readme",
            headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "text/html"})
        if r.status_code == 404: return {"success": False, "error": "no readme", "name": name}
        if r.status_code != 200: return {"success": False, "error": api_error("crates.io readme", r)}
        return {"success": True, "readme": r.text[:10000], "name": name}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def crates_get_summary() -> dict:
    """Get crates.io summary statistics."""
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/summary",
            headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200: return {"success": False, "error": api_error("crates.io summary", r)}
        d = r.json()
        top_pkgs = []
        for crate in (d.get("most_downloaded",[]) or [])[:10]:
            top_pkgs.append({"name": crate.get("id",""), "downloads": crate.get("downloads",0)})
        return {"success": True, "num_crates": d.get("num_crates",0), "num_downloads": d.get("num_downloads",0),
            "most_downloaded": top_pkgs}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def _crates_keywords() -> dict:
    try:
        c = get_http_client()
        r = await c.get(f"{CRATES_API}/keywords",
                        headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("crates.io keywords", r)}
        data = r.json()
        results = []
        for kw in (data.get("keywords", []) or []):
            results.append({
                "id": kw.get("id", ""),
                "keyword": kw.get("keyword", ""),
                "crates_count": kw.get("crates_cnt", 0),
            })
        return {"success": True, "results": results}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
