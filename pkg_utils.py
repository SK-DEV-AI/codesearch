"""Package utilities: resolution, extraction, changelog, upgrade review."""

from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import tarfile
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from config import _cached, _set_cache, get_http_client

PACKAGE_CACHE = os.path.expanduser("~/.local/share/cortexkit/aft/pkg-cache")


def _gh_repo_url(url: str) -> str | None:
    """Normalize various GitHub URL formats to 'owner/repo'."""
    if not url:
        return None
    url = url.strip().rstrip("/").rstrip(".git")
    m = re.match(r"(?:git\+)?https?://github\.com/([^/]+/[^/]+)", url)
    if m:
        return m.group(1)
    m = re.match(r"git@github\.com:([^/]+/[^/]+)", url)
    if m:
        return m.group(1)
    m = re.match(r"github:([^/]+/[^/]+)", url)
    if m:
        return m.group(1)
    return None


async def resolve_package(registry: str, name: str, version: str = "") -> dict[str, Any]:
    """Resolve a package to metadata including repo URL.

    Returns structured: success, name, version, registry, repo_owner, repo_name,
    description, homepage, license, tarball_url, download_url.
    """
    # Normalize scoped npm names
    safe_name = urllib.parse.quote(name, safe="@")
    try:
        c = get_http_client()
        if registry == "npm" or registry == "auto":
            r = await c.get(f"https://registry.npmjs.org/{safe_name}",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
            if r.status_code == 200:
                d = r.json()
                latest = d.get("dist-tags", {}).get("latest", "")
                ver = version or latest
                repo_url = ""
                repo_raw = d.get("repository", {})
                if isinstance(repo_raw, dict):
                    repo_url = repo_raw.get("url", "")
                elif isinstance(repo_raw, str):
                    repo_url = repo_raw
                gh = _gh_repo_url(repo_url)
                # Get version-specific data
                ver_data = d.get("versions", {}).get(ver, {})
                tarball_url = ver_data.get("dist", {}).get("tarball", "")
                return {
                    "success": True, "name": name, "version": ver,
                    "registry": "npm", "description": d.get("description", ""),
                    "homepage": d.get("homepage", ""), "license": d.get("license", ""),
                    "repo_url": repo_url, "repo_owner": gh.split("/")[0] if gh else "",
                    "repo_name": gh.split("/")[1] if gh else "",
                    "tarball_url": tarball_url,
                    "download_url": f"https://www.npmjs.com/package/{name}",
                }

        if registry == "pypi" or registry == "auto":
            r = await c.get(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json",
                            headers={"User-Agent": "mcp-codesearch/1.0"})
            if r.status_code == 200:
                d = r.json()
                info = d.get("info", {})
                latest = info.get("version", "")
                ver = version or latest
                repo_url = info.get("project_urls", {}).get("Source", "")
                gh = _gh_repo_url(repo_url)
                # Find tarball URL for the version
                tarball_url = ""
                for u in d.get("urls", []) or []:
                    if u.get("version") == ver and u.get("packagetype") == "sdist":
                        tarball_url = u.get("url", "")
                        break
                if not tarball_url and d.get("urls"):
                    tarball_url = d["urls"][0].get("url", "")
                return {
                    "success": True, "name": name, "version": ver,
                    "registry": "pypi", "description": info.get("summary", ""),
                    "homepage": info.get("home_page", ""), "license": info.get("license", ""),
                    "repo_url": repo_url, "repo_owner": gh.split("/")[0] if gh else "",
                    "repo_name": gh.split("/")[1] if gh else "",
                    "tarball_url": tarball_url,
                    "download_url": f"https://pypi.org/project/{name}/",
                }

        if registry == "crates" or registry == "auto":
            r = await c.get(f"https://crates.io/api/v1/crates/{urllib.parse.quote(name)}",
                            headers={"User-Agent": "mcp-codesearch/1.0", "Accept": "application/json"})
            if r.status_code == 200:
                d = r.json()
                crate = d.get("crate", {})
                latest = crate.get("max_stable_version", crate.get("max_version", ""))
                ver = version or latest
                repo_url = crate.get("repository", "")
                gh = _gh_repo_url(repo_url)
                return {
                    "success": True, "name": name, "version": ver,
                    "registry": "crates", "description": crate.get("description", ""),
                    "homepage": crate.get("homepage", ""), "license": crate.get("license", ""),
                    "repo_url": repo_url, "repo_owner": gh.split("/")[0] if gh else "",
                    "repo_name": gh.split("/")[1] if gh else "",
                    "tarball_url": f"https://crates.io/api/v1/crates/{urllib.parse.quote(name)}/{urllib.parse.quote(ver)}/download",
                    "download_url": f"https://crates.io/crates/{name}",
                }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        pass

    return {"success": False, "error": f"package '{name}' not found in registry '{registry}'"}


async def _download_and_extract(registry: str, name: str, version: str) -> str | None:
    """Download package tarball and extract to cache dir. Returns path or None."""
    cache_dir = Path(PACKAGE_CACHE) / registry / f"{name}@{version}"
    if cache_dir.is_dir():
        return str(cache_dir)
    # Resolve tarball URL
    info = await resolve_package(registry, name, version)
    if not info.get("success") or not info.get("tarball_url"):
        return None
    tarball_url = info["tarball_url"]
    try:
        c = get_http_client()
        r = await c.get(tarball_url, headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        # npm packages pack everything under a single top-level dir (package/)
        # PyPI sdist has variant top-level dirs
        content = io.BytesIO(r.content)
        with tarfile.open(fileobj=content, mode="r:*") as tar:
            first = tar.next()
            if first is None:
                return str(cache_dir)
            strip = first.name.split("/")[0] if "/" in first.name else ""
            tar.extractall(path=cache_dir, filter="data")
        # If there's a strip prefix, move files up
        if strip:
            strip_dir = cache_dir / strip
            if strip_dir.is_dir():
                for f in strip_dir.iterdir():
                    f.rename(cache_dir / f.name)
                shutil.rmtree(str(strip_dir))
        return str(cache_dir)
    except (httpx.HTTPError, OSError, tarfile.TarError):
        return None


async def list_package_files(registry: str, name: str, version: str = "",
                             path_filter: str = "") -> dict[str, Any]:
    """List files in a package at the given registry/name/version."""
    info = await resolve_package(registry, name, version)
    if not info.get("success"):
        return info
    ver = info["version"]
    extract_dir = await _download_and_extract(registry, name, ver)
    if not extract_dir:
        return {"success": False, "error": "could not download and extract package"}
    base = Path(extract_dir)
    files = []
    for f in sorted(base.rglob("*")):
        if f.is_dir():
            continue
        rel = str(f.relative_to(base))
        if path_filter and not rel.startswith(path_filter):
            continue
        files.append({
            "path": rel,
            "size": f.stat().st_size,
            "ext": f.suffix.lstrip("."),
        })
    return {"success": True, "files": files, "total": len(files),
            "package": {"name": name, "version": ver, "registry": registry}}


async def read_package_file(registry: str, name: str, path: str,
                            version: str = "") -> dict[str, Any]:
    """Read a specific file from a downloaded package."""
    info = await resolve_package(registry, name, version)
    if not info.get("success"):
        return info
    ver = info["version"]
    extract_dir = await _download_and_extract(registry, name, ver)
    if not extract_dir:
        return {"success": False, "error": "could not download and extract package"}
    file_path = Path(extract_dir) / path
    if not file_path.exists() or not file_path.is_file():
        return {"success": False, "error": f"file '{path}' not found in package"}
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        return {"success": True, "content": content, "path": path,
                "total_lines": len(lines), "total_chars": len(content),
                "package": {"name": name, "version": ver, "registry": registry}}
    except (OSError, UnicodeDecodeError) as e:
        return {"success": False, "error": f"could not read file: {e}"}


async def get_pkg_changelog(name: str, registry: str = "auto",
                            from_version: str = "", to_version: str = "",
                            count: int = 10) -> dict[str, Any]:
    """Get changelog/release notes for a package.

    Resolves package → repo → GitHub releases API. Falls back to CHANGELOG.md.
    """
    info = await resolve_package(registry, name)
    if not info.get("success"):
        return info
    owner, repo_name = info.get("repo_owner", ""), info.get("repo_name", "")
    if not owner or not repo_name:
        return {"success": False, "error": f"no GitHub repo found for {registry}/{name}"}
    # Fetch GitHub releases
    try:
        from github_api import gh_get_releases
        r = await gh_get_releases(owner, repo_name, count=count * 2)
    except ImportError:
        return {"success": False, "error": "github_api module not available"}
    if not r.get("success"):
        return r
    releases = r.get("releases", [])
    filtered = []
    for rel in releases:
        tag = rel.get("tag", "").lstrip("v")
        if from_version and tag < from_version:
            continue
        if to_version and tag > to_version:
            continue
        filtered.append(rel)
        if len(filtered) >= count:
            break
    if filtered:
        return {"success": True, "package": {"name": name, "version": info["version"],
                "registry": registry, "repo": f"{owner}/{repo_name}"},
                "changelog": filtered}
    # Fallback: try CHANGELOG.md from raw GitHub
    try:
        c = get_http_client()
        for changelog_file in ["CHANGELOG.md", "CHANGELOG", "HISTORY.md", "RELEASE_NOTES.md"]:
            r2 = await c.get(
                f"https://raw.githubusercontent.com/{owner}/{repo_name}/master/{changelog_file}",
                headers={"User-Agent": "mcp-codesearch/1.0"})
            if r2.status_code != 200:
                r2 = await c.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/{changelog_file}",
                    headers={"User-Agent": "mcp-codesearch/1.0"})
            if r2.status_code == 200:
                return {"success": True, "package": {"name": name, "version": info["version"],
                        "registry": registry, "repo": f"{owner}/{repo_name}"},
                        "changelog": [{"file": changelog_file, "content": r2.text[:5000]}]}
    except httpx.HTTPError:
        pass
    return {"success": False, "error": "no releases or changelog found",
            "package": {"name": name, "version": info["version"],
                        "repo": f"{owner}/{repo_name}"}}


async def get_pkg_upgrade_review(name: str, registry: str = "auto",
                                 current_version: str = "",
                                 target_version: str = "") -> dict[str, Any]:
    """Compare two versions: vuln diff + changelog + deps changes."""
    if not current_version or not target_version:
        return {"success": False, "error": "current_version and target_version required"}
    info = await resolve_package(registry, name, target_version)
    if not info.get("success"):
        return info
    results = {"package": {"name": name, "registry": registry,
                           "current_version": current_version,
                           "target_version": target_version}}
    # 1. Vuln scan both versions
    try:
        from oss_index import scan_vulnerabilities
        cur_vulns, tgt_vulns = None, None
        cur_task = asyncio.create_task(scan_vulnerabilities(registry, name, current_version))
        tgt_task = asyncio.create_task(scan_vulnerabilities(registry, name, target_version))
        cur_vulns = await cur_task
        tgt_vulns = await tgt_task
        cur_ids = {v["id"] for v in cur_vulns.get("reports", [{}])[0].get("vulnerabilities", [])} if cur_vulns.get("success") else set()
        tgt_ids = {v["id"] for v in tgt_vulns.get("reports", [{}])[0].get("vulnerabilities", [])} if tgt_vulns.get("success") else set()
        results["vulnerabilities"] = {
            "current_count": len(cur_ids), "target_count": len(tgt_ids),
            "fixed": list(cur_ids - tgt_ids),
            "added": list(tgt_ids - cur_ids),
            "still_present": list(cur_ids & tgt_ids),
        }
    except (ImportError, Exception) as e:
        results["vulnerabilities"] = {"error": str(e)}

    # 2. Changelog between versions
    results["changelog"] = await get_pkg_changelog(name, registry,
                                                   from_version=current_version,
                                                   to_version=target_version, count=5)

    # 3. Deps diff
    try:
        from registries import npm_get_version, crates_get_version
        cur_deps, tgt_deps = {}, {}
        if registry == "npm" or registry == "auto":
            cur = await npm_get_version(name, current_version)
            tgt = await npm_get_version(name, target_version)
            if cur.get("success"):
                cur_deps = set(cur.get("dependencies", []))
            if tgt.get("success"):
                tgt_deps = set(tgt.get("dependencies", []))
        results["dependencies"] = {
            "current_deps": list(cur_deps),
            "target_deps": list(tgt_deps),
            "removed": list(cur_deps - tgt_deps),
            "added": list(tgt_deps - cur_deps),
        }
    except (ImportError, Exception) as e:
        results["dependencies"] = {"error": str(e)}

    return {"success": True, **results}
