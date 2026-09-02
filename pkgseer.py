"""Direct HTTP wrappers for PkgSeer GraphQL (pkgseer.dev), jsDelivr API + CDN.

PkgSeer: per-package code navigation (search, files, grep, deps) across
npm/PyPI/crates. Requires GITHITS_API_TOKEN.
jsDelivr: free CDN for npm package file listing + raw content reading.
No auth needed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from config import GITHITS_API_TOKEN, api_error, get_http_client
from security import safe_fetch

_PKGSEER_URL = "https://pkgseer.dev"
_JSDELIVR_API = "https://data.jsdelivr.com/v1"
_TIMEOUT = 90

_HEADERS = {"Authorization": f"Bearer {GITHITS_API_TOKEN}", "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# PkgSeer: helpers
# ---------------------------------------------------------------------------

if not GITHITS_API_TOKEN:
    _TOKEN_ERROR = {"success": False, "error": "GITHITS_API_TOKEN not configured"}
else:
    _TOKEN_ERROR = None


def _check_token() -> dict[str, Any] | None:
    if _TOKEN_ERROR:
        return _TOKEN_ERROR
    return None


async def _post_json(url: str, body: dict[str, Any], timeout: int = _TIMEOUT) -> dict[str, Any]:
    """POST JSON, return parsed JSON on 2xx or an error dict."""
    try:
        resp = await get_http_client().post(url, headers=_HEADERS, json=body, timeout=timeout)
    except httpx.TimeoutException:
        return {"success": False, "error": f"PkgSeer request timed out ({timeout}s)"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"PkgSeer request failed: {e}"}
    if resp.status_code == 401:
        return {"success": False, "error": "PkgSeer API rejected the token (401)"}
    if resp.status_code >= 400:
        return {"success": False, "error": f"PkgSeer API error ({resp.status_code}): {resp.text[:300]}"}
    try:
        return {"success": True, "data": resp.json()}
    except json.JSONDecodeError:
        return {"success": True, "text": resp.text}


async def _pkgseer_graphql(
    query: str,
    variables: dict[str, Any],
    timeout: int = _TIMEOUT,
) -> dict[str, Any]:
    """POST a GraphQL query to the PkgSeer backend."""
    result = await _post_json(f"{_PKGSEER_URL}/api/graphql", {"query": query, "variables": variables}, timeout)
    if not result.get("success"):
        return result
    data = result.get("data", {})
    if isinstance(data, dict) and data.get("errors"):
        msgs = [e.get("message", str(e)) for e in data["errors"]]
        return {"success": False, "error": "; ".join(msgs)}
    return result


def _parse_spec(spec: str) -> dict[str, str]:
    """Parse 'registry:name[@version]' into GraphQL-compatible dict.
    Provides both 'name' (deps/search queries) and 'packageName' (code nav queries).
    Omits 'version' when not specified — empty string breaks PkgSeer.
    """
    registry = "npm"
    package_name = spec
    version = ""
    if ":" in package_name:
        registry, package_name = package_name.split(":", 1)
    if "@" in package_name and not package_name.startswith("@"):
        package_name, version = package_name.rsplit("@", 1)
    registry_map = {"npm": "NPM", "pypi": "PYPI", "crates": "CRATESIO"}
    reg = registry_map.get(registry.lower())
    if reg is None:
        return {"error": f"PkgSeer indexes npm/pypi/crates only — '{registry}:' specs are not supported. For GitHub repos use the searchcode tools (repository=URL) or codesearch_analyze."}
    registry = reg
    result = {"registry": registry, "name": package_name, "packageName": package_name}
    if version:
        result["version"] = version
    return result


# ---------------------------------------------------------------------------
# PkgSeer GraphQL — pkgseer.dev/api/graphql
# ---------------------------------------------------------------------------


async def search(query: str, target: str = "", source: str = "",
                 lang: str = "", limit: int = 10) -> dict[str, Any]:
    """Unified search across indexed packages/repos."""
    err = _check_token()
    if err:
        return err
    targets = [{"name": query}]
    if target:
        parts = _parse_spec(target)
        if "error" in parts:
            return {"success": False, "error": parts["error"]}
        targets = [{"name": parts["name"], "registry": parts["registry"]}]
    variables = {
        "targets": targets,
        "query": query,
        "sources": [source.upper()] if source else None,
        "limit": min(limit, 100),
        "waitTimeoutMs": 60000,
    }
    q = """query UnifiedSearch($targets:[SearchPackageInput!]!$query:String!$sources:[DiscoverySearchSource!]$limit:Int$waitTimeoutMs:Int){search(targets:$targets query:$query sources:$sources limit:$limit waitTimeoutMs:$waitTimeoutMs){completed searchRef result{query sources results{id resultType targetLabel title summary score locator{registry packageName version repoUrl filePath startLine endLine symbolRef kind category language}}page{offset limit returned hasMore}}progress{searchRef status elapsedMs}}}"""
    result = await _pkgseer_graphql(q, variables)
    if not result.get("success"):
        return result
    data = result.get("data", {})
    search_data = (data or {}).get("data", {}).get("search", {})
    results_list = (search_data.get("result") or {}).get("results", [])
    return {"success": True, "results": results_list}


LIST_FILES_QUERY = """query ListRepoFiles($registry:Registry $packageName:String $repoUrl:String $gitRef:String $version:String $pathPrefix:String $extensions:[String!] $limit:Int $waitTimeoutMs:Int){listRepoFiles(registry:$registry packageName:$packageName repoUrl:$repoUrl gitRef:$gitRef version:$version pathPrefix:$pathPrefix extensions:$extensions limit:$limit waitTimeoutMs:$waitTimeoutMs){files{path name language fileType byteSize}total hasMore indexedVersion}}"""


async def code_files(spec: str, path_prefix: str = "") -> dict[str, Any]:
    """List files in an indexed dependency."""
    err = _check_token()
    if err:
        return err
    pkg = _parse_spec(spec)
    if "error" in pkg:
        return {"success": False, "error": pkg["error"]}
    variables = {**pkg, "pathPrefix": path_prefix or None, "limit": 200, "waitTimeoutMs": 30000}
    result = await _pkgseer_graphql(LIST_FILES_QUERY, variables, timeout=30)
    if not result.get("success"):
        return result
    data = result.get("data", {}).get("data", {}).get("listRepoFiles", {}) or {}
    return {"success": True, "files": data.get("files", []), "total": data.get("total", 0)}


FETCH_CODE_CONTEXT_QUERY = """query FetchCodeContext($registry:Registry $packageName:String $repoUrl:String $gitRef:String $version:String $filePath:String! $startLine:Int $endLine:Int $waitTimeoutMs:Int){fetchCodeContext(registry:$registry packageName:$packageName repoUrl:$repoUrl gitRef:$gitRef version:$version filePath:$filePath startLine:$startLine endLine:$endLine waitTimeoutMs:$waitTimeoutMs){content filePath language totalLines startLine endLine isBinary}}"""


async def code_read(spec: str, path: str) -> dict[str, Any]:
    """Read a file from an indexed dependency."""
    err = _check_token()
    if err:
        return err
    pkg = _parse_spec(spec)
    if "error" in pkg:
        return {"success": False, "error": pkg["error"]}
    variables = {**pkg, "filePath": path, "startLine": None, "endLine": None, "waitTimeoutMs": 30000}
    result = await _pkgseer_graphql(FETCH_CODE_CONTEXT_QUERY, variables, timeout=30)
    if not result.get("success"):
        return result
    data = result.get("data", {}).get("data", {}).get("fetchCodeContext", {}) or {}
    return {"success": True, "content": data.get("content", ""), "filePath": data.get("filePath", path)}


GREP_REPO_QUERY = """query GrepRepo($registry:Registry $packageName:String $repoUrl:String $gitRef:String $version:String $waitTimeoutMs:Int $pattern:String! $patternType:GrepPatternType $caseSensitive:Boolean $pathSelectors:[GrepPathSelectorInput!] $extensions:[String!] $allowUnscoped:Boolean $contextLinesBefore:Int $contextLinesAfter:Int $maxMatches:Int $maxMatchesPerFile:Int){grepRepo(registry:$registry packageName:$packageName repoUrl:$repoUrl gitRef:$gitRef version:$version waitTimeoutMs:$waitTimeoutMs pattern:$pattern patternType:$patternType caseSensitive:$caseSensitive pathSelectors:$pathSelectors extensions:$extensions allowUnscoped:$allowUnscoped contextLinesBefore:$contextLinesBefore contextLinesAfter:$contextLinesAfter maxMatches:$maxMatches maxMatchesPerFile:$maxMatchesPerFile){matches{filePath line lineContent contextBefore contextAfter}nextCursor hasMore totalMatches uniqueFilesMatched}}"""


async def code_grep(spec: str, pattern: str, path_prefix: str = "") -> dict[str, Any]:
    """Grep through indexed dependency source for a pattern."""
    err = _check_token()
    if err:
        return err
    pkg = _parse_spec(spec)
    if "error" in pkg:
        return {"success": False, "error": pkg["error"]}
    path_selectors = [{"kind": "PREFIX", "value": path_prefix}] if path_prefix else None
    variables = {
        **pkg, "pattern": pattern, "patternType": "LITERAL",
        "caseSensitive": False, "pathSelectors": path_selectors,
        "allowUnscoped": not bool(path_prefix),
        "maxMatches": 50, "contextLinesBefore": 2, "contextLinesAfter": 2,
        "waitTimeoutMs": 60000,
    }
    result = await _pkgseer_graphql(GREP_REPO_QUERY, variables, timeout=60)
    if not result.get("success"):
        return result
    data = result.get("data", {}).get("data", {}).get("grepRepo", {}) or {}
    return {"success": True, "matches": data.get("matches", [])}


PKG_DEPS_QUERY = """query PackageDependencies($registry:Registry! $name:String! $version:String $includeTransitive:Boolean $maxDepth:Int){packageDependencies(registry:$registry name:$name version:$version includeTransitive:$includeTransitive maxDepth:$maxDepth){package{name registry version}dependencies{direct{name versionConstraint type}transitive{totalEdges uniquePackagesCount dependencyConflicts{packageName requiredVersions conflictingEdges{fromIndex toIndex versionConstraint}}circularDependencyCycles{cycleStart circularPath}}}}}"""


async def pkg_deps(spec: str) -> dict[str, Any]:
    """Analyze transitive dependencies with conflict detection."""
    err = _check_token()
    if err:
        return err
    pkg = _parse_spec(spec)
    if "error" in pkg:
        return {"success": False, "error": pkg["error"]}
    variables = {**pkg, "includeTransitive": True, "maxDepth": 5}
    result = await _pkgseer_graphql(PKG_DEPS_QUERY, variables, timeout=60)
    if not result.get("success"):
        return result
    data = result.get("data", {}).get("data", {}).get("packageDependencies", {}) or {}
    deps = data.get("dependencies", {})
    return {"success": True, "dependencies": deps, "package": data.get("package", {})}


# ---------------------------------------------------------------------------
# jsDelivr — free npm file listing + content via CDN
# ---------------------------------------------------------------------------


async def jsdelivr_list_files(spec: str) -> dict[str, Any]:
    """List all files in an npm package via jsDelivr Data API (flat structure).
    spec format: 'npm:express' or 'npm:express@5.2.1'. npm registry only.
    3-5x faster than PkgSeer for npm, no rate limits.
    """
    parts = spec.split(":")
    pkg_name = parts[-1] if len(parts) > 1 else spec
    version = ""
    if "@" in pkg_name and not pkg_name.startswith("@"):
        pkg_name, version = pkg_name.rsplit("@", 1)
    c = get_http_client()
    # The /packages endpoint only returns files when a version is pinned.
    # Resolve latest from the tags listing first, then fetch files flat.
    if not version:
        meta = await c.get(f"{_JSDELIVR_API}/packages/npm/{pkg_name}")
        if meta.status_code != 200:
            return {"success": False, "error": api_error("jsDelivr", meta)}
        try:
            version = str((meta.json().get("tags") or {}).get("latest", ""))
        except ValueError:
            return {"success": False, "error": "jsDelivr: malformed tags response"}
        if not version:
            return {"success": False, "error": "jsDelivr: no latest tag for package"}
    url = f"{_JSDELIVR_API}/packages/npm/{pkg_name}@{version}?structure=flat"
    try:
        r = await c.get(url)
    except httpx.RequestError as e:
        return {"success": False, "error": f"jsDelivr request failed: {e}"}
    if r.status_code == 403:
        return {"success": False, "error": "jsDelivr: package too large (>100 MB)"}
    if r.status_code != 200:
        return {"success": False, "error": api_error("jsDelivr", r)}
    data = r.json()
    files_list = data.get("files", [])
    names = [f["name"] for f in files_list if isinstance(f, dict) and not f.get("files")]
    return {
        "success": True,
        "package": pkg_name,
        "version": data.get("version", version or "latest"),
        "default": data.get("default", ""),
        "files": names,
        "total": len(names),
    }


async def jsdelivr_read_file(spec: str, file_path: str) -> dict[str, Any]:
    """Read raw file content from an npm package via jsDelivr CDN.
    spec format: 'npm:express' or 'npm:express@5.2.1'.
    """
    parts = spec.split(":")
    pkg_name = parts[-1] if len(parts) > 1 else spec
    version = ""
    if "@" in pkg_name and not pkg_name.startswith("@"):
        pkg_name, version = pkg_name.rsplit("@", 1)
    ver_part = f"@{version}" if version else ""
    # L9: reject traversal up front — CDN normalizes it, but don't send it
    if ".." in file_path.split("/"):
        return {"success": False, "error": f"invalid path: {file_path}"}
    file_path_clean = file_path if file_path.startswith("/") else f"/{file_path}"
    url = f"https://cdn.jsdelivr.net/npm/{pkg_name}{ver_part}{file_path_clean}"
    try:
        r = await safe_fetch(get_http_client(), url)
    except httpx.RequestError as e:
        return {"success": False, "error": f"jsDelivr CDN request failed: {e}"}
    if r.status_code != 200:
        return {"success": False, "error": api_error("jsDelivr CDN", r)}
    return {
        "success": True,
        "content": r.text,
        "file_path": file_path_clean,
        "content_type": r.headers.get("content-type", ""),
        "total_chars": len(r.text),
    }
