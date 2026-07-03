"""Direct HTTP wrappers for GitHits APIs (REST + PkgSeer GraphQL).

No CLI subprocess, no asyncio.Lock, no race conditions — plain httpx
calls to api.githits.com (REST search) and pkgseer.dev (GraphQL).
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from config import GITHITS_API_TOKEN

_API_URL = "https://api.githits.com"
_PKGSEER_URL = "https://pkgseer.dev"
_TIMEOUT = 90

_HEADERS = {"Authorization": f"Bearer {GITHITS_API_TOKEN}", "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Helpers
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
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(url, headers=_HEADERS, json=body)
    except httpx.TimeoutException:
        return {"success": False, "error": f"GitHits request timed out ({timeout}s)"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"GitHits request failed: {e}"}
    if resp.status_code == 401:
        return {"success": False, "error": "GitHits API rejected the token (401)"}
    if resp.status_code >= 400:
        return {"success": False, "error": f"GitHits API error ({resp.status_code}): {resp.text[:300]}"}
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
    # Surface GraphQL-level errors
    if isinstance(data, dict) and data.get("errors"):
        msgs = [e.get("message", str(e)) for e in data["errors"]]
        return {"success": False, "error": "; ".join(msgs)}
    return result


def _parse_spec(spec: str) -> dict[str, str]:
    """Parse 'registry:name[@version]' into dict."""
    registry = "npm"
    name = spec
    version = ""
    if ":" in name:
        registry, name = name.split(":", 1)
    if "@" in name and not name.startswith("@"):
        name, version = name.rsplit("@", 1)
    # Handle scoped npm packages like @angular/core
    if name.count("@") > 0 and not version:
        parts = name.rsplit("@", 1)
        if parts[0].startswith("@"):
            name = spec.split(":", 1)[1] if ":" in spec else spec  # keep original
    # Normalise npm registries to uppercase
    registry_map = {"npm": "NPM", "pypi": "PYPI", "crates": "CRATESIO"}
    registry = registry_map.get(registry.lower(), registry.upper())
    return {"registry": registry, "name": name, "version": version}


def _extract_solution_id(text: str) -> tuple[str, str | None]:
    """Extract trailing solution_id line from markdown response."""
    m = re.search(r"\nsolution_id:\s*(\S+)\s*$", text)
    if m:
        return text[: m.start()], m.group(1)
    return text, None


# ---------------------------------------------------------------------------
# REST API — api.githits.com
# ---------------------------------------------------------------------------


async def get_example(query: str, language: str = "") -> dict[str, Any]:
    """Canonical open-source examples via REST /search."""
    err = _check_token()
    if err:
        return err
    body = {"query": query, "include_explanation": False}
    if language:
        body["language"] = language
    result = await _post_json(f"{_API_URL}/search", body, timeout=_TIMEOUT)
    if not result.get("success"):
        return result
    text = result.get("text", "")
    if not text:
        return {"success": False, "error": "empty response from GitHits search"}
    content, solution_id = _extract_solution_id(text)
    payload = {"result": content}
    if solution_id:
        payload["solution_id"] = solution_id
    return {"success": True, **payload}


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
        targets = [parts]
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
    variables = {**pkg, "filePath": path, "startLine": None, "endLine": None, "waitTimeoutMs": 30000}
    result = await _pkgseer_graphql(FETCH_CODE_CONTEXT_QUERY, variables, timeout=30)
    if not result.get("success"):
        return result
    data = result.get("data", {}).get("data", {}).get("fetchCodeContext", {}) or {}
    return {"success": True, "content": data.get("content", ""), "filePath": data.get("filePath", path)}


GREP_REPO_QUERY = """query GrepRepo($registry:Registry $packageName:String $repoUrl:String $gitRef:String $version:String $waitTimeoutMs:Int $pattern:String! $patternType:GrepPatternType $caseSensitive:Boolean $pathSelectors:[GrepPathSelectorInput!] $extensions:[String!] $contextLinesBefore:Int $contextLinesAfter:Int $maxMatches:Int $maxMatchesPerFile:Int){grepRepo(registry:$registry packageName:$packageName repoUrl:$repoUrl gitRef:$gitRef version:$version waitTimeoutMs:$waitTimeoutMs pattern:$pattern patternType:$patternType caseSensitive:$caseSensitive pathSelectors:$pathSelectors extensions:$extensions contextLinesBefore:$contextLinesBefore contextLinesAfter:$contextLinesAfter maxMatches:$maxMatches maxMatchesPerFile:$maxMatchesPerFile){matches{filePath line lineContent contextBefore contextAfter}nextCursor hasMore totalMatches uniqueFilesMatched}}"""


async def code_grep(spec: str, pattern: str, path_prefix: str = "") -> dict[str, Any]:
    """Grep through indexed dependency source for a pattern."""
    err = _check_token()
    if err:
        return err
    pkg = _parse_spec(spec)
    path_selectors = [{"kind": "PREFIX", "value": path_prefix}] if path_prefix else None
    variables = {
        **pkg, "pattern": pattern, "patternType": "LITERAL",
        "caseSensitive": False, "pathSelectors": path_selectors,
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
    variables = {**pkg, "includeTransitive": True, "maxDepth": 5}
    result = await _pkgseer_graphql(PKG_DEPS_QUERY, variables, timeout=60)
    if not result.get("success"):
        return result
    data = result.get("data", {}).get("data", {}).get("packageDependencies", {}) or {}
    deps = data.get("dependencies", {})
    return {"success": True, "dependencies": deps, "package": data.get("package", {})}
