from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from config import GH_API, GH_SEARCH_CODE, GH_SEARCH_ISSUES, GH_SEARCH_REPOS, GH_TOKEN, _cached, _set_cache, _http_request, _next_gh_key, get_http_client

logger = logging.getLogger("codesearch.github")


async def _gh_headers(media_type: str = "github+json") -> dict[str, str]:
    h = {"User-Agent": "mcp-codesearch/1.0", "Accept": f"application/vnd.{media_type}"}
    token = await _next_gh_key() or GH_TOKEN
    if token:
        h["Authorization"] = f"token {token}"
    return h


async def search_github(q: str, search_type: str = "code", count: int = 10,
                        owner: str = "", repo: str = "", language: str = "",
                        sort: str = "", order: str = "",
                        filename: str = "", extension: str = "",
                        path: str = "", created: str = "",
                        state: str = "", labels: str = "",
                        user: str = "", org: str = "",
                        size: str = "", in_qualifier: str = "",
                        is_: str = "", pushed: str = "",
                        stars: str = "", forks: str = "",
                        topics: str = "", page: int = 1,
                        exclude_qualifier: str = "",
                        merged: str = "", head: str = "", base: str = "",
                        review: str = "") -> dict:
    cache_key = f"gh:{search_type}:{q}:{owner}:{repo}:{count}:{sort}:{order}:{filename}:{extension}:{path}:{created}:{state}:{user}:{org}"
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "results": cached, "cached": True}
    try:
        query_parts = [q]
        if owner:
            query_parts.append(f"user:{owner}" if not repo else f"repo:{owner}/{repo}")
        if repo and not owner:
            query_parts.append(f"repo:{repo}")
        if user:
            query_parts.append(f"user:{user}")
        if org:
            query_parts.append(f"org:{org}")
        if language:
            query_parts.append(f"language:{language}")
        if filename:
            query_parts.append(f"filename:{filename}")
        if extension:
            query_parts.append(f"extension:{extension}")
        if path:
            query_parts.append(f"path:{path}")
        if size:
            query_parts.append(f"size:{size}")
        if in_qualifier:
            query_parts.append(f"in:{in_qualifier}")
        if is_:
            query_parts.append(f"is:{is_}")
        if created and search_type == "issues":
            query_parts.append(f"created:{created}")
        if pushed and search_type == "repos":
            query_parts.append(f"pushed:{pushed}")
        if stars and search_type in ("repos", "issues"):
            query_parts.append(f"stars:{stars}")
        if forks and search_type == "repos":
            query_parts.append(f"forks:{forks}")
        if topics and search_type == "repos":
            query_parts.append(f"topics:{topics}")
        if state and search_type == "issues":
            query_parts.append(f"state:{state}")
        if labels and search_type == "issues":
            for lbl in labels.split(","):
                query_parts.append(f"label:{lbl.strip()}")
        if merged and search_type == "issues":
            query_parts.append(f"merged:{merged}")
        if head:
            query_parts.append(f"head:{head}")
        if base:
            query_parts.append(f"base:{base}")
        if review and search_type == "issues":
            query_parts.append(f"review:{review}")
        if exclude_qualifier:
            query_parts.append(f"NOT {exclude_qualifier}")
        full_query = " ".join(query_parts)
        if search_type == "repos":
            url = GH_SEARCH_REPOS
        elif search_type == "issues":
            url = GH_SEARCH_ISSUES
        elif search_type == "users":
            url = "https://api.github.com/search/users"
        else:
            url = GH_SEARCH_CODE
        params: dict[str, Any] = {"q": full_query, "per_page": min(count, 100)}
        if page > 1:
            params["page"] = page
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        mt = "github.v3.text-match+json" if search_type == "code" else "github+json"
        r = await _http_request("GET", url, params=params, headers=await _gh_headers(mt), timeout=15)
        remaining = r.headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                remaining_int = int(remaining)
                if remaining_int < 10:
                    logger.warning("GitHub API rate limit low: %s remaining", remaining_int)
            except (ValueError, TypeError):
                pass
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub API: {r.status_code} {r.text[:200]}"}
        data = r.json()
        items = data.get("items", [])[:count]
        results = []
        if search_type == "code":
            for item in items:
                text_matches = item.get("text_matches", [])
                snippet = text_matches[0]["fragment"] if text_matches else ""
                results.append({
                    "file": item["name"], "path": item["path"],
                    "url": item["html_url"], "repo": item["repository"]["full_name"],
                    "snippet": snippet[:500],
                })
        elif search_type == "repos":
            for item in items:
                results.append({
                    "full_name": item["full_name"],
                    "description": (item.get("description") or "")[:200],
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "language": item.get("language") or "",
                    "topics": item.get("topics", []),
                    "url": item["html_url"],
                    "updated_at": item.get("updated_at", ""),
                    "open_issues": item.get("open_issues_count", 0),
                    "license": (item.get("license") or {}).get("spdx_id", ""),
                })
        elif search_type == "users":
            for item in items:
                results.append({
                    "login": item.get("login", ""),
                    "avatar": item.get("avatar_url", ""),
                    "html_url": item.get("html_url", ""),
                    "type": item.get("type", "User"),
                    "score": item.get("score", 0),
                })
        else:
            for item in items:
                results.append({
                    "title": item["title"],
                    "state": item["state"],
                    "body": (item.get("body") or "")[:500],
                    "labels": [l["name"] for l in item.get("labels", [])],
                    "url": item["html_url"],
                    "repo": item.get("repository_url", "").replace("https://api.github.com/repos/", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                    "comments": item.get("comments", 0),
                    "user": item.get("user", {}).get("login", ""),
                })
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": data.get("total_count", 0)}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def fetch_readme(owner: str, repo: str, branch: str = "") -> dict:
    cache_key = f"readme:{owner}:{repo}:{branch}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        url = f"{GH_API}/repos/{owner}/{repo}/readme"
        if branch:
            url += f"?ref={branch}"
        gh_headers = await _gh_headers()
        r = await _http_request("GET", url, headers={**gh_headers, "Accept": "application/vnd.github.raw"})
        if r.status_code == 404:
            return {"success": False, "error": "no README found"}
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        result = {"success": True, "content": r.text[:15000], "repo": f"{owner}/{repo}"}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_contents(owner: str, repo: str, path: str = "", branch: str = "") -> dict:
    try:
        url = f"{GH_API}/repos/{owner}/{repo}/contents/{path}"
        params: dict[str, str] = {}
        if branch:
            params["ref"] = branch
        r = await _http_request("GET", url, params=params, headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        data = r.json()
        if isinstance(data, list):
            entries = [{"name": f["name"], "type": f["type"], "size": f.get("size", 0)} for f in data]
            return {"success": True, "entries": entries}
        content = ""
        if data.get("encoding") == "base64":
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return {"success": True, "name": data.get("name", ""), "content": content[:15000], "size": data.get("size", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_languages(owner: str, repo: str) -> dict:
    cache_key = f"gh_lang:{owner}:{repo}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/languages", headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        result = {"success": True, "languages": r.json()}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_topics(owner: str, repo: str) -> dict:
    cache_key = f"gh_topics:{owner}:{repo}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        gh_hdrs = await _gh_headers()
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/topics",
                                headers=gh_hdrs)
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        result = {"success": True, "topics": r.json().get("names", [])}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_repo(owner: str, repo: str) -> dict:
    """Get repository metadata (stars, forks, license, description, etc.)."""
    cache_key = f"gh_repo:{owner}:{repo}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}", headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        d = r.json()
        result = {"success": True,
            "name": d.get("name", ""), "full_name": d.get("full_name", ""),
            "description": d.get("description", ""),
            "stars": d.get("stargazers_count", 0), "forks": d.get("forks_count", 0),
            "watchers": d.get("subscribers_count", 0), "open_issues": d.get("open_issues_count", 0),
            "language": d.get("language") or "", "topics": d.get("topics", []),
            "license": d.get("license", {}).get("spdx_id", "") if d.get("license") else "",
            "url": d.get("html_url", ""), "homepage": d.get("homepage", "") or "",
            "created_at": d.get("created_at", ""), "updated_at": d.get("updated_at", ""),
            "archived": d.get("archived", False), "fork": d.get("fork", False),
            "default_branch": d.get("default_branch", ""),
        }
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_commits(query: str, count: int = 10, sort: str = "", order: str = "",
                          owner: str = "", repo: str = "", author: str = "",
                          committer: str = "", author_date: str = "", committer_date: str = "",
                          merge: str = "", hash_: str = "", page: int = 1) -> dict:
    """Search commits with qualifiers."""
    cache_key = f"gh_commits:{query}:{owner}:{repo}:{count}:{sort}:{order}:{author}:{committer}"
    cached = await _cached(cache_key)
    if cached is not None:
        return {"success": True, "results": cached, "cached": True}
    try:
        query_parts = [query]
        if owner and repo:
            query_parts.append(f"repo:{owner}/{repo}")
        if author:
            query_parts.append(f"author:{author}")
        if committer:
            query_parts.append(f"committer:{committer}")
        if author_date:
            query_parts.append(f"author-date:{author_date}")
        if committer_date:
            query_parts.append(f"committer-date:{committer_date}")
        if merge:
            query_parts.append(f"merge:{merge}")
        if hash_:
            query_parts.append(f"hash:{hash_}")
        full_query = " ".join(query_parts)
        url = "https://api.github.com/search/commits"
        params: dict[str, Any] = {"q": full_query, "per_page": min(count, 100)}
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        if page > 1:
            params["page"] = page
        r = await _http_request("GET", url, params=params, headers=await _gh_headers("github.v3.text-match+json"))
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub commits: {r.status_code} {r.text[:200]}"}
        data = r.json()
        results = []
        for item in (data.get("items", []) or [])[:count]:
            commit = item.get("commit", {})
            author_info = commit.get("author", {}) or {}
            results.append({
                "sha": item.get("sha", ""),
                "message": (commit.get("message", "") or "")[:500],
                "author": author_info.get("name", ""),
                "author_email": author_info.get("email", ""),
                "author_date": author_info.get("date", ""),
                "committer": (commit.get("committer", {}) or {}).get("name", ""),
                "url": item.get("html_url", ""),
                "repo": item.get("repository", {}).get("full_name", ""),
            })
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": data.get("total_count", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_branches(owner: str, repo: str) -> dict:
    """List branches for a repository."""
    cache_key = f"gh_branches:{owner}:{repo}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/branches", headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        branches = [{"name": b.get("name", ""), "sha": b.get("commit", {}).get("sha", ""),
                      "protected": b.get("protected", False)}
                     for b in (r.json() or [])]
        result = {"success": True, "branches": branches, "default_branch": "main"}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_tags(owner: str, repo: str) -> dict:
    """List tags for a repository."""
    cache_key = f"gh_tags:{owner}:{repo}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/tags", headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        tags = [{"name": t.get("name", ""), "sha": t.get("commit", {}).get("sha", "")}
                 for t in (r.json() or [])]
        result = {"success": True, "tags": tags}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_tree(owner: str, repo: str, tree_sha: str = "HEAD", recursive: bool = True) -> dict:
    """Get a git tree recursively for a repository."""
    cache_key = f"gh_tree:{owner}:{repo}:{tree_sha}:{recursive}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        if tree_sha == "HEAD":
            ref_r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/git/refs/heads/main",
                headers=await _gh_headers())
            if ref_r.status_code != 200:
                ref_r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/git/refs/heads/master",
                    headers=await _gh_headers())
            if ref_r.status_code == 200:
                tree_sha = ref_r.json().get("object", {}).get("sha", "HEAD")
        params = {"recursive": "1"} if recursive else {}
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/git/trees/{tree_sha}",
            params=params, headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub tree {r.status_code}"}
        data = r.json()
        entries = []
        for t in (data.get("tree", []) or []):
            entries.append({
                "path": t.get("path", ""),
                "type": t.get("type", ""),
                "size": t.get("size", 0),
            })
        result = {"success": True, "tree_sha": data.get("sha", ""), "entries": entries,
                   "truncated": data.get("truncated", False)}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_releases(owner: str, repo: str, count: int = 5) -> dict:
    cache_key = f"gh_rel:{owner}:{repo}:{count}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/releases",
                                params={"per_page": min(count, 20)}, headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub {r.status_code}"}
        items = r.json()[:count]
        results = [{"tag": rel.get("tag_name", ""), "name": rel.get("name", ""),
                     "published": rel.get("published_at", ""), "prerelease": rel.get("prerelease", False),
                     "body": (rel.get("body") or "")[:500]}
                    for rel in items]
        result = {"success": True, "releases": results}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_labels(query: str, repository_id: int = 0, sort: str = "",
                        order: str = "", count: int = 10) -> dict:
    """Search labels within a repository by repository_id."""
    cache_key = f"gh_lbl:{query}:{repository_id}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        params: dict[str, Any] = {"q": query, "per_page": min(count, 100)}
        if sort: params["sort"] = sort
        if order: params["order"] = order
        if repository_id: params["repository_id"] = repository_id
        r = await _http_request("GET", "https://api.github.com/search/labels",
                                params=params, headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub labels: {r.status_code}"}
        items = (r.json().get("items", []) or [])[:count]
        results = [{"name": lb.get("name", ""), "description": lb.get("description", ""),
                     "color": lb.get("color", ""), "default": lb.get("default", False)}
                    for lb in items]
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": r.json().get("total_count", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def search_topics(query: str, count: int = 10) -> dict:
    """Search topics by query."""
    cache_key = f"gh_tpc:{query}:{count}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET", "https://api.github.com/search/topics",
            params={"q": query, "per_page": min(count, 100)},
            headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": f"GitHub topics: {r.status_code}"}
        items = (r.json().get("items", []) or [])[:count]
        results = [{"name": t.get("name", ""), "description": t.get("description", ""),
                     "short_description": t.get("short_description", ""),
                     "aliases": t.get("aliases", []),
                     "created": t.get("created_at", ""), "updated": t.get("updated_at", "")}
                    for t in items]
        await _set_cache(cache_key, results)
        return {"success": True, "results": results, "total": r.json().get("total_count", 0)}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
