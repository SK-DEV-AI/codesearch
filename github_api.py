from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from config import GH_API, GH_SEARCH_CODE, GH_SEARCH_ISSUES, GH_SEARCH_REPOS, GH_TOKEN, _cached, _set_cache, _http_request, _next_gh_key, get_http_client, api_error

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
    cache_key = f"gh:{search_type}:{q}:{owner}:{repo}:{count}:{sort}:{order}:{filename}:{extension}:{path}:{created}:{state}:{user}:{org}:{page}:{exclude_qualifier}:{merged}:{head}:{base}:{review}:{in_qualifier}:{is_}:{pushed}:{stars}:{forks}:{topics}:{labels}:{size}"
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
            return {"success": False, "error": api_error("GitHub", r)}
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
            return {"success": False, "error": api_error("GitHub", r)}
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
            return {"success": False, "error": api_error("GitHub", r)}
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
            return {"success": False, "error": api_error("GitHub", r)}
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
            return {"success": False, "error": api_error("GitHub", r)}
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
    cache_key = f"gh_commits:{query}:{owner}:{repo}:{count}:{sort}:{order}:{author}:{committer}:{hash_}:{author_date}:{committer_date}:{merge}:{page}"
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
        repo_r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}", headers=await _gh_headers())
        default_branch = "main"
        if repo_r.status_code == 200:
            default_branch = repo_r.json().get("default_branch", "main")
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/branches", headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": api_error("GitHub", r)}
        branches = [{"name": b.get("name", ""), "sha": b.get("commit", {}).get("sha", ""),
                      "protected": b.get("protected", False)}
                     for b in (r.json() or [])]
        result = {"success": True, "branches": branches, "default_branch": default_branch}
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
            return {"success": False, "error": api_error("GitHub", r)}
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
            repo_r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}",
                headers=await _gh_headers())
            if repo_r.status_code == 200:
                tree_sha = repo_r.json().get("default_branch", "HEAD")
            else:
                for branch in ("main", "master"):
                    ref_r = await _http_request("GET",
                        f"{GH_API}/repos/{owner}/{repo}/git/refs/heads/{branch}",
                        headers=await _gh_headers())
                    if ref_r.status_code == 200:
                        tree_sha = ref_r.json().get("object", {}).get("sha", "HEAD")
                        break
        params = {"recursive": "1"} if recursive else {}
        r = await _http_request("GET", f"{GH_API}/repos/{owner}/{repo}/git/trees/{tree_sha}",
            params=params, headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": api_error("GitHub tree", r)}
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
            return {"success": False, "error": api_error("GitHub", r)}
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
    cache_key = f"gh_lbl:{query}:{repository_id}:{sort}:{order}"
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
            return {"success": False, "error": api_error("GitHub labels", r)}
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
            return {"success": False, "error": api_error("GitHub topics", r)}
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


async def gh_get_issue(owner: str, repo: str, issue_number: int) -> dict:
    """Get a full issue/PR with body, comments, events, and timeline."""
    cache_key = f"gh_issue:{owner}:{repo}:{issue_number}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        issue_r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/issues/{issue_number}",
            headers=await _gh_headers())
        if issue_r.status_code != 200:
            return {"success": False, "error": api_error("GitHub issue", issue_r)}
        issue = issue_r.json()

        comments_r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/issues/{issue_number}/comments",
            params={"per_page": 30, "sort": "created", "direction": "asc"},
            headers=await _gh_headers())
        comments = []
        if comments_r.status_code == 200:
            for c in (comments_r.json() or []):
                comments.append({
                    "user": c.get("user", {}).get("login", ""),
                    "body": (c.get("body", "") or "")[:2000],
                    "created_at": c.get("created_at", ""),
                    "updated_at": c.get("updated_at", ""),
                })

        timeline_r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/issues/{issue_number}/timeline",
            params={"per_page": 30},
            headers=await _gh_headers())
        events = []
        if timeline_r.status_code == 200:
            for ev in (timeline_r.json() or []):
                events.append({
                    "event": ev.get("event", ""),
                    "actor": ev.get("actor", {}).get("login", "") if ev.get("actor") else "",
                    "created_at": ev.get("created_at", ""),
                    "label": ev.get("label", {}).get("name", "") if ev.get("label") else "",
                    "commit_id": (ev.get("commit_id", "") or "")[:8],
                })

        result = {
            "success": True,
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "state": issue.get("state", ""),
            "body": (issue.get("body", "") or "")[:8000],
            "user": issue.get("user", {}).get("login", ""),
            "labels": [l.get("name", "") for l in (issue.get("labels", []) or [])],
            "assignees": [a.get("login", "") for a in (issue.get("assignees", []) or [])],
            "milestone": issue.get("milestone", {}).get("title", "") if issue.get("milestone") else "",
            "created_at": issue.get("created_at", ""),
            "updated_at": issue.get("updated_at", ""),
            "closed_at": issue.get("closed_at"),
            "comments": issue.get("comments", 0),
            "pull_request": issue.get("pull_request", {}).get("url", "") if issue.get("pull_request") else "",
            "comment_data": comments,
            "timeline_events": events,
        }
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_pr(owner: str, repo: str, pr_number: int) -> dict:
    """Get a full PR with body, commits, files changed, and merge status."""
    cache_key = f"gh_pr:{owner}:{repo}:{pr_number}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        pr_r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers=await _gh_headers())
        if pr_r.status_code != 200:
            return {"success": False, "error": api_error("GitHub PR", pr_r)}
        pr = pr_r.json()

        commits_r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/pulls/{pr_number}/commits",
            params={"per_page": 30},
            headers=await _gh_headers())
        commits = []
        if commits_r.status_code == 200:
            for c in (commits_r.json() or []):
                commit = c.get("commit", {})
                commits.append({
                    "sha": c.get("sha", "")[:8],
                    "message": (commit.get("message", "") or "")[:200],
                    "author": commit.get("author", {}).get("name", "") if commit.get("author") else "",
                    "date": commit.get("committer", {}).get("date", "") if commit.get("committer") else "",
                })

        files_r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/pulls/{pr_number}/files",
            params={"per_page": 30},
            headers=await _gh_headers())
        files = []
        if files_r.status_code == 200:
            for f in (files_r.json() or []):
                files.append({
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "changes": f.get("changes", 0),
                })

        result = {
            "success": True,
            "number": pr.get("number"),
            "title": pr.get("title", ""),
            "state": pr.get("state", ""),
            "body": (pr.get("body", "") or "")[:8000],
            "user": pr.get("user", {}).get("login", ""),
            "base_branch": pr.get("base", {}).get("ref", "") if pr.get("base") else "",
            "head_branch": pr.get("head", {}).get("ref", "") if pr.get("head") else "",
            "head_repo": pr.get("head", {}).get("repo", {}).get("full_name", "") if pr.get("head") and pr.get("head", {}).get("repo") else "",
            "mergeable": pr.get("mergeable"),
            "mergeable_state": pr.get("mergeable_state", ""),
            "merged": pr.get("merged", False),
            "merged_by": pr.get("merged_by", {}).get("login", "") if pr.get("merged_by") else "",
            "draft": pr.get("draft", False),
            "created_at": pr.get("created_at", ""),
            "updated_at": pr.get("updated_at", ""),
            "closed_at": pr.get("closed_at"),
            "merged_at": pr.get("merged_at"),
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0),
            "changed_files": pr.get("changed_files", 0),
            "commits": pr.get("commits", 0),
            "comments": pr.get("comments", 0),
            "review_comments": pr.get("review_comments", 0),
            "commit_data": commits,
            "file_data": files,
        }
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_pr_reviews(owner: str, repo: str, pr_number: int) -> dict:
    """Get PR reviews with comments and state (approved, changes_requested, etc.)."""
    cache_key = f"gh_pr_reviews:{owner}:{repo}:{pr_number}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        r = await _http_request("GET",
            f"{GH_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            headers=await _gh_headers())
        if r.status_code != 200:
            return {"success": False, "error": api_error("GitHub PR reviews", r)}
        reviews = []
        for rev in (r.json() or []):
            reviews.append({
                "user": rev.get("user", {}).get("login", ""),
                "state": rev.get("state", ""),
                "body": (rev.get("body", "") or "")[:1000],
                "submitted_at": rev.get("submitted_at", ""),
                "commit_id": (rev.get("commit_id", "") or "")[:8],
            })
        result = {"success": True, "reviews": reviews, "total": len(reviews)}
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}


async def gh_get_user(username: str) -> dict:
    """Get a GitHub user profile with bio, stats, orgs, and recent repos."""
    cache_key = f"gh_user:{username}"
    cached = await _cached(cache_key)
    if cached is not None:
        return cached
    try:
        user_r = await _http_request("GET", f"{GH_API}/users/{username}",
            headers=await _gh_headers())
        if user_r.status_code != 200:
            return {"success": False, "error": api_error("GitHub user", user_r)}
        u = user_r.json()

        repos_r = await _http_request("GET", f"{GH_API}/users/{username}/repos",
            params={"sort": "updated", "per_page": 10, "type": "public"},
            headers=await _gh_headers())
        repos = []
        if repos_r.status_code == 200:
            for repo in (repos_r.json() or []):
                repos.append({
                    "name": repo.get("full_name", ""),
                    "description": (repo.get("description", "") or "")[:100],
                    "stars": repo.get("stargazers_count", 0),
                    "language": repo.get("language") or "",
                    "updated_at": repo.get("updated_at", ""),
                })

        orgs_r = await _http_request("GET", f"{GH_API}/users/{username}/orgs",
            headers=await _gh_headers())
        orgs = []
        if orgs_r.status_code == 200:
            orgs = [o.get("login", "") for o in (orgs_r.json() or [])]

        result = {
            "success": True,
            "login": u.get("login", ""),
            "name": u.get("name", "") or "",
            "avatar_url": u.get("avatar_url", ""),
            "bio": (u.get("bio", "") or "")[:500],
            "company": u.get("company", "") or "",
            "location": u.get("location", "") or "",
            "blog": u.get("blog", "") or "",
            "email": u.get("email", "") or "",
            "twitter": u.get("twitter_username", "") or "",
            "public_repos": u.get("public_repos", 0),
            "public_gists": u.get("public_gists", 0),
            "followers": u.get("followers", 0),
            "following": u.get("following", 0),
            "created_at": u.get("created_at", ""),
            "updated_at": u.get("updated_at", ""),
            "hireable": u.get("hireable", False),
            "type": u.get("type", "User"),
            "site_admin": u.get("site_admin", False),
            "repos": repos,
            "orgs": orgs,
        }
        await _set_cache(cache_key, result)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"success": False, "error": str(e)}



