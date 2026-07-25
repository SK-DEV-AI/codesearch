"""Repo analyzer meta-tool: parallel-fetch GitHub metadata + code quality + architecture."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from config import get_http_client
from github_api import (gh_get_repo, fetch_readme, gh_get_languages,
                         gh_get_releases, gh_get_topics)
from searchcode import analyze as searchcode_analyze
from deepwiki import deepwiki_ask
from codewiki import codewiki_fetch_repo

_GH_URL_RE = re.compile(
    r"(?:https?://(?:www\.)?github\.com/)?([^/\s]+)/([^/\s#?]+)"
)


def _parse_repo(input_str: str) -> tuple[str, str] | None:
    m = _GH_URL_RE.match(input_str.strip())
    if m:
        return m.group(1), m.group(2).rstrip("/").rstrip(".git")
    return None


async def analyze_repo(repository: str) -> dict:
    """Parallel-call GitHub metadata + SearchCode + DeepWiki + CodeWiki for a repo.

    Args:
        repository: GitHub repo URL (https://github.com/owner/repo) or "owner/repo" string.

    Returns:
        Combined response with metadata, readme, languages, releases, topics,
        code quality analysis, architecture, and per-source availability.
    """
    parsed = _parse_repo(repository)
    if not parsed:
        return {"success": False,
                "error": "could not parse repository. Use format: owner/repo or full GitHub URL"}

    owner, repo = parsed
    repo_full = f"{owner}/{repo}"
    repo_url = f"https://github.com/{repo_full}"

    tasks: dict[str, Any] = {
        "repo": gh_get_repo(owner, repo),
        "readme": fetch_readme(owner, repo),
        "languages": gh_get_languages(owner, repo),
        "releases": gh_get_releases(owner, repo, 5),
        "topics": gh_get_topics(owner, repo),
        "searchcode": searchcode_analyze(repo_url),
        "deepwiki": deepwiki_ask(owner, repo,
                                f"What is the architecture of {repo}? Describe its structure, modules, and data flow."),
        "codewiki": codewiki_fetch_repo(owner, repo),
    }

    sem = asyncio.Semaphore(8)
    results: dict[str, Any] = {}

    async def _run(name: str, coro: Any) -> None:
        async with sem:
            try:
                res = await asyncio.wait_for(coro, timeout=15)
                results[name] = res
            except asyncio.TimeoutError:
                results[name] = {"success": False, "error": f"{name} timed out"}
            except Exception as e:
                results[name] = {"success": False, "error": str(e)[:200]}

    await asyncio.gather(*[_run(n, t) for n, t in tasks.items()])

    output: dict[str, Any] = {"success": True}

    # Merge repo metadata
    repo_res = results.get("repo", {})
    if repo_res.get("success"):
        d = repo_res
        output["metadata"] = {
            "name": repo_full,
            "description": (d.get("description", "") or "")[:500],
            "stars": d.get("stargazers_count", 0),
            "forks": d.get("forks_count", 0),
            "language": d.get("language", "") or "",
            "default_branch": d.get("default_branch", ""),
            "license": d.get("license", ""),
            "homepage": d.get("homepage", "") or "",
            "topics": d.get("topics", []),
            "open_issues": d.get("open_issues_count", 0),
            "created_at": d.get("created_at", ""),
            "updated_at": d.get("updated_at", ""),
            "pushed_at": d.get("pushed_at", ""),
            "size_kb": d.get("size", 0),
            "has_wiki": d.get("has_wiki", False),
            "archived": d.get("archived", False),
        }
    else:
        output["metadata"] = {"name": repo_full}
        output["metadata_error"] = repo_res.get("error", "unavailable")

    # README preview
    readme_res = results.get("readme", {})
    if readme_res.get("success") and readme_res.get("content"):
        output["readme_preview"] = readme_res["content"][:3000]
    else:
        output["readme_preview"] = ""

    # Languages
    lang_res = results.get("languages", {})
    if lang_res.get("success") and lang_res.get("languages"):
        total = sum(lang_res["languages"].values()) or 1
        langs = [(k, v, round(v / total * 100, 1))
                 for k, v in sorted(lang_res["languages"].items(), key=lambda x: -x[1])]
        output["languages"] = [{"language": k, "bytes": v, "percent": pct}
                                for k, v, pct in langs]
        output["total_code_bytes"] = total
    else:
        output["languages"] = []

    # Releases
    rel_res = results.get("releases", {})
    if rel_res.get("success") and rel_res.get("releases"):
        output["releases"] = [
            {
                "tag_name": r.get("tag_name", ""),
                "name": r.get("name", "") or r.get("tag_name", ""),
                "published_at": r.get("published_at", ""),
                "prerelease": r.get("prerelease", False),
                "body_preview": (r.get("body", "") or "")[:300],
            }
            for r in rel_res["releases"][:5]
        ]
    else:
        output["releases"] = []

    # Topics
    top_res = results.get("topics", {})
    if top_res.get("success") and top_res.get("names"):
        output["topics"] = top_res["names"]
    else:
        output["topics"] = []

    # Code quality (SearchCode)
    sc_res = results.get("searchcode", {})
    if isinstance(sc_res, dict) and sc_res.get("success"):
        sc_data = sc_res.get("data", sc_res.get("results", {}))
        if isinstance(sc_data, dict):
            output["code_quality"] = {
                "total_lines": sc_data.get("total_lines", 0),
                "languages": sc_data.get("languages", []),
                "tech_stack": sc_data.get("tech_stack", {}),
                "credentials_found": sc_data.get("credentials", False),
                "file_count": sc_data.get("file_count", 0),
            }
        else:
            output["code_quality"] = {"summary": str(sc_data)[:500]}
    else:
        output["code_quality"] = None

    # Architecture (DeepWiki + CodeWiki)
    arch: dict[str, Any] = {}
    dw_res = results.get("deepwiki", {})
    if isinstance(dw_res, dict) and dw_res.get("success"):
        arch["deepwiki"] = (dw_res.get("answer", "") or "")[:3000]
    else:
        arch["deepwiki"] = None

    cw_res = results.get("codewiki", {})
    if isinstance(cw_res, dict) and cw_res.get("success"):
        sections = cw_res.get("sections", cw_res.get("results", []))
        if sections:
            arch["codewiki"] = [
                {"title": s.get("title", s.get("name", "")),
                 "content": (s.get("content", s.get("summary", "")) or "")[:500]}
                for s in (sections if isinstance(sections, list) else [sections])
            ][:5]
        else:
            arch["codewiki"] = []
    else:
        arch["codewiki"] = []

    output["architecture"] = arch

    # Per-source availability
    output["sources"] = {
        "github": bool(repo_res.get("success")),
        "searchcode": bool(isinstance(sc_res, dict) and sc_res.get("success")),
        "deepwiki": bool(isinstance(dw_res, dict) and dw_res.get("success")),
        "codewiki": bool(isinstance(cw_res, dict) and cw_res.get("success")),
    }

    return output
