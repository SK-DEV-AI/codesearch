from __future__ import annotations

import asyncio
import json

from typing import Any

import httpx

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from config import GH_TOKEN, SOFA_KEY, LI_KEY, _next_fc_key, _next_tv_key, FIRECRAWL_SEARCH, TAVILY_SEARCH
from embed import _embed, _dedup_rank, _hybrid_rank
from code_expand import expand_code_query
from context7 import context7_resolve, search_llms_txt
from github_api import search_github, fetch_readme, gh_get_contents, gh_get_languages, gh_get_topics, gh_get_releases
from deepwiki import deepwiki_fetch, deepwiki_ask
from codewiki import codewiki_fetch_repo, codewiki_search_repos, codewiki_ask_repo
from stack_exchange import search_so, so_similar, so_tags_info, so_tags_wikis
from sofa import search_sofa
from hackernews import search_hn, hn_get_item, hn_firebase_stories
from libraries_io import search_libraries_io, libraries_io_search
from oss_index import scan_vulnerabilities, get_vulnerability_detail, get_component_latest_version
from readthedocs import search_readthedocs, readthedocs_project_info, readthedocs_versions
from registries import search_package, npm_search, crates_search
from devdocs import devdocs_list_docs, devdocs_fetch, devdocs_fetch_content, devdocs_search, devdocs_meta
from semantic_scholar import (search_papers, get_paper_details,
                               get_papers_batch, get_paper_citations,
                               get_paper_references)
from reranker import rerank as _rerank

server = Server("codesearch")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_docs",
            description="Library docs via context7, ReadTheDocs, llms.txt, or DevDocs. Returns candidates when ambiguous.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "library": {"type": "string"},
                    "version": {"type": "string"},
                    "library_id": {"type": "string", "description": "Context7 library ID (skip search)"},
                    "fast": {"type": "boolean"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_code",
            description="GitHub code/repos/issues search with qualifiers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_type": {"type": "string", "enum": ["code","repos","issues","users"], "default": "code"},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "language": {"type": "string"},
                    "count": {"type": "integer", "default": 10},
                    "sort": {"type": "string"},
                    "order": {"type": "string"},
                    "filename": {"type": "string"},
                    "extension": {"type": "string"},
                    "path": {"type": "string"},
                    "created": {"type": "string"},
                    "pushed": {"type": "string"},
                    "state": {"type": "string"},
                    "labels": {"type": "string"},
                    "user": {"type": "string"},
                    "org": {"type": "string"},
                    "size": {"type": "string"},
                    "is_": {"type": "string"},
                    "stars": {"type": "string"},
                    "forks": {"type": "string"},
                    "topics": {"type": "string"},
                    "page": {"type": "integer", "default": 1},
                    "in_name": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="wiki",
            description="Repo architecture and wiki via DeepWiki + CodeWiki.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "fetch", "ask", "architecture"]},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "question": {"type": "string"},
                    "wiki_name": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="search_all",
            description="Unified search across 14+ sources with embedding dedup and reranking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "library": {"type": "string"},
                    "version": {"type": "string"},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "language": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_package",
            description="Package metadata from npm/PyPI/crates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "registry": {"type": "string", "default": "auto"},
                    "type": {"type": "string", "description": "npm_dist_tags|crates_downloads|crates_reverse_deps|crates_owners|crates_categories|crates_keywords"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="so_search",
            description="Stack Overflow: Stack Exchange API (free) or SOFA (needs SOFA_KEY).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["stackexchange", "sofa"], "default": "stackexchange"},
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                    "tags": {"type": "string"},
                    "accepted": {"type": "boolean"},
                    "fromdate": {"type": "string"},
                    "todate": {"type": "string"},
                    "closed": {"type": "boolean"},
                    "sort": {"type": "string", "default": "relevance"},
                    "views": {"type": "integer", "default": 0},
                    "answers": {"type": "integer", "default": 0},
                    "type": {"type": "string", "default": "search"},
                    "site": {"type": "string", "default": "stackoverflow"},
                    "question_id": {"type": "integer"},
                    "content_type": {"type": "string", "enum": ["question", "til", "blueprint"]},
                    "page": {"type": "integer", "default": 1},
                    "post_id": {"type": "string"},
                    "steering": {"type": "string"},
                },
            },
        ),
        Tool(
            name="hn",
            description="Hacker News: search, item detail, or story lists (top/new/best/ask/show).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search","item","stories"]},
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                    "sort_by_date": {"type": "boolean"},
                    "tags": {"type": "string", "default": "story"},
                    "min_points": {"type": "integer", "default": 0},
                    "min_comments": {"type": "integer", "default": 0},
                    "before": {"type": "integer"},
                    "after": {"type": "integer"},
                    "item_id": {"type": "integer"},
                    "firebase_type": {"type": "string"},
                },
                "required": [],
            },
        ),
        Tool(
            name="search_libraries",
            description="Libraries.io dependency metadata and source rank.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "platform": {"type": "string", "default": ""},
                    "query": {"type": "string"},
                    "sort": {"type": "string"},
                    "languages": {"type": "string"},
                    "licenses": {"type": "string"},
                    "keywords": {"type": "string"},
                },
            },
        ),
        Tool(
            name="vulns",
            description="Sonatype Guide: scan packages, vulnerability details, or latest version by PURL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["scan", "detail", "latest_version"]},
                    "platform": {"type": "string"},
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "coordinates": {"type": "string"},
                    "vuln_id": {"type": "string"},
                    "purl": {"type": "string"},
                },
                "required": [],
            },
        ),
        Tool(
            name="github",
            description="GitHub repo operations: readme, contents, languages, topics, releases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["readme", "contents", "languages", "topics", "releases"]},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "path": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                },
                "required": ["action", "owner", "repo"],
            },
        ),
        Tool(
            name="docs",
            description="DevDocs + ReadTheDocs operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["devdocs_list", "devdocs_search", "devdocs_fetch", "devdocs_fetch_content", "devdocs_meta", "rtd_info", "rtd_versions", "rtd_search"]},
                    "slug": {"type": "string"},
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "project": {"type": "string"},
                    "version": {"type": "string"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="papers",
            description="Semantic Scholar: search, details, batch, citations, or references.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "details", "batch", "citations", "references"]},
                    "query": {"type": "string"},
                    "paper_id": {"type": "string"},
                    "paper_ids": {"type": "string", "description": "Comma-separated paper IDs for batch action"},
                    "limit": {"type": "integer", "default": 10},
                    "year": {"type": "string"},
                    "fields_of_study": {"type": "string"},
                    "open_access": {"type": "boolean"},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["action"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    if not isinstance(arguments, dict):
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": "arguments must be a dict"}))],
            isError=True,
        )

    def safe_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _res(data, ok=True):
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(data, default=str))],
            isError=not ok,
        )

    try:
        if name == "search_docs":
            query = str(arguments.get("query", ""))
            library = str(arguments.get("library", "") or query.split()[0])
            version = str(arguments.get("version", ""))
            fast = bool(arguments.get("fast", False))
            library_id = str(arguments.get("library_id", ""))
            resolved = await context7_resolve(
                library if library else query,
                version=version, fast=fast, library_id=library_id,
            )
            if resolved.get("success"):
                result = resolved["docs"]
                result["resolved_library_id"] = resolved["library"]["id"]
                if version:
                    result["version_pinned"] = version
                result["library"] = resolved["library"]
                result["source"] = "context7"
                if resolved.get("candidates"):
                    result["candidates"] = resolved["candidates"]
                return _res(result)
            if resolved.get("candidates"):
                return _res({
                    "success": False,
                    "source": "context7",
                    "candidates": resolved["candidates"],
                    "error": resolved.get("error", "ambiguous library"),
                    "hint": "Use library_id param to pick the right library",
                })
            rtd = await search_readthedocs(library, query)
            if rtd.get("success") and rtd.get("results"):
                rtd["source"] = "readthedocs"
                return _res(rtd)
            base = library.lower().replace('_', '-').replace(' ', '-')
            domains = [
                f"{base}.dev", f"docs.{base}.io", f"{base}.readthedocs.io",
                f"docs.{base}.org", f"{base}.docs.org", f"{base}.docs.dev", f"www.{base}.dev",
            ]
            llms = None
            for domain in domains:
                llms = await search_llms_txt(domain, query)
                if llms.get("success") and llms.get("results"):
                    llms["source"] = "llms.txt"
                    llms["domain"] = domain
                    break
            if llms and llms.get("success") and llms.get("results"):
                return _res(llms)
            dd = await devdocs_search(base, query)
            if dd.get("success") and dd.get("results"):
                dd["source"] = "devdocs"
                return _res(dd)
            return _res({
                "success": False,
                "source": "none",
                "error": f"Could not find docs for '{library}' in context7, readthedocs, or devdocs",
                "hint": "Try search_code or wiki for this library",
            })

        elif name == "search_code":
            r = await search_github(
                q=str(arguments.get("query", "")),
                search_type=str(arguments.get("search_type", "code")),
                count=int(arguments.get("count", 10)),
                owner=str(arguments.get("owner", "")),
                repo=str(arguments.get("repo", "")),
                language=str(arguments.get("language", "")),
                sort=str(arguments.get("sort", "")),
                order=str(arguments.get("order", "")),
                filename=str(arguments.get("filename", "")),
                extension=str(arguments.get("extension", "")),
                path=str(arguments.get("path", "")),
                created=str(arguments.get("created", "")),
                pushed=str(arguments.get("pushed", "")),
                state=str(arguments.get("state", "")),
                labels=str(arguments.get("labels", "")),
                user=str(arguments.get("user", "")),
                org=str(arguments.get("org", "")),
                size=str(arguments.get("size", "")),
                is_=str(arguments.get("is_", "")),
                stars=str(arguments.get("stars", "")),
                forks=str(arguments.get("forks", "")),
                topics=str(arguments.get("topics", "")),
                page=int(arguments.get("page", 1)),
                in_name="true" if bool(arguments.get("in_name", False)) else "",
            )
            return _res(r, r.get("success", False))

        elif name == "wiki":
            action = str(arguments.get("action", ""))
            if action == "search":
                r = await codewiki_search_repos(
                    query=str(arguments.get("query", "")),
                    limit=int(arguments.get("limit", 5)),
                    offset=int(arguments.get("offset", 0)),
                )
                return _res(r, r.get("success", False))
            elif action == "fetch":
                r = await codewiki_fetch_repo(
                    owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                )
                return _res(r, r.get("success", False))
            elif action == "ask":
                owner = str(arguments.get("owner", ""))
                repo = str(arguments.get("repo", ""))
                question = str(arguments.get("question", ""))
                try:
                    r = await asyncio.wait_for(deepwiki_ask(owner, repo, question), timeout=60)
                except asyncio.TimeoutError:
                    r = {"success": False, "error": "DeepWiki timeout"}
                else:
                    if r.get("success"):
                        return _res(r)
                r2 = await codewiki_ask_repo(owner, repo, question)
                return _res(r2, r2.get("success", False))
            elif action == "architecture":
                r = await deepwiki_fetch(
                    owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    wiki_name=str(arguments.get("wiki_name", "")),
                )
                return _res(r, r.get("success", False))
            else:
                return _res({"error": f"unknown wiki action: {action}"}, False)

        elif name == "search_package":
            r = await search_package(
                name=str(arguments.get("name", "")),
                registry=str(arguments.get("registry", "auto")),
                type=str(arguments.get("type", "")),
            )
            return _res(r, r.get("success", False))

        elif name == "so_search":
            action = str(arguments.get("action", "stackexchange"))
            if action == "sofa":
                r = await search_sofa(
                    query=str(arguments.get("query", "")),
                    count=int(arguments.get("count", 5)),
                    content_type=str(arguments.get("content_type", "question")),
                    page=int(arguments.get("page", 1)),
                    post_id=str(arguments.get("post_id", "")),
                    steering=str(arguments.get("steering", "")),
                )
            else:
                r = await search_so(
                    query=str(arguments.get("query", "")),
                    count=int(arguments.get("count", 5)),
                    tags=str(arguments.get("tags", "")),
                    accepted=arguments.get("accepted") if arguments.get("accepted") is not None else None,
                    fromdate=str(arguments.get("fromdate", "")),
                    todate=str(arguments.get("todate", "")),
                    closed=arguments.get("closed") if arguments.get("closed") is not None else None,
                    sort=str(arguments.get("sort", "relevance")),
                    views=int(arguments.get("views", 0)),
                    answers=int(arguments.get("answers", 0)),
                    type=str(arguments.get("type", "search")),
                    site=str(arguments.get("site", "stackoverflow")),
                    question_id=int(arguments.get("question_id", 0)),
                    page=int(arguments.get("page", 1)),
                )
            return _res(r, r.get("success", False))

        elif name == "hn":
            action = str(arguments.get("action", ""))
            if action == "item":
                r = await hn_get_item(int(arguments.get("item_id", 0)))
            elif action == "stories":
                r = await hn_firebase_stories(story_type=str(arguments.get("firebase_type", "top")),
                    count=int(arguments.get("count", 10)))
            elif action == "search":
                r = await search_hn(
                    query=str(arguments.get("query", "")),
                    count=int(arguments.get("count", 5)),
                    sort_by_date=bool(arguments.get("sort_by_date", False)),
                    tags=str(arguments.get("tags", "story")),
                    min_points=int(arguments.get("min_points", 0)),
                    min_comments=int(arguments.get("min_comments", 0)),
                    before=int(arguments.get("before", 0)),
                    after=int(arguments.get("after", 0)),
                )
            else:
                return _res({"error": f"unknown hn action: {action}"}, False)
            return _res(r, r.get("success", False))

        elif name == "search_libraries":
            q = str(arguments.get("query", ""))
            n = str(arguments.get("name", ""))
            sort = str(arguments.get("sort", ""))
            languages = str(arguments.get("languages", ""))
            licenses = str(arguments.get("licenses", ""))
            keywords = str(arguments.get("keywords", ""))
            if q and not n:
                r = await libraries_io_search(q, platform=str(arguments.get("platform", "")),
                                              sort=sort, languages=languages, licenses=licenses, keywords=keywords)
            else:
                r = await search_libraries_io(n or q, platform=str(arguments.get("platform", "")))
            return _res(r, r.get("success", False))

        elif name == "vulns":
            action = str(arguments.get("action", ""))
            if action == "detail":
                r = await get_vulnerability_detail(vuln_id=str(arguments.get("vuln_id", "")))
            elif action == "latest_version":
                r = await get_component_latest_version(purl=str(arguments.get("purl", "")))
            else:
                r = await scan_vulnerabilities(
                    platform=str(arguments.get("platform", "")),
                    name=str(arguments.get("name", "")),
                    version=str(arguments.get("version", "")),
                    coordinates=str(arguments.get("coordinates", "")),
                )
            return _res(r, r.get("success", False))

        elif name == "github":
            action = str(arguments.get("action", ""))
            owner = str(arguments.get("owner", ""))
            repo = str(arguments.get("repo", ""))
            if action == "readme":
                r = await fetch_readme(owner=owner, repo=repo, branch=str(arguments.get("branch", "")))
            elif action == "contents":
                r = await gh_get_contents(owner=owner, repo=repo, path=str(arguments.get("path", "")),
                                          branch=str(arguments.get("branch", "")))
            elif action == "languages":
                r = await gh_get_languages(owner=owner, repo=repo)
            elif action == "topics":
                r = await gh_get_topics(owner=owner, repo=repo)
            elif action == "releases":
                r = await gh_get_releases(owner=owner, repo=repo, count=int(arguments.get("count", 5)))
            else:
                return _res({"error": f"unknown github action: {action}"}, False)
            return _res(r, r.get("success", False))

        elif name == "docs":
            action = str(arguments.get("action", ""))
            if action == "devdocs_list":
                r = await devdocs_list_docs()
            elif action == "devdocs_search":
                r = await devdocs_search(slug=str(arguments.get("slug", "")),
                                         query=str(arguments.get("query", "")))
            elif action == "devdocs_fetch":
                r = await devdocs_fetch(slug=str(arguments.get("slug", "")))
            elif action == "devdocs_fetch_content":
                r = await devdocs_fetch_content(slug=str(arguments.get("slug", "")),
                                                path=str(arguments.get("path", "")))
            elif action == "devdocs_meta":
                r = await devdocs_meta(slug=str(arguments.get("slug", "")))
            elif action == "rtd_info":
                r = await readthedocs_project_info(str(arguments.get("project", "")))
            elif action == "rtd_versions":
                r = await readthedocs_versions(str(arguments.get("project", "")))
            elif action == "rtd_search":
                r = await search_readthedocs(str(arguments.get("project", "")),
                                             str(arguments.get("query", "")),
                                             version=str(arguments.get("version", "")))
            else:
                return _res({"error": f"unknown docs action: {action}"}, False)
            return _res(r, r.get("success", False))

        elif name == "search_all":
            query = str(arguments.get("query", ""))
            library = str(arguments.get("library", ""))
            version = str(arguments.get("version", ""))
            owner = str(arguments.get("owner", ""))
            repo = str(arguments.get("repo", ""))
            language = str(arguments.get("language", ""))
            lib = library or (query.split()[0] if query.strip() else "")

            expanded = await expand_code_query(query)
            code_q = expanded[1] if len(expanded) > 1 else query

            tasks = []
            task_names = []

            tasks.append(context7_resolve(lib, version=version))
            task_names.append("context7")

            if GH_TOKEN:
                tasks.append(search_github(query, "code", 10, owner, repo, language))
                task_names.append("github")

            if owner and repo:
                tasks.append(deepwiki_fetch(owner, repo))
                task_names.append("deepwiki")

            tasks.append(codewiki_search_repos(query, 3))
            task_names.append("codewiki")

            tasks.append(search_so(query, 3, ""))
            task_names.append("so")

            if SOFA_KEY:
                tasks.append(search_sofa(query, 3))
                task_names.append("sofa")

            tasks.append(search_hn(query, 3))
            task_names.append("hn")

            if LI_KEY:
                tasks.append(libraries_io_search(query))
                task_names.append("libraries_io")

            tasks.append(npm_search(query, 5))
            task_names.append("npm")

            tasks.append(crates_search(query, 5))
            task_names.append("crates")

            tasks.append(devdocs_search(lib, query) if lib else devdocs_list_docs())
            task_names.append("devdocs")

            tasks.append(search_papers(query, 5))
            task_names.append("s2_papers")

            async def _firecrawl():
                key = await _next_fc_key()
                if not key:
                    return {"success": False, "results": []}
                body = {"query": code_q, "limit": 10, "categories": ["github"],
                        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}}
                try:
                    async with httpx.AsyncClient(timeout=15) as c:
                        r = await c.post(FIRECRAWL_SEARCH, json=body,
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json"})
                    if r.status_code != 200:
                        return {"success": False, "results": []}
                    data = r.json()
                    results = []
                    for item in (data.get("data", {}).get("web", []) or []):
                        results.append({"full_name": item.get("title","")[:120], "url": item.get("url",""),
                                        "snippet": (item.get("markdown","") or item.get("description",""))[:300]})
                    return {"success": True, "results": results}
                except Exception:
                    return {"success": False, "results": []}

            async def _tavily():
                key = await _next_tv_key()
                if not key:
                    return {"success": False, "results": []}
                try:
                    async with httpx.AsyncClient(timeout=15) as c:
                        r = await c.post(TAVILY_SEARCH,
                            json={"query": code_q, "search_depth": "basic", "max_results": 5,
                                  "include_answer": False, "topic": "general"},
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                    if r.status_code != 200:
                        return {"success": False, "results": []}
                    data = r.json()
                    results = []
                    for item in (data.get("results", []) or []):
                        results.append({"title": item.get("title","")[:120], "url": item.get("url",""),
                                        "snippet": (item.get("content","") or "")[:300]})
                    return {"success": True, "results": results}
                except Exception:
                    return {"success": False, "results": []}

            tasks.append(_firecrawl())
            task_names.append("firecrawl_github")
            tasks.append(_tavily())
            task_names.append("tavily")

            results = await asyncio.gather(*tasks, return_exceptions=True)
            merged: dict[str, Any] = {}
            flat_items: list[dict] = []
            for name_, result in zip(task_names, results):
                if isinstance(result, BaseException):
                    continue
                if not isinstance(result, dict) or not result.get("success"):
                    continue
                if name_ == "context7":
                    merged["docs"] = result["docs"]
                    merged["library"] = result["library"]
                    merged["docs_source"] = "context7"
                    for s in (result.get("docs", {}).get("snippets", []) or []):
                        flat_items.append({"source": "context7_doc", "title": s.get("title", ""), "text": s.get("content", ""), "url": s.get("url", "")})
                    for cs in (result.get("docs", {}).get("code_snippets", []) or []):
                        flat_items.append({"source": "context7_code", "title": cs.get("title", ""), "text": cs.get("code", ""), "language": cs.get("language", "")})
                elif name_ == "github":
                    merged["code_examples"] = result["results"]
                    for cr in (result.get("results", []) or []):
                        flat_items.append({"source": "github_code", "title": cr.get("file", ""), "text": cr.get("snippet", ""), "repo": cr.get("repo", ""), "url": cr.get("url", "")})
                elif name_ == "deepwiki":
                    arch_url = result.get("url") or f"https://deepwiki.com/{owner}/{repo}"
                    arch_content = result.get("content") or result.get("detail", "")
                    merged["architecture"] = {"url": arch_url, "content_preview": arch_content[:2000]}
                    flat_items.append({"source": "deepwiki", "title": f"architecture: {owner}/{repo}", "text": arch_content[:3000], "url": arch_url})
                elif name_ == "codewiki":
                    merged["ai_wikis"] = result["results"]
                    for wr in (result.get("results", []) or []):
                        flat_items.append({"source": "codewiki", "title": wr.get("full_name", ""), "text": wr.get("description", ""), "url": wr.get("url", "")})
                elif name_ == "so":
                    merged["stackoverflow"] = result["results"]
                    for sr in (result.get("results", []) or []):
                        flat_items.append({"source": "stackoverflow", "title": sr.get("title", ""), "text": f"{sr.get('body','')} {sr.get('top_answer','')}", "url": sr.get("url", "")})
                elif name_ == "sofa":
                    merged["sofa"] = result["results"]
                    if result.get("steering"):
                        merged["sofa_steering"] = result["steering"]
                    for sr in (result.get("results", []) or []):
                        flat_items.append({"source": "sofa", "title": sr.get("title", ""), "text": sr.get("body", ""), "url": sr.get("url", "")})
                elif name_ == "hn":
                    merged["hackernews"] = result["results"]
                    for hr in (result.get("results", []) or []):
                        flat_items.append({"source": "hackernews", "title": hr.get("title", ""), "text": hr.get("title", ""), "url": hr.get("url", "")})
                elif name_ == "libraries_io":
                    merged["libraries_io"] = result["results"]
                    for lr in (result.get("results", []) or []):
                        flat_items.append({"source": "libraries_io", "title": lr.get("name", ""), "text": lr.get("description", ""), "url": ""})
                elif name_ == "npm":
                    merged["npm"] = result["results"]
                    for nr in (result.get("results", []) or []):
                        flat_items.append({"source": "npm", "title": nr.get("name", ""), "text": nr.get("description", ""), "url": ""})
                elif name_ == "crates":
                    merged["crates"] = result["results"]
                    for cr in (result.get("results", []) or []):
                        flat_items.append({"source": "crates", "title": cr.get("name", ""), "text": cr.get("description", ""), "url": ""})
                elif name_ == "devdocs":
                    merged["devdocs"] = result.get("results", [])
                elif name_ == "s2_papers":
                    merged["semantic_scholar"] = result.get("results", [])
                    for pr in (result.get("results", []) or []):
                        flat_items.append({"source": "semantic_scholar", "title": pr.get("title", ""), "text": pr.get("abstract", ""), "url": pr.get("url", "")})
                elif name_ == "firecrawl_github":
                    merged["firecrawl_github"] = result.get("results", [])
                    for fr in (result.get("results", []) or []):
                        flat_items.append({"source": "firecrawl_github", "title": fr.get("title", fr.get("full_name", "")), "text": fr.get("snippet", ""), "url": fr.get("url", "")})
                elif name_ == "tavily":
                    merged["tavily"] = result.get("results", [])
                    for tr in (result.get("results", []) or []):
                        flat_items.append({"source": "tavily", "title": tr.get("title", ""), "text": tr.get("snippet", ""), "url": tr.get("url", "")})

            query_embed = await _embed([query], "query") if flat_items else None
            query_emb = query_embed[0] if query_embed else None
            if query_emb and flat_items:
                texts_to_embed = [fi.get("text", fi.get("title", ""))[:500] for fi in flat_items]
                embeds = await _embed(texts_to_embed, "passage")
                if embeds:
                    for fi, emb in zip(flat_items, embeds):
                        fi["_embedding"] = emb
                deduped = _dedup_rank(flat_items, query_emb)
                deduped = _hybrid_rank(deduped, query)
                merged["deduped_results"] = deduped
                merged["total_raw"] = len(flat_items)
                merged["total_deduped"] = len(deduped)
            if merged.get("deduped_results"):
                try:
                    merged["deduped_results"] = await _rerank(query, merged["deduped_results"], top_k=20)
                except Exception:
                    pass
            return _res(merged, bool(merged))

        elif name == "papers":
            action = str(arguments.get("action", ""))
            if action == "details":
                r = await get_paper_details(paper_id=str(arguments.get("paper_id", "")))
            elif action == "batch":
                raw = str(arguments.get("paper_ids", ""))
                ids = [x.strip() for x in raw.split(",") if x.strip()]
                r = await get_papers_batch(ids)
            elif action == "citations":
                r = await get_paper_citations(
                    paper_id=str(arguments.get("paper_id", "")),
                    limit=int(arguments.get("limit", 20)))
            elif action == "references":
                r = await get_paper_references(
                    paper_id=str(arguments.get("paper_id", "")),
                    limit=int(arguments.get("limit", 20)))
            else:
                r = await search_papers(
                    query=str(arguments.get("query", "")),
                    limit=int(arguments.get("limit", 10)),
                    year=str(arguments.get("year", "")),
                    fields_of_study=str(arguments.get("fields_of_study", "")),
                    open_access=bool(arguments.get("open_access", False)),
                    offset=int(arguments.get("offset", 0)),
                )
            return _res(r, r.get("success", False))

        else:
            return _res({"error": f"unknown tool: {name}"}, False)

    except ValueError as e:
        return _res({"error": str(e)}, False)
    except KeyError as e:
        return _res({"error": f"Missing required argument: {e}"}, False)
    except TypeError as e:
        return _res({"error": str(e)}, False)
    except RuntimeError as e:
        return _res({"error": str(e)}, False)
    except Exception as e:
        return _res({"error": f"{type(e).__name__}: {e}"}, False)


async def _warmup_reranker():
    """Pre-load reranker model in background to avoid cold-start delay."""
    try:
        from reranker import rerank
        await rerank("warmup", [{"title": "warmup", "snippet": "warmup"}], top_k=1)
    except Exception:
        pass


async def main():
    asyncio.create_task(_warmup_reranker())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
