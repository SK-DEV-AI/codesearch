from __future__ import annotations

import asyncio
import json
import logging
import re

from typing import Any

logger = logging.getLogger("codesearch")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from config import GH_TOKEN, SOFA_KEY, LI_KEY, close_http_client, get_http_client, _KeyRotator
from embed import _embed, _dedup_rank, _hybrid_rank
from code_expand import expand_code_query
from context7 import context7_resolve, search_llms_txt, context7_add_repo
from github_api import (search_github, fetch_readme, gh_get_contents, gh_get_languages,
    gh_get_topics, gh_get_releases, gh_get_repo, search_commits, gh_get_branches,
    gh_get_tags, gh_get_tree, search_labels, search_topics)
from deepwiki import deepwiki_fetch, deepwiki_ask
from codewiki import codewiki_fetch_repo, codewiki_search_repos, codewiki_ask_repo
from pkgseer import (
    search as pkgseer_search,
    code_files as pkgseer_code_files,
    code_read as pkgseer_code_read,
    code_grep as pkgseer_code_grep,
    pkg_deps as pkgseer_pkg_deps,
    jsdelivr_list_files,
    jsdelivr_read_file,
)
from analyze import analyze_repo
from searchcode import (
    analyze as searchcode_analyze,
    search as searchcode_search,
    file_tree as searchcode_file_tree,
    get_file as searchcode_get_file,
    findings as searchcode_findings,
)
from stack_exchange import (search_so, so_similar, so_tags_info, so_tags_wikis,
    get_questions_by_ids, search_users, search_tags, get_question_comments,
    get_answers_by_ids, get_questions_by_sort, get_users_by_ids)
from sofa import search_sofa
from hackernews import search_hn, hn_get_item, hn_firebase_stories, hn_get_user
from libraries_io import (search_libraries_io, libraries_io_search, get_versions,
    get_dependencies, get_dependents, get_github_repo, get_github_dependencies,
    li_list_platforms, li_list_licenses, li_keyword_projects)
from oss_index import (scan_vulnerabilities, get_vulnerability_detail, get_component_latest_version,
                       search_vulnerabilities, analyze_license, quick_component_report)
from readthedocs import (search_readthedocs, readthedocs_project_info, readthedocs_versions,
    readthedocs_translations, readthedocs_subprojects, readthedocs_builds)
from registries import (search_package, npm_search, crates_search, get_npm_versions,
    get_npm_time, get_npm_version, get_crates_versions,
    get_pypi_version, get_pypi_versions,
    npm_get_version, crates_get_version, crates_get_readme, crates_get_summary)
from devdocs import (devdocs_list_docs, devdocs_fetch, devdocs_fetch_content,
    devdocs_search, devdocs_meta, devdocs_toc)
from semantic_scholar import (search_papers, get_paper_details,
    get_papers_batch, get_paper_citations, get_paper_references,
    get_paper_recommendations, search_authors, get_author_papers,
    autocomplete_papers, s2_author_by_id, s2_bulk_search,
    s2_recommendations_with_negatives)
from core_api import search_core_works, CORE_API_AVAILABLE
from depsdev import get_resolved_dependencies, get_package_info as get_depsdev_package_info, get_advisory, query_by_hash
from reranker import rerank as _rerank
from tavily_search import tavily_search
from enrich import enrich_results
from pkg_utils import (get_pkg_changelog, get_pkg_upgrade_review,
    list_package_files, read_package_file, resolve_package)
from openalex import search_openalex, search_openalex_authors, search_openalex_concepts, search_openalex_institutions

server = Server("codesearch", instructions="""# CodeSearch MCP

Code search, package analysis, documentation, vulnerability scanning.

## Tools

**github**(action, owner, repo, ...) — GitHub API: search code/repos/issues, read files, tree, commits.

**docs**(query, library, fast) — Context7→ReadTheDocs→DevDocs fallback.

**wiki**(owner, repo, repos=[], question=ask) — DeepWiki + CodeWiki architecture Q&A.

**search_all**(query, language, owner, repo, count) — 17 sources: SO+SOFA+HN+GitHub+DeepWiki+CodeWiki+DevDocs+S2+CORE+arXiv+OpenAlex+libraries.io+npm+crates+Tavily+Context7 → dedup → hybrid rank → rerank.

**openalex**(query, action=works|authors|topics|institutions, count) — OpenAlex academic database. Free, no API key.

**papers**(query, fields_of_study, year, count) — S2 + CORE + arXiv with FoS filter.

**so_search**(action=stackexchange|sofa, tags, accepted, count) — Stack Exchange (accepted/resolved) or SOFA (community posts).

**hn**(action=search, tags=story|show|ask, min_points, count) — structured HN threads.

**searchcode**(action=analyze|search|findings|file_tree|get_file, repository, ...) — per-repo code intelligence via SearchCode. No auth, no rate limits. Unique: analyze (instant repo overview) and findings (code quality issues).

**code_search**(query, target=ns:lib, lang) — PkgSeer cross-ecosystem symbol/function search.

**code_files**(spec, path_prefix) → **code_read**(spec, path) — browse package source. jsDelivr-backed for npm (no auth, faster).

**code_grep**(spec, pattern, path_prefix) — grep indexed package source.

**pkg**(name, registry=auto, action=info|...|files|read) — composite: Libraries.io + Sonatype + PkgSeer.

**pkg_deps**(spec) — transitive deps + conflict detection.

**search_libraries**(name, platform) — Libraries.io metadata.

**vulns**(action=scan|detail|...|quick_report, name, version) — Sonatype scanning.

**search_package**(name, registry=auto) — raw registry queries (npm/PyPI/crates/deeps.dev). For composite use **pkg**.

**analyze**(repository) — deep repo analysis: parallel GitHub metadata + SearchCode + DeepWiki + CodeWiki in one call.

**enrich**(query, results, top_k, max_fetch_size) — fetch+NIM dedup+BM25+rerank for external results.

## Pipeline (search_all)

1. Groq code expansion: pseudo-code + API call + library term variants.
2. 17 parallel sources.
3. NIM nv-embedcode-7b-v1 dedup (source-clustered cosine).
4. BM25 hybrid rank (keyword/syntax + embedding).
5. gte-reranker (704 tokens).

## Params

- `language`: code/github/search_all — always pass known languages
- `tags`: SO queries — scope to ecosystem (python, react, rust)
- `owner`/`repo`: GitHub-specific queries
- `accepted=true`: so_search for resolved/verified answers
- `fields_of_study`: papers domain filter (Computer Science, Physics, etc.)
- `min_points`/`min_comments`: HN quality floor
- `count`: results (default 10, max 50 search_all)
""")
# Query->tag/platform/domain detection helpers for search_all precision
_SE_TAGS = re.compile(r"(?i)\b(react|typescript|javascript|python|rust|golang?|docker|kubernetes|postgresql|mysql|sql|aws|git|node\.?js|angular|vue|django|flask|fastapi|spring|jvm|scala|kotlin|swift|ruby|rails|php|laravel|lua|c\+\+|csharp|dotnet|unity|unreal|tensorflow|pytorch|jax|linux|bash|shell|nix|nixos|ansible|terraform|graphql|rest|grpc|websocket|redis|mongodb|sqlite|svelte|next\.?js|nuxt|deno|bun)\b")
_LI_PLATFORM = re.compile(r"(?i)\b(python|javascript|typescript|rust|golang?|java|ruby|php|swift|kotlin|lua|c\+\+|csharp|dart|elixir|haskell|scala|perl|r)\b")
_LI_PLATFORM_MAP = {"python":"pypi","javascript":"npm","typescript":"npm","rust":"cargo","golang":"go","java":"maven","ruby":"rubygems","php":"packagist","swift":"swift","kotlin":"maven","csharp":"nuget","dart":"pub","elixir":"hex","haskell":"hackage","scala":"maven","perl":"cpan"}


_warmup_task: asyncio.Task | None = None


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


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="github",
            description="GitHub operations: search code/repos/issues/users/commits, repo readme/contents/languages/topics/releases/metadata/branches/tags/file-tree, issue/PR detail with comments/reviews, user/org profiles, community health, SBOM. Use start_line/end_line with contents action for targeted source reads. e.g. github(action='issue', owner='sst', repo='opencode', issue_number=42)",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "readme", "contents", "languages", "topics", "releases", "repo", "commits", "branches", "tags", "tree", "search_labels", "search_topics", "issue", "pr", "pr_reviews", "user"], "default": "search"},
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
                    "in_qualifier": {"type": "string"},
                    "exclude_qualifier": {"type": "string"},
                    "merged": {"type": "string"},
                    "head": {"type": "string"},
                    "base": {"type": "string"},
                    "review": {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-based start line for contents action (slices file content by newline)"},
                    "end_line": {"type": "integer", "description": "1-based end line (inclusive) for contents action"},
                    "author": {"type": "string", "description": "Author to filter by (commits action)"},
                    "tree_sha": {"type": "string", "default": "HEAD", "description": "Tree SHA or HEAD for tree action"},
                    "recursive": {"type": "boolean", "default": True, "description": "Recursive tree for tree action"},
                    "branch": {"type": "string", "description": "Branch name (readme/contents actions)"},
                    "issue_number": {"type": "integer", "description": "Issue/PR number (issue, pr, pr_reviews actions)"},
                    "pr_number": {"type": "integer", "description": "PR number (pr, pr_reviews actions)"},
                    "username": {"type": "string", "description": "GitHub username (user action)"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="wiki",
            description="Repo architecture and wiki via DeepWiki + CodeWiki. For single-repo queries use owner+repo; for multi-repo questions pass repos (array of owner/repo strings, max 10). e.g. wiki(owner='torvalds', repo='linux', question='How does the scheduler work?')",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "fetch", "ask", "architecture"]},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "repos": {"type": "array", "items": {"type": "string"}, "description": "Array of owner/repo strings for multi-repo queries (DeepWiki ask supports up to 10)"},
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
            description="Unified search across 14+ sources with embedding dedup and reranking. e.g. search_all(query='rust async runtime', language='rust')",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "library": {"type": "string"},
                    "version": {"type": "string"},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "language": {"type": "string"},
                    "count": {"type": "integer", "default": 10, "description": "Results per source (max 50)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="openalex",
            description="Search OpenAlex academic research database — papers, authors, institutions, and research topics. Free, no API key needed. e.g. openalex(query='transformer attention', action='works')",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "action": {"type": "string", "description": "Search entity type", "enum": ["works", "authors", "topics", "institutions"]},
                    "count": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_package",
            description="Raw package registry queries (npm/PyPI/crates + deps.dev). Use for fast single-source lookups. For composite intelligence (Libraries.io + Sonatype vulns), use `pkg` instead. e.g. search_package(name='express', registry='npm')",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "registry": {"type": "string", "default": "auto"},
                    "action": {"type": "string", "description": "npm_dist_tags|npm_versions|npm_time|npm_get_version|pypi_versions|pypi_get_version|crates_downloads|crates_reverse_deps|crates_owners|crates_categories|crates_keywords|crates_versions|crates_get_version|crates_get_readme|crates_summary|depsdev_dependencies|depsdev_info|depsdev_advisory|depsdev_query"},
                    "version": {"type": "string", "description": "Package version (required for version-specific queries)"},
                    "advisory_id": {"type": "string", "description": "OSV advisory ID for depsdev_advisory"},
                    "hash_type": {"type": "string", "description": "Hash type for depsdev_query: SHA1, SHA256, etc"},
                    "hash_value": {"type": "string", "description": "Base64-encoded hash value for depsdev_query"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="so_search",
            description="Stack Overflow: Stack Exchange API (free) or SOFA (requires SOFA_KEY). e.g. so_search(query='python async', tags='python', accepted=true)",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["stackexchange", "sofa"], "default": "stackexchange"},
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                    "tags": {"type": "string"},
                    "accepted": {"type": "boolean"},
                    "closed": {"type": "boolean"},
                    "sort": {"type": "string", "default": "relevance"},
                    "type": {"type": "string", "default": "search", "enum": ["search", "excerpts", "faq", "answers", "similar", "tags_info", "tags_wikis", "questions_by_ids", "search_users", "search_tags", "question_comments", "questions", "answers_by_ids", "users_by_ids"]},
                    "site": {"type": "string", "default": "stackoverflow"},
                    "question_id": {"type": "integer"},
                    "ids": {"type": "string", "description": "Comma-separated IDs for questions_by_ids / answers_by_ids / users_by_ids"},
                    "page": {"type": "integer", "default": 1, "description": "Page number for pagination"},
                    "fromdate": {"type": "string", "description": "Unix timestamp or date string for earliest creation date"},
                    "todate": {"type": "string", "description": "Unix timestamp or date string for latest creation date"},
                    "views": {"type": "integer", "default": 0, "description": "Minimum view count"},
                    "answers": {"type": "integer", "default": 0, "description": "Minimum answer count"},
                    "content_type": {"type": "string", "default": "question", "description": "SOFA only: question|til|blueprint|playbook"},
                    "post_id": {"type": "string", "description": "SOFA only: get post by ID instead of searching"},
                    "steering": {"type": "string", "description": "SOFA only: curation steering parameter"},
                },
            },
        ),
        Tool(
            name="hn",
            description="Hacker News: search, item detail, user profile, or story lists (top/new/best/ask/show). e.g. hn(query='rust', tags='story', min_points=50)",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search","item","stories","user"]},
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                    "sort_by_date": {"type": "boolean"},
                    "tags": {"type": "string", "default": "story", "description": "story|comment|poll|front_page|ask_hn|show_hn"},
                    "min_points": {"type": "integer", "default": 0},
                    "min_comments": {"type": "integer", "default": 0},
                    "item_id": {"type": "integer"},
                    "firebase_type": {"type": "string"},
                    "username": {"type": "string"},
                },
                "required": [],
            },
        ),
        Tool(
            name="search_libraries",
            description="Libraries.io dependency metadata and source rank. e.g. search_libraries(name='lodash', platform='npm')",
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
                    "action": {"type": "string", "description": "versions|dependencies|dependents|github_repo|github_dependencies|platforms|licenses|keywords"},
                    "version": {"type": "string"},
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "keyword": {"type": "string", "description": "Keyword for keywords action"},
                },
            },
        ),
        Tool(
            name="vulns",
            description="Sonatype Guide: scan packages, vulnerability details, latest version, search, license analysis, or quick report by PURL. e.g. vulns(action='scan', name='lodash', version='4.17.20')",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["scan", "detail", "latest_version", "search", "license", "quick_report"]},
                    "platform": {"type": "string"},
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "coordinates": {"type": "string"},
                    "vuln_id": {"type": "string"},
                    "purl": {"type": "string"},
                    "keyword": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        ),
        Tool(
            name="docs",
            description="Documentation operations. Default action runs smart fallback (Context7→ReadTheDocs→llms.txt→DevDocs). After finding docs, use fetch(url) to read full pages or devdocs_fetch_content for structured content. e.g. docs(query='authentication', library='express')",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "devdocs_list", "devdocs_search", "devdocs_fetch", "devdocs_fetch_content", "devdocs_meta", "devdocs_toc", "context7_add_repo", "rtd_info", "rtd_versions", "rtd_search", "rtd_translations", "rtd_subprojects", "rtd_builds"], "default": "search"},
                    "query": {"type": "string"},
                    "library": {"type": "string"},
                    "library_id": {"type": "string", "description": "Context7 library ID (skip search)"},
                    "version": {"type": "string"},
                    "fast": {"type": "boolean"},
                    "slug": {"type": "string"},
                    "path": {"type": "string"},
                    "project": {"type": "string"},
                    "provider": {"type": "string", "description": "Git provider for context7_add_repo (github, gitlab)"},
                    "repo_url": {"type": "string", "description": "Repository URL for context7_add_repo"},
                },
            },
        ),
        Tool(
            name="papers",
            description="Academic papers from Semantic Scholar, CORE API, and arXiv. Actions: search, details, batch, citations, references, recommendations, author_search, author_papers, autocomplete, core_search, arxiv_search. e.g. papers(query='transformer attention', fields_of_study='Computer Science')",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "details", "batch", "citations", "references", "recommendations", "author_search", "author_papers", "autocomplete", "core_search", "arxiv_search", "author_by_id", "bulk_search", "recommendations_negatives"]},
                    "query": {"type": "string"},
                    "paper_id": {"type": "string"},
                    "paper_ids": {"type": "string", "description": "Comma-separated paper IDs for batch action"},
                    "author_id": {"type": "string"},
                    "count": {"type": "integer", "default": 10},
                    "year": {"type": "string"},
                    "fields_of_study": {"type": "string"},
                    "open_access": {"type": "boolean"},
                    "positive_ids": {"type": "string", "description": "Comma-separated positive paper IDs for recommendations_negatives"},
                    "negative_ids": {"type": "string", "description": "Comma-separated negative paper IDs for recommendations_negatives"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="searchcode",
            description="Per-repo code intelligence via SearchCode (free, no auth). Use analyze for instant repo overview (languages, complexity, tech stack, credentials), search for searching code within a repo, findings for code quality issues, file_tree to list files, or get_file to read a file. No rate limits. e.g. searchcode(action='analyze', repository='https://github.com/expressjs/express')",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["analyze", "search", "findings", "file_tree", "get_file"], "description": "analyze=repo overview, search=search code, findings=quality issues, file_tree=list files, get_file=read file"},
                    "repository": {"type": "string", "description": "Git URL of a public remote repository (e.g. https://github.com/expressjs/express)"},
                    "query": {"type": "string", "description": "Search query (for search action)"},
                    "path": {"type": "string", "description": "Subdirectory path filter (for analyze/findings/file_tree)"},
                    "language": {"type": "string", "description": "Language filter (for analyze action)"},
                    "detail_level": {"type": "string", "enum": ["summary", "full"], "default": "summary", "description": "Response verbosity (for analyze action)"},
                    "symbol_name": {"type": "string", "description": "Function/class name to extract (for get_file action)"},
                    "case_sensitive": {"type": "boolean", "default": False, "description": "Case-sensitive search (for search action)"},
                    "context_lines": {"type": "integer", "default": 2, "description": "Context lines around matches (for search action)"},
                    "max_results": {"type": "integer", "default": 10, "description": "Max results (for search/findings)"},
                    "severity": {"type": "string", "enum": ["error", "warning", "info"], "description": "Filter by severity (for findings action)"},
                    "category": {"type": "string", "enum": ["security", "deprecated", "safety", "correctness", "maintainability", "accessibility", "modernization", "performance", "concurrency"], "description": "Filter findings by category (for findings action)"},
                    "start_line": {"type": "integer", "default": 1, "description": "Start line (for get_file without symbol_name)"},
                    "end_line": {"type": "integer", "description": "End line (for get_file without symbol_name)"},
                },
            },
        ),
        Tool(
            name="code_search",
            description="Search code, docs, and symbols across indexed dependencies and repositories. Supports qualifiers like kind:, category:, lang:, and package-scoped targets (npm:express, pypi:requests). Powered by PkgSeer. e.g. code_search(query='handleAuth', target='npm:express')",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query supporting implicit AND, OR, parens, -exclude, and qualifiers (kind:, lang:, path:)"},
                    "target": {"type": "string", "description": "Search scope: registry:name[@version] (e.g. npm:express) or github:org/repo"},
                    "source": {"type": "string", "enum": ["docs", "code", "symbol"], "description": "Restrict results to a specific source type"},
                    "lang": {"type": "string", "description": "Programming language filter"},
                    "limit": {"type": "integer", "default": 10, "description": "Max results (1-100)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="code_files",
            description="List files in an indexed dependency by package-scoped path (e.g. npm:express/src/). No GitHub URL needed. PkgSeer + jsDelivr (npm is jsDelivr-backed: 3-5x faster, no auth). e.g. code_files(spec='npm:express')",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {"type": "string", "description": "Package spec: registry:name[@version] (e.g. npm:express, pypi:requests)"},
                    "path_prefix": {"type": "string", "description": "Optional path prefix filter (e.g. src/)"},
                },
                "required": ["spec"],
            },
        ),
        Tool(
            name="code_read",
            description="Read a file from an indexed dependency by package-scoped path. No GitHub URL needed. PkgSeer + jsDelivr (npm is jsDelivr-backed: CDN-served, no auth). e.g. code_read(spec='npm:express', path='src/index.js')",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {"type": "string", "description": "Package spec: registry:name[@version] (e.g. npm:express)"},
                    "path": {"type": "string", "description": "File path within the package (e.g. src/index.js)"},
                },
                "required": ["spec", "path"],
            },
        ),
        Tool(
            name="code_grep",
            description="Grep through indexed dependency source for a text pattern. No clone needed. Powered by PkgSeer. e.g. code_grep(spec='npm:express', pattern='handleAuth')",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {"type": "string", "description": "Package spec: registry:name[@version] (e.g. npm:express)"},
                    "pattern": {"type": "string", "description": "Text pattern to search for"},
                    "path_prefix": {"type": "string", "description": "Optional path prefix to narrow the search"},
                },
                "required": ["spec", "pattern"],
            },
        ),
        Tool(
            name="pkg_deps",
            description="Analyze transitive dependencies for a package with conflict detection across 8+ registries. Powered by PkgSeer. e.g. pkg_deps(spec='npm:express')",
            inputSchema={
                "type": "object",
                "properties": {
                    "spec": {"type": "string", "description": "Package spec: registry:name[@version] (e.g. npm:express)"},
                },
                "required": ["spec"],
            },
        ),
        Tool(
            name="pkg",
            description="Package intelligence: info (composite metadata — Libraries.io + Sonatype + PkgSeer), changelog (release notes), upgrade_review (vulns+changelog+deps diff between versions), files (list source files), read (read a source file). For raw single-source registry queries (npm versions, crates categories), use `search_package` instead. e.g. pkg(name='express', action='info')",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["info", "changelog", "upgrade_review", "files", "read"], "default": "info"},
                    "name": {"type": "string"},
                    "registry": {"type": "string", "default": "auto"},
                    "version": {"type": "string", "description": "Package version (default: latest)"},
                    "current_version": {"type": "string", "description": "Current version for upgrade_review"},
                    "target_version": {"type": "string", "description": "Target version for upgrade_review"},
                    "from_version": {"type": "string", "description": "Start of version range for changelog (optional)"},
                    "to_version": {"type": "string", "description": "End of version range for changelog (optional)"},
                    "path": {"type": "string", "description": "File path filter (files) or exact path (read)"},
                    "count": {"type": "integer", "default": 10, "description": "Max changelog entries"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="enrich",
            description="Fetch full content for a list of search results, deduplicate by embedding, and rerank by relevance. Give it results from any search tool plus the original query. e.g. enrich(query='rust async', results=[{'url':'...','title':'...'}])",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "text": {"type": "string"},
                                "snippet": {"type": "string"},
                                "source": {"type": "string"},
                            },
                        },
                        "description": "List of result objects to enrich (url, title, text/snippet, source)",
                    },
                    "top_k": {"type": "integer", "default": 20, "description": "Max enriched results to return"},
                    "max_fetch_size": {"type": "integer", "default": 50, "description": "Max results to fetch content for"},
                    "include_html": {"type": "boolean", "default": False, "description": "Include raw HTML in content (default strips to text)"},
                },
                "required": ["query", "results"],
            },
        ),
        Tool(
            name="analyze",
            description="Deep repo analysis: parallel-fetch GitHub metadata + SearchCode code quality + DeepWiki architecture + CodeWiki sections in one call. Accepts GitHub URLs or owner/repo strings. e.g. analyze(repository='sst/opencode')",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository": {"type": "string", "description": "GitHub repo: URL (https://github.com/owner/repo) or owner/repo string"},
                },
                "required": ["repository"],
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

    def _res(data, ok=True):
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(data, default=str))],
            isError=not ok,
        )

    try:
        if name == "github":
            action = str(arguments.get("action", "search"))
            if action == "search":
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
                    in_qualifier=str(arguments.get("in_qualifier", "")),
                    exclude_qualifier=str(arguments.get("exclude_qualifier", "")),
                    merged=str(arguments.get("merged", "")),
                    head=str(arguments.get("head", "")),
                    base=str(arguments.get("base", "")),
                    review=str(arguments.get("review", "")),
                )
            elif action == "readme":
                r = await fetch_readme(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    branch=str(arguments.get("branch", "")))
            elif action == "contents":
                r = await gh_get_contents(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    path=str(arguments.get("path", "")),
                    branch=str(arguments.get("branch", "")))
                if r.get("success") and r.get("content"):
                    all_lines = r["content"].split("\n")
                    r["total_lines"] = len(all_lines)
                    r["total_chars"] = len(r["content"])
                    start_line = int(arguments.get("start_line", 0))
                    end_line = int(arguments.get("end_line", 0))
                    if start_line > 0:
                        if end_line > 0:
                            r["content"] = "\n".join(all_lines[start_line - 1:end_line])
                        else:
                            r["content"] = "\n".join(all_lines[start_line - 1:])
                        r["returned_lines"] = r["content"].count("\n") + 1
            elif action == "languages":
                r = await gh_get_languages(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")))
            elif action == "topics":
                r = await gh_get_topics(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")))
            elif action == "releases":
                r = await gh_get_releases(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    count=int(arguments.get("count", 5)))
            elif action == "repo":
                r = await gh_get_repo(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")))
            elif action == "commits":
                r = await search_commits(query=str(arguments.get("query", "")),
                    count=int(arguments.get("count", 10)),
                    owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    author=str(arguments.get("author", "")),
                    sort=str(arguments.get("sort", "")),
                    order=str(arguments.get("order", "")),
                    page=int(arguments.get("page", 1)))
            elif action == "branches":
                r = await gh_get_branches(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")))
            elif action == "tags":
                r = await gh_get_tags(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")))
            elif action == "tree":
                r = await gh_get_tree(owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    tree_sha=str(arguments.get("tree_sha", "HEAD")),
                    recursive=bool(arguments.get("recursive", True)))
            elif action == "search_labels":
                r = await search_labels(query=str(arguments.get("query","")),
                    count=int(arguments.get("count",10)))
            elif action == "search_topics":
                r = await search_topics(query=str(arguments.get("query","")),
                    count=int(arguments.get("count",10)))
            elif action == "issue":
                r = await gh_get_issue(
                    owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    issue_number=int(arguments.get("issue_number", 0)))
            elif action == "pr":
                r = await gh_get_pr(
                    owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    pr_number=int(arguments.get("pr_number", 0)))
            elif action == "pr_reviews":
                r = await gh_get_pr_reviews(
                    owner=str(arguments.get("owner", "")),
                    repo=str(arguments.get("repo", "")),
                    pr_number=int(arguments.get("pr_number", 0)))
            elif action == "user":
                r = await gh_get_user(
                    username=str(arguments.get("username", "")))
            else:
                return _res({"error": f"unknown github action: {action}"}, False)
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
                repos = arguments.get("repos")
                if isinstance(repos, list) and repos:
                    deep_repos = repos[:10]
                    fallback_repo = repos[0]
                    fallback_owner = ""
                    fallback_repo_name = ""
                    if "/" in fallback_repo:
                        parts = fallback_repo.split("/", 1)
                        fallback_owner, fallback_repo_name = parts[0], parts[1]
                    else:
                        fallback_owner, fallback_repo_name = owner, fallback_repo
                else:
                    deep_repos = None
                    fallback_owner, fallback_repo_name = owner, repo
                try:
                    r = await asyncio.wait_for(deepwiki_ask(owner=owner, repo=repo, question=question, repos=deep_repos), timeout=60)
                except asyncio.TimeoutError:
                    r = {"success": False, "error": "DeepWiki timeout"}
                else:
                    if r.get("success"):
                        return _res(r)
                r2 = await codewiki_ask_repo(fallback_owner, fallback_repo_name, question)
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
            action_type = str(arguments.get("action", ""))
            pkg_name = str(arguments.get("name", ""))
            if action_type == "npm_get_version":
                r = await npm_get_version(name=pkg_name, version=str(arguments.get("version","")))
            elif action_type == "crates_get_version":
                r = await crates_get_version(name=pkg_name, version=str(arguments.get("version","")))
            elif action_type == "pypi_get_version":
                r = await get_pypi_version(name=pkg_name, version=str(arguments.get("version","")))
            elif action_type == "crates_get_readme":
                r = await crates_get_readme(name=pkg_name, version=str(arguments.get("version","")))
            elif action_type == "crates_summary":
                r = await crates_get_summary()
            elif action_type == "depsdev_dependencies":
                parts = pkg_name.split("/", 1)
                system = parts[0] if len(parts) > 1 else "npm"
                pkg = parts[1] if len(parts) > 1 else pkg_name
                r = await get_resolved_dependencies(system, pkg, str(arguments.get("version", "")))
            elif action_type == "depsdev_info":
                parts = pkg_name.split("/", 1)
                system = parts[0] if len(parts) > 1 else "npm"
                pkg = parts[1] if len(parts) > 1 else pkg_name
                r = await get_depsdev_package_info(system, pkg)
            elif action_type == "depsdev_advisory":
                r = await get_advisory(advisory_id=str(arguments.get("advisory_id","")))
            elif action_type == "depsdev_query":
                r = await query_by_hash(
                    hash_type=str(arguments.get("hash_type","SHA256")),
                    hash_value=str(arguments.get("hash_value","")))
            else:
                r = await search_package(
                    name=pkg_name,
                    registry=str(arguments.get("registry", "auto")),
                    type=action_type,
                )
            return _res(r, r.get("success", False))

        elif name == "pkg":
            action = str(arguments.get("action", "info"))
            pkg_name = str(arguments.get("name", ""))
            registry = str(arguments.get("registry", "auto"))
            if action == "changelog":
                r = await get_pkg_changelog(
                    name=pkg_name, registry=registry,
                    from_version=str(arguments.get("from_version", "")),
                    to_version=str(arguments.get("to_version", "")),
                    count=int(arguments.get("count", 10)))
            elif action == "upgrade_review":
                r = await get_pkg_upgrade_review(
                    name=pkg_name, registry=registry,
                    current_version=str(arguments.get("current_version", "")),
                    target_version=str(arguments.get("target_version", "")))
            elif action == "files":
                r = await list_package_files(
                    name=pkg_name, registry=registry,
                    version=str(arguments.get("version", "")),
                    path_filter=str(arguments.get("path", "")))
            elif action == "read":
                r = await read_package_file(
                    name=pkg_name, registry=registry,
                    path=str(arguments.get("path", "")),
                    version=str(arguments.get("version", "")))
            else:  # info
                pkg_info = await resolve_package(registry, pkg_name)
                r = pkg_info
                if pkg_info.get("success"):
                    try:
                        from libraries_io import search_libraries
                        lib_info = await search_libraries(
                            name=pkg_name, platform=pkg_info["registry"])
                        if lib_info.get("success"):
                            r["libraries_io"] = lib_info
                    except ImportError:
                        pass
                    try:
                        from oss_index import scan_vulnerabilities
                        vuln_info = await scan_vulnerabilities(
                            platform=pkg_info["registry"], name=pkg_name)
                        if vuln_info.get("success"):
                            r["vulnerabilities"] = len(
                                vuln_info.get("reports", [{}])[0].get("vulnerabilities", []))
                    except ImportError:
                        pass
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
            elif action == "questions_by_ids":
                ids_str = str(arguments.get("ids", ""))
                ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
                r = await get_questions_by_ids(ids, site=str(arguments.get("site", "stackoverflow")))
            elif action == "search_users":
                r = await search_users(
                    query=str(arguments.get("query", "")),
                    site=str(arguments.get("site", "stackoverflow")),
                    count=int(arguments.get("count", 10)),
                )
            elif action == "search_tags":
                r = await search_tags(
                    query=str(arguments.get("query", "")),
                    site=str(arguments.get("site", "stackoverflow")),
                    count=int(arguments.get("count", 10)),
                )
            elif action == "question_comments":
                r = await get_question_comments(
                    question_id=int(arguments.get("question_id", 0)),
                    site=str(arguments.get("site", "stackoverflow")),
                    count=int(arguments.get("count", 20)),
                )
            elif action == "questions":
                r = await get_questions_by_sort(sort=str(arguments.get("sort","hot")),
                    tagged=str(arguments.get("tags","")),
                    site=str(arguments.get("site","stackoverflow")),
                    count=int(arguments.get("count",10)))
            elif action == "answers":
                ids_raw = str(arguments.get("ids",""))
                ids_list = [int(x) for x in ids_raw.split(",") if x.strip().isdigit()]
                r = await get_answers_by_ids(ids_list, site=str(arguments.get("site","stackoverflow")))
            elif action == "users":
                ids_raw = str(arguments.get("ids",""))
                ids_list = [int(x) for x in ids_raw.split(",") if x.strip().isdigit()]
                r = await get_users_by_ids(ids_list, site=str(arguments.get("site","stackoverflow")))
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
            elif action == "user":
                r = await hn_get_user(str(arguments.get("username", "")))
            elif action == "search":
                r = await search_hn(
                    query=str(arguments.get("query", "")),
                    count=int(arguments.get("count", 5)),
                    sort_by_date=bool(arguments.get("sort_by_date", False)),
                    tags=str(arguments.get("tags", "story")),
                    min_points=int(arguments.get("min_points", 0)),
                    min_comments=int(arguments.get("min_comments", 0)),
                )
            else:
                return _res({"error": f"unknown hn action: {action}"}, False)
            return _res(r, r.get("success", False))

        elif name == "search_libraries":
            action = str(arguments.get("action", ""))
            q = str(arguments.get("query", ""))
            n = str(arguments.get("name", ""))
            platform = str(arguments.get("platform", ""))
            if action == "platforms":
                r = await li_list_platforms(count=int(arguments.get("count",50)))
            elif action == "licenses":
                r = await li_list_licenses(count=int(arguments.get("count",50)))
            elif action == "keywords":
                r = await li_keyword_projects(keyword=str(arguments.get("keyword","")),
                    count=int(arguments.get("count",10)))
            elif action == "versions":
                r = await get_versions(platform, n)
            elif action == "dependencies":
                r = await get_dependencies(platform, n, version=str(arguments.get("version", "")))
            elif action == "dependents":
                r = await get_dependents(platform, n)
            elif action == "github_repo":
                r = await get_github_repo(str(arguments.get("owner", "")), str(arguments.get("repo", "")))
            elif action == "github_dependencies":
                r = await get_github_dependencies(str(arguments.get("owner", "")), str(arguments.get("repo", "")))
            elif q and not n:
                sort = str(arguments.get("sort", ""))
                languages = str(arguments.get("languages", ""))
                licenses = str(arguments.get("licenses", ""))
                keywords = str(arguments.get("keywords", ""))
                r = await libraries_io_search(q, platform=platform,
                                              sort=sort, languages=languages, licenses=licenses, keywords=keywords)
            else:
                r = await search_libraries_io(n or q, platform=platform)
            return _res(r, r.get("success", False))

        elif name == "vulns":
            action = str(arguments.get("action", ""))
            if action == "detail":
                r = await get_vulnerability_detail(vuln_id=str(arguments.get("vuln_id", "")))
            elif action == "latest_version":
                r = await get_component_latest_version(purl=str(arguments.get("purl", "")))
            elif action == "search":
                r = await search_vulnerabilities(
                    keyword=str(arguments.get("keyword", "")),
                    limit=int(arguments.get("limit", 10)),
                )
            elif action == "license":
                r = await analyze_license(purl=str(arguments.get("purl", "")))
            elif action == "quick_report":
                r = await quick_component_report(purl=str(arguments.get("purl", "")))
            else:
                r = await scan_vulnerabilities(
                    platform=str(arguments.get("platform", "")),
                    name=str(arguments.get("name", "")),
                    version=str(arguments.get("version", "")),
                    coordinates=str(arguments.get("coordinates", "")),
                )
            return _res(r, r.get("success", False))

        elif name == "docs":
            action = str(arguments.get("action", "search"))
            if action == "search":
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
                    "hint": "Try wiki, github, or fetch(url) for this library",
                })
            elif action == "devdocs_list":
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
            elif action == "devdocs_toc":
                r = await devdocs_toc(doc=str(arguments.get("slug","")),
                    version=str(arguments.get("version","")))
            elif action == "context7_add_repo":
                r = await context7_add_repo(provider=str(arguments.get("provider","github")),
                    repo_url=str(arguments.get("repo_url","")))
            elif action == "rtd_info":
                r = await readthedocs_project_info(str(arguments.get("project", "")))
            elif action == "rtd_versions":
                r = await readthedocs_versions(str(arguments.get("project", "")))
            elif action == "rtd_search":
                r = await search_readthedocs(str(arguments.get("project", "")),
                                             str(arguments.get("query", "")),
                                             version=str(arguments.get("version", "")))
            elif action == "rtd_translations":
                r = await readthedocs_translations(str(arguments.get("project", "")))
            elif action == "rtd_subprojects":
                r = await readthedocs_subprojects(str(arguments.get("project", "")))
            elif action == "rtd_builds":
                r = await readthedocs_builds(str(arguments.get("project", "")))
            else:
                return _res({"error": f"unknown docs action: {action}"}, False)
            return _res(r, r.get("success", False))

        elif name == "openalex":
            query = str(arguments.get("query", ""))
            action = str(arguments.get("action", "works"))
            cnt = int(arguments.get("count", 10))
            if action == "authors":
                r = await search_openalex_authors(query, cnt)
            elif action == "topics":
                r = await search_openalex_concepts(query, cnt)
            elif action == "institutions":
                r = await search_openalex_institutions(query, cnt)
            else:
                r = await search_openalex(query, cnt)
            return _res(r, r.get("success", False))

        elif name == "search_all":
            query = str(arguments.get("query", ""))
            library = str(arguments.get("library", ""))
            version = str(arguments.get("version", ""))
            owner = str(arguments.get("owner", ""))
            repo = str(arguments.get("repo", ""))
            language = str(arguments.get("language", ""))
            cnt = min(max(safe_int(arguments.get("count", 10)), 1), 50)
            lib = library or (query.split()[0] if query.strip() else "")

            expanded = await expand_code_query(query)
            code_q = expanded[1] if len(expanded) > 1 else query

            tasks = []
            task_names = []

            tasks.append(context7_resolve(lib, version=version))
            task_names.append("context7")

            if GH_TOKEN:
                tasks.append(search_github(query, "code", cnt, owner, repo, language))
                task_names.append("github")

            if owner and repo:
                tasks.append(deepwiki_ask(owner=owner, repo=repo, question=query))
                task_names.append("deepwiki")

            tasks.append(codewiki_search_repos(query, cnt // 3 + 1))
            task_names.append("codewiki")

            _so_tags_t = " ".join(_SE_TAGS.findall(query))
            tasks.append(search_so(query, cnt // 3 + 1, tags=_so_tags_t, sort="votes"))
            task_names.append("so")

            if SOFA_KEY:
                sc = cnt // 5 + 1
                for ct in ("question", "til", "blueprint"):
                    tasks.append(search_sofa(query, sc, content_type=ct))
                    task_names.append(f"sofa_{ct}")

            tasks.append(search_hn(query, cnt // 3 + 1, min_points=50))
            task_names.append("hn")

            if LI_KEY:
                _li_p = _LI_PLATFORM_MAP.get(next(iter(_LI_PLATFORM.findall(query) or []), "").lower(), "")
                tasks.append(libraries_io_search(query, platform=_li_p, per_page=cnt))
                task_names.append("libraries_io")

            tasks.append(npm_search(query, cnt))
            task_names.append("npm")

            tasks.append(crates_search(query, cnt))
            task_names.append("crates")

            tasks.append(devdocs_search(lib, query) if lib else devdocs_list_docs())
            task_names.append("devdocs")

            tasks.append(search_papers(query, cnt, fields_of_study="Computer Science"))
            task_names.append("s2_papers")

            if CORE_API_AVAILABLE:
                core_q = f"{query} language.code:en"
                tasks.append(search_core_works(core_q, cnt))
                task_names.append("core_papers")

            tasks.append(tavily_search(code_q, cnt, include_domains=["github.com", "docs.*", "dev.to", "stackoverflow.com"]))
            task_names.append("tavily")

            tasks.append(search_openalex(query, cnt))
            task_names.append("openalex")

            async def _gather_with_deadline(tks, names, dl):
                sem = asyncio.Semaphore(15)
                async def _run(task):
                    async with sem:
                        return await task
                wrapped = [asyncio.ensure_future(_run(t)) for t in tks]
                done, pending = await asyncio.wait(wrapped, timeout=dl)
                for p in pending:
                    p.cancel()
                res = {}
                for n, w in zip(names, wrapped):
                    if w in done:
                        try:
                            res[n] = w.result()
                        except asyncio.CancelledError:
                            continue
                        except Exception as e:
                            res[n] = e
                    else:
                        res[n] = asyncio.TimeoutError(f"{n} exceeded deadline")
                return res

            results = await _gather_with_deadline(tasks, task_names, 30)
            merged: dict[str, Any] = {}
            flat_items: list[dict] = []
            for name_, result in results.items():
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
                    dw_answer = result.get("answer", "")
                    dw_url = f"https://deepwiki.com/{owner}/{repo}" if owner and repo else ""
                    merged["architecture"] = {"url": dw_url, "answer": dw_answer[:2000]}
                    flat_items.append({"source": "deepwiki", "title": f"architecture Q&A: {owner}/{repo}", "text": dw_answer[:3000], "url": dw_url})
                elif name_ == "codewiki":
                    merged["ai_wikis"] = result["results"]
                    for wr in (result.get("results", []) or []):
                        flat_items.append({"source": "codewiki", "title": wr.get("full_name", ""), "text": wr.get("description", ""), "url": wr.get("url", "")})
                elif name_ == "so":
                    merged["stackoverflow"] = result["results"]
                    for sr in (result.get("results", []) or []):
                        flat_items.append({"source": "stackoverflow", "title": sr.get("title", ""), "text": f"{sr.get('body','')} {sr.get('top_answer','')}", "url": sr.get("url", "")})
                elif name_ in ("sofa_question", "sofa_til", "sofa_blueprint"):
                    content_type_map = {"sofa_question": "question", "sofa_til": "til", "sofa_blueprint": "blueprint"}
                    ct = content_type_map.get(name_, "question")
                    sofa_key = f"sofa_{ct}"
                    if sofa_key not in merged:
                        merged[sofa_key] = []
                    merged[sofa_key].extend(result.get("results", []))
                    if result.get("steering"):
                        merged["sofa_steering"] = result["steering"]
                    for sr in (result.get("results", []) or []):
                        flat_items.append({"source": f"sofa_{ct}", "title": sr.get("title", ""), "text": sr.get("body", ""), "url": sr.get("url", "")})
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
                elif name_ == "core_papers":
                    merged["core_papers"] = result.get("results", [])
                    for cr in (result.get("results", []) or []):
                        flat_items.append({"source": "core", "title": cr.get("title", ""), "text": cr.get("abstract", ""), "url": cr.get("downloadUrl", "") or ""})
                elif name_ == "openalex":
                    merged["openalex"] = result.get("results", [])
                    for pr in (result.get("results", []) or []):
                        flat_items.append({"source": "openalex", "title": pr.get("title", ""), "text": pr.get("snippet", ""), "url": pr.get("url", "")})
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
                    merged["deduped_results"] = await _rerank(query, merged["deduped_results"], top_k=min(cnt * 2, 50))
                except Exception as e:
                    logger.warning("reranker failed: %s", e)

            if merged.get("deduped_results"):
                try:
                    top = merged["deduped_results"][:3]
                    ctx = "\n\n".join(f"[{i+1}] (source: {x.get('source','?')}) {x.get('title','')}: {(x.get('text','') or x.get('snippet','') or '')[:400]}"
                                     for i, x in enumerate(top))
                    _groq_keys = _KeyRotator("GROQ_API_KEYS")
                    if _groq_keys.has_keys:
                        c = get_http_client()
                        resp = await c.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {_groq_keys.next()}", "Content-Type": "application/json"},
                            json={"model": "openai/gpt-oss-120b",
                                  "messages": [{"role": "system", "content": "Answer concisely about code/libraries from sources. Use [N] citations like [1][2]."},
                                               {"role": "user", "content": f"Query: {query}\n\nSources:\n{ctx}"}],
                                  "temperature": 0.3, "max_tokens": 256}, timeout=15)
                        if resp.status_code == 200:
                            merged["synthesis"] = resp.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    logger.warning("synthesis failed: %s", e)
                    merged["synthesis_error"] = str(e)

            return _res(merged, bool(merged))

        elif name == "papers":
            action = str(arguments.get("action", ""))
            if action == "details":
                r = await get_paper_details(paper_id=str(arguments.get("paper_id", "")))
            elif action == "batch":
                raw = str(arguments.get("paper_ids", ""))
                ids = [x.strip() for x in raw.split(",") if x.strip()]
                r = await get_papers_batch(ids)
            elif action == "core_search":
                r = await search_core_works(query=str(arguments.get("query", "")),
                    limit=int(arguments.get("count", 10)),
                    offset=int(arguments.get("offset", 0)))
            elif action == "citations":
                r = await get_paper_citations(
                    paper_id=str(arguments.get("paper_id", "")),
                    limit=int(arguments.get("count", 20)))
            elif action == "references":
                r = await get_paper_references(
                    paper_id=str(arguments.get("paper_id", "")),
                    limit=int(arguments.get("count", 20)))
            elif action == "recommendations":
                r = await get_paper_recommendations(
                    paper_id=str(arguments.get("paper_id", "")),
                    limit=int(arguments.get("count", 10)))
            elif action == "author_search":
                r = await search_authors(
                    query=str(arguments.get("query", "")),
                    limit=int(arguments.get("count", 10)))
            elif action == "author_papers":
                r = await get_author_papers(
                    author_id=str(arguments.get("author_id", "")),
                    limit=int(arguments.get("count", 10)))
            elif action == "autocomplete":
                r = await autocomplete_papers(
                    query=str(arguments.get("query", "")))
            elif action == "author_by_id":
                r = await s2_author_by_id(author_id=str(arguments.get("author_id","")))
            elif action == "bulk_search":
                raw = str(arguments.get("paper_ids",""))
                ids = [x.strip() for x in raw.split(",") if x.strip()]
                r = await s2_bulk_search(ids)
            elif action == "recommendations_negatives":
                pos_raw = str(arguments.get("positive_ids",""))
                pos_ids = [x.strip() for x in pos_raw.split(",") if x.strip()]
                neg_raw = str(arguments.get("negative_ids",""))
                neg_ids = [x.strip() for x in neg_raw.split(",") if x.strip()]
                r = await s2_recommendations_with_negatives(pos_ids, neg_ids or None,
                    limit=int(arguments.get("count",10)))
            elif action == "arxiv_search":
                c = get_http_client()
                from urllib.parse import urlencode
                query = str(arguments.get("query", ""))
                count = min(int(arguments.get("count", 10)), 50)
                params = {"search_query": f"all:{query}", "max_results": count,
                          "sortBy": "relevance", "sortOrder": "descending"}
                import xml.etree.ElementTree as ET
                try:
                    resp = await c.get("https://export.arxiv.org/api/query",
                                       params=params, timeout=15)
                    if resp.status_code == 200:
                        papers = []
                        root = ET.fromstring(resp.content)
                        ns = {"a": "http://www.w3.org/2005/Atom",
                              "arxiv": "http://arxiv.org/schemas/atom"}
                        for entry in root.findall("a:entry", ns):
                            pid = entry.find("a:id", ns)
                            title = entry.find("a:title", ns)
                            summary = entry.find("a:summary", ns)
                            published = entry.find("a:published", ns)
                            cats = [c.get("term", "") for c in entry.findall("arxiv:primary_category", ns)]
                            authors = [a.find("a:name", ns).text if a.find("a:name", ns) is not None else ""
                                       for a in entry.findall("a:author", ns)]
                            pdf_link = ""
                            for link in entry.findall("a:link", ns):
                                if link.get("title") == "pdf":
                                    pdf_link = link.get("href", "")
                                    break
                            papers.append({
                                "paper_id": pid.text.strip().split("/")[-1] if pid is not None and pid.text else "",
                                "title": title.text.strip() if title is not None and title.text else "",
                                "summary": summary.text.strip()[:500] if summary is not None and summary.text else "",
                                "published": published.text.strip()[:10] if published is not None and published.text else "",
                                "authors": authors,
                                "categories": cats,
                                "pdf_url": pdf_link,
                            })
                        r = {"success": True, "total": len(papers), "papers": papers}
                    else:
                        r = {"success": False, "error": f"arXiv returned {resp.status_code}"}
                except Exception as e:
                    r = {"success": False, "error": str(e)}
            else:
                r = await search_papers(
                    query=str(arguments.get("query", "")),
                    limit=int(arguments.get("count", 10)),
                    year=str(arguments.get("year", "")),
                    fields_of_study=str(arguments.get("fields_of_study", "")),
                    open_access=bool(arguments.get("open_access", False)),
                )
            return _res(r, r.get("success", False))

        elif name == "searchcode":
            action = str(arguments.get("action", ""))
            repo = str(arguments.get("repository", ""))
            if not repo:
                return _res({"error": "repository is required for searchcode"}, False)
            if action == "analyze":
                r = await searchcode_analyze(
                    repo, language=str(arguments.get("language", "")),
                    path=str(arguments.get("path", "")),
                    detail_level=str(arguments.get("detail_level", "summary")))
            elif action == "search":
                r = await searchcode_search(
                    repo, query=str(arguments.get("query", "")),
                    max_results=int(arguments.get("max_results", 10)),
                    context_lines=int(arguments.get("context_lines", 2)),
                    case_sensitive=bool(arguments.get("case_sensitive", False)))
            elif action == "findings":
                r = await searchcode_findings(
                    repo, path=str(arguments.get("path", "")),
                    severity=str(arguments.get("severity", "")),
                    category=str(arguments.get("category", "")),
                    max_results=int(arguments.get("max_results", 50)))
            elif action == "file_tree":
                r = await searchcode_file_tree(
                    repo, path_filter=str(arguments.get("path", "")),
                    query=str(arguments.get("query", "")))
            elif action == "get_file":
                r = await searchcode_get_file(
                    repo, path=str(arguments.get("path", "")),
                    symbol_name=str(arguments.get("symbol_name", "")),
                    start_line=int(arguments.get("start_line", 1)),
                    end_line=int(arguments.get("end_line", 0)))
            else:
                return _res({"error": f"unknown searchcode action: {action}"}, False)
            return _res(r, r.get("success", False))

        elif name == "code_search":
            r = await pkgseer_search(
                query=str(arguments.get("query", "")),
                target=str(arguments.get("target", "")),
                source=str(arguments.get("source", "")),
                lang=str(arguments.get("lang", "")),
                limit=int(arguments.get("limit", 10)),
            )
            return _res(r, r.get("success", False))

        elif name == "code_files":
            spec = str(arguments.get("spec", ""))
            # Try jsDelivr first for npm (faster, no auth)
            if spec.startswith("npm:") or ":" not in spec:
                jr = await jsdelivr_list_files(spec)
                if jr.get("success"):
                    return _res(jr)
            r = await pkgseer_code_files(
                spec=spec,
                path_prefix=str(arguments.get("path_prefix", "")),
            )
            return _res(r, r.get("success", False))

        elif name == "code_read":
            spec = str(arguments.get("spec", ""))
            path = str(arguments.get("path", ""))
            # Try jsDelivr CDN first for npm (faster, CDN-served)
            if spec.startswith("npm:") or ":" not in spec or spec == "npm":
                jr = await jsdelivr_read_file(spec, path)
                if jr.get("success"):
                    return _res(jr)
            r = await pkgseer_code_read(
                spec=spec,
                path=path,
            )
            return _res(r, r.get("success", False))

        elif name == "code_grep":
            r = await pkgseer_code_grep(
                spec=str(arguments.get("spec", "")),
                pattern=str(arguments.get("pattern", "")),
                path_prefix=str(arguments.get("path_prefix", "")),
            )
            return _res(r, r.get("success", False))

        elif name == "pkg_deps":
            r = await pkgseer_pkg_deps(spec=str(arguments.get("spec", "")))
            return _res(r, r.get("success", False))

        elif name == "enrich":
            query = str(arguments.get("query", ""))
            raw_results = arguments.get("results", [])
            top_k = int(arguments.get("top_k", 20))
            max_fetch = int(arguments.get("max_fetch_size", 50))
            include_html = bool(arguments.get("include_html", False))
            if not raw_results or not isinstance(raw_results, list):
                return _res({"error": "results must be a non-empty list"})
            r = await enrich_results(query, raw_results, top_k=top_k,
                                     max_fetch_size=max_fetch, include_html=include_html)
            return _res(r, r.get("success", False))

        elif name == "analyze":
            repo_str = str(arguments.get("repository", ""))
            r = await analyze_repo(repo_str)
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
    except BaseException as e:
        return _res({"error": f"handler interrupted: {type(e).__name__}"}, False)


async def _warmup_reranker():
    try:
        from reranker import warmup
        await warmup()
    except Exception:
        pass


async def main():
    global _warmup_task
    _warmup_task = asyncio.create_task(_warmup_reranker())
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await close_http_client()


if __name__ == "__main__":
    asyncio.run(main())
