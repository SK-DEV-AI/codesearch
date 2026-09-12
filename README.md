# codesearch MCP Server

Code, package, docs, and paper search MCP server. Per-package source navigation (npm/PyPI/crates), per-repo intelligence (GitHub URL or owner/repo), Q&A communities, vulnerability data, documentation, and academic papers — with embedding dedup and a local cross-encoder reranker.

## Tools (17)

- **analyze** — Start here for unfamiliar repos: GitHub metadata + code quality + wiki architecture in one call (10–15s).
- **search_all** — Broad multi-source discovery (Context7, GitHub, DeepWiki, CodeWiki, SO, HN, Libraries.io, npm, DevDocs, Semantic Scholar, Tavily…). For surgical single-source work call the tool below directly.
- **code_search** — Search code/docs/symbols across indexed dependencies (`target=npm:express`). npm/PyPI/crates only — for GitHub repos use `searchcode` or `analyze`.
- **code_files** / **code_read** — List files / read a file in an indexed dependency by package spec.
- **searchcode** — Per-repo intelligence: `analyze` (overview), `search` (code within the repo), `findings` (quality issues), `file_tree`, `get_file`. Needs `repository` for every action. Free, no auth.
- **wiki** — Repo architecture Q&A via DeepWiki + CodeWiki. Single repo (`owner`+`repo`) or up to 10 repos (`repos`).
- **search_package** — Raw registry queries: versions, dist-tags, deps.dev graphs, advisories (`version='latest'` OK). For composite intelligence use `pkg`.
- **pkg** — Composite package intelligence: metadata, changelog, upgrade review (vulns + changelog + deps diff), file list/read.
- **search_libraries** — Libraries.io metadata and source rank. Compare dependency candidates, find the repo behind a package.
- **vulns** — Sonatype Guide: `scan` (needs platform+name+version), `detail` (needs vuln_id), `latest_version`/`license` (need purl).
- **docs** — Docs with smart fallback (Context7 → ReadTheDocs → llms.txt → DevDocs); then `fetch(url)` or `devdocs_fetch_content` for full pages.
- **so_search** — Stack Overflow: `stackexchange` (authoritative resolved answers) or `sofa` (fresher agent content, needs `SOFA_KEY`).
- **hn** — Hacker News discussion and sentiment (search, stories, items, users).
- **papers** — Academic papers. Prefer `action=search` (S2 → arXiv → OpenAlex → CORE fallback chain); `arxiv_search` is single-source and rate-limited.
- **enrich** — Fetch full content for a result list from any search tool, dedup + rerank.
- **ping** — Connectivity check. Use before expensive calls when uncertain.

## Setup

Requires Python 3.14+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp run.example run   # fill in your API keys (run is gitignored — never commit it)
chmod +x run
```

Point your MCP client at `run` (stdio):

```json
{ "codesearch": { "command": "/path/to/codesearch/run" } }
```

### API keys (`run`)

| Key | Used for | Required? |
|---|---|---|
| `GITHUB_TOKEN` | GitHub API (repo ops, higher rate limits) | Recommended |
| `CONTEXT7_API_KEY` | Context7 docs | Optional |
| `TAVILY_KEYS` | Tavily lane in search_all | Optional |
| `S2_API_KEY` | Semantic Scholar (1 rps dedicated vs shared anonymous pool) | Optional |
| `SE_API_KEY` / `SOFA_KEY` | Stack Exchange / Stack Overflow for Agents | Optional |
| `LI_KEY` | Libraries.io | Optional |
| `OSS_TOKEN` | Sonatype OSS Index | Optional |
| `FIRECRAWL_KEYS` | Firecrawl fallback | Optional |
| `NV_KEY` | NVIDIA embedding endpoint | Optional |

## Notes

- **Reranker (optional, local):** shared with the websearch server — one cross-encoder worker over a Unix socket. Without it, results return unranked; everything still works.
- **Caching:** per-source file caches with TTLs; repeat queries are cheap.
- **Safety:** all user-URL fetches are SSRF-validated (private-IP blocking + redirect-hop revalidation).
