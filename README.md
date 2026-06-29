# codesearch MCP Server

Unified code search across 15+ sources with embedding dedup + cross-encoder reranking for OpenCode.

## Tools

- **search_all** — Unified search across 15+ sources (Context7, GitHub, DeepWiki, CodeWiki, SO, SOFA, HN, Libraries.io, npm, crates, DevDocs, Semantic Scholar, CORE API, arXiv, Tavily) with NIM dedup + BM25 + cross-encoder reranker
- **github** — GitHub search (code/repos/issues/users) + repo ops (readme, contents, languages, topics, releases)
- **docs** — Documentation smart search (Context7→ReadTheDocs→llms.txt→DevDocs) + direct DevDocs/ReadTheDocs access
- **search_package** — Package metadata from npm/PyPI/crates + deps.dev dependency graphs
- **so_search** — Stack Overflow via Stack Exchange API (free) or SOFA
- **hn** — Hacker News search, item detail, story lists
- **search_libraries** — Libraries.io dependency metadata
- **vulns** — Sonatype Guide vulnerability scanning
- **papers** — Semantic Scholar, CORE API, and arXiv academic paper search
- **wiki** — Repo architecture via DeepWiki + CodeWiki

## Setup

1. Copy `run.example` to `run` and fill in API keys
2. Install dependencies in a Python 3.14+ venv
3. Run via OpenCode MCP config pointing to `run`
