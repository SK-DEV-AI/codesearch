# codesearch MCP Server

Unified code search across 14+ sources with embedding dedup + cross-encoder reranking for OpenCode.

## Tools

- **search_all** — Unified search across 14+ sources (GitHub, SO, HN, npm/crates/PyPI, Libraries.io, DevDocs, etc.)
- **github_search** — GitHub code/repos/issues search with qualifiers
- **search_docs** — Library docs via Context7 → ReadTheDocs → llms.txt → DevDocs
- **search_package** — Package metadata from npm/PyPI/crates
- **so_search** — Stack Overflow via Stack Exchange API (free) or SOFA
- **hn** — Hacker News search, item detail, story lists
- **search_libraries** — Libraries.io dependency metadata
- **vulns** — Sonatype Guide vulnerability scanning
- **github** — GitHub repo operations (readme, contents, languages, releases)
- **docs** — DevDocs + ReadTheDocs docs
- **papers** — Semantic Scholar search or detail
- **wiki** — Repo architecture via DeepWiki + CodeWiki

## Setup

1. Copy `run.example` to `run` and fill in API keys
2. Install dependencies in a Python 3.14+ venv
3. Run via OpenCode MCP config pointing to `run`
