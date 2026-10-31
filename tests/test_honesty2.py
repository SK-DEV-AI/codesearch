import asyncio
import json
from types import SimpleNamespace

import pytest

import nvd
import server


def _params(name, arguments):
    return SimpleNamespace(name=name, arguments=arguments)


# ---------- CA13/CS7: _availability_status empty success ----------

def test_availability_empty_success():
    assert server._availability_status({"success": True, "results": []}) == "success (0 results)"
    assert server._availability_status({"success": True, "total": 0}) == "success (0 results)"
    assert server._availability_status({"success": True}) == "success"


# ---------- CA7/CS1: code_read surfaces truncation ----------

@pytest.mark.asyncio
async def test_code_read_surfaces_truncation(monkeypatch):
    async def fake_js(spec, path):
        return {"success": True, "content": "hi", "total_chars": 2}
    monkeypatch.setattr(server, "jsdelivr_read_file", fake_js)
    res = await server.handle_call_tool(None, _params("code_read", {"spec": "npm:x", "path": "a.js"}))
    data = json.loads(res.content[0].text)
    assert data["truncated"] is False
    assert data["original_length"] == 2

    async def fake_ps(spec, path):
        return {"success": True, "content": "partial", "totalLines": 10, "endLine": 3}
    monkeypatch.setattr(server, "jsdelivr_read_file", lambda *a, **k: {"success": False, "error": "skip"})
    monkeypatch.setattr(server, "pkgseer_code_read", fake_ps)
    res = await server.handle_call_tool(None, _params("code_read", {"spec": "pypi:x", "path": "a.py"}))
    data = json.loads(res.content[0].text)
    assert data["truncated"] is True
    assert data["next_offset"] == 4
    assert data["original_length"] == 10


# ---------- CA8/CS2: code_search per-lane status + truncated ----------

@pytest.mark.asyncio
async def test_code_search_lane_status_and_truncated(monkeypatch):
    async def gm(query, *a, **k):
        return {"success": True, "hits": [
            {"repo": "r", "file": "f", "matches": [{"preview": "p"}]},
            {"repo": "r2", "file": "f2", "matches": [{"preview": "p2"}]},
        ]}
    async def sg(query, *a, **k):
        return {"success": True, "results": []}
    monkeypatch.setattr(server, "grep_mcp_search", gm)
    monkeypatch.setattr(server, "search_sourcegraph", sg)
    res = await server.handle_call_tool(None, _params("code_search", {"query": "q", "limit": 1}))
    data = json.loads(res.content[0].text)
    assert data["truncated"] is True
    # PkgSeer needs a registry+name target, so the global path must report it
    # as skipped, not as a failed lane (that was a phantom failure).
    assert data["lane_status"]["pkgseer"].startswith("skipped")
    assert data["lane_status"]["grep-mcp"].startswith("success")
    assert "lane_status" in data


# ---------- CA9/CS3: pkg enrichment_errors ----------

@pytest.mark.asyncio
async def test_pkg_enrichment_errors(monkeypatch):
    async def resolve_pkg(registry, name):
        return {"success": True, "registry": "npm", "repo_owner": "o", "repo_name": "r", "version": "1.0"}
    async def boom(*a, **k):
        raise RuntimeError("boom")
    async def ok(*a, **k):
        return {"success": False}
    monkeypatch.setattr(server, "resolve_package", resolve_pkg)
    monkeypatch.setattr(server, "search_libraries_io", boom)
    monkeypatch.setattr(server, "scan_vulnerabilities", ok)
    monkeypatch.setattr(server, "_ghsa_search", ok)
    monkeypatch.setattr(server, "scorecard_score", ok)
    res = await server.handle_call_tool(None, _params("pkg", {"name": "express"}))
    data = json.loads(res.content[0].text)
    assert any("libraries_io" in e and "boom" in e for e in data.get("enrichment_errors", []))


# ---------- CA10/CS4: wiki ask fallback ----------

@pytest.mark.asyncio
async def test_wiki_ask_fallback(monkeypatch):
    async def dw(*a, **k):
        return {"success": False, "error": "boom"}
    async def cw(owner, repo, question):
        return {"success": True, "answer": "hi"}
    monkeypatch.setattr(server, "deepwiki_ask", dw)
    monkeypatch.setattr(server, "codewiki_ask_repo", cw)
    res = await server.handle_call_tool(None, _params("wiki", {"action": "ask", "owner": "o", "repo": "r", "question": "q"}))
    data = json.loads(res.content[0].text)
    assert data.get("fallback") is True
    assert data.get("answered_by") == "codewiki"


# ---------- CA11/CS5: vulns dedupe description ----------

@pytest.mark.asyncio
async def test_vulns_dedupe_description(monkeypatch):
    async def fake_detail(vuln_id=""):
        return {"success": True, "vulnerability": {"id": "CVE-2021-12345", "description": "same text"}}
    monkeypatch.setattr(server, "get_vulnerability_detail", fake_detail)

    async def fake_enrich(cid):
        return {"success": True, "cve": {"description": "same text"}}
    monkeypatch.setattr(nvd, "nvd_enrich", fake_enrich)

    res = await server.handle_call_tool(None, _params("vulns", {"action": "detail", "vuln_id": "CVE-2021-12345"}))
    data = json.loads(res.content[0].text)
    assert data["vulnerability"]["description"] == "same text"
    assert "description" not in (data.get("nvd") or {})
    assert (data.get("nvd") or {}).get("description_sources") == ["vulnerability", "nvd"] or data.get("description_sources") == ["vulnerability", "nvd"]


# ---------- CS8: long natural-language queries relax instead of returning 0 ----------

import codequery
import grep_mcp
import sourcegraph


def test_relax_queries_shape():
    assert codequery.relax_queries("httpx aiohttp") == []          # already short
    v = codequery.relax_queries("httpx vs aiohttp benchmark")
    assert v == ["httpx aiohttp benchmark", "httpx aiohttp"]      # stopword dropped, then 2 terms
    # No stopwords: still narrows to 3 then 2 terms.
    assert codequery.relax_queries("sqlite fts5 bm25 ranking") == [
        "sqlite fts5 bm25", "sqlite fts5"]
    # Nothing to relax when every token is filler-free and short.
    assert codequery.relax_queries("") == []


@pytest.mark.asyncio
async def test_grep_mcp_relaxes_on_empty(monkeypatch):
    seen = []

    async def fake(query, **k):
        seen.append(query)
        hits = [] if len(query.split()) > 2 else [{"repo": "r", "file": "f", "matches": []}]
        return {"success": True, "hits": hits}

    monkeypatch.setattr(grep_mcp, "_grep_mcp_call", fake)
    r = await grep_mcp.grep_mcp_search("sqlite fts5 bm25 ranking")
    assert r["hits"] and r["query_relaxed"] is True
    assert r["query_requested"] == "sqlite fts5 bm25 ranking"
    assert seen == ["sqlite fts5 bm25 ranking", "sqlite fts5 bm25", "sqlite fts5"]


@pytest.mark.asyncio
async def test_grep_mcp_does_not_relax_after_error(monkeypatch):
    calls = []

    async def fake(query, **k):
        calls.append(query)
        return {"success": False, "error": "grep MCP timeout"}

    monkeypatch.setattr(grep_mcp, "_grep_mcp_call", fake)
    r = await grep_mcp.grep_mcp_search("sqlite fts5 bm25 ranking")
    assert r["success"] is False and len(calls) == 1


@pytest.mark.asyncio
async def test_sourcegraph_relaxes_on_empty(monkeypatch):
    async def fake(q):
        # The built query carries `context:global` + count; the terms are last.
        if "ranking" in q:
            return {"success": True, "results": [], "total": 0, "query": q}
        return {"success": True, "results": [{"source": "sourcegraph"}],
                "total": 1, "query": q}

    monkeypatch.setattr(sourcegraph, "_sg_attempt", fake)
    r = await sourcegraph.search_sourcegraph("sqlite fts5 bm25 ranking")
    assert r["results"] and r["query_relaxed"] is True
    assert r["query_requested"] == "sqlite fts5 bm25 ranking"


@pytest.mark.asyncio
async def test_code_search_reports_relaxed_query(monkeypatch):
    async def gm(query, *a, **k):
        return {"success": True, "hits": [{"repo": "r", "file": "f", "matches": []}],
                "query_relaxed": True, "query_requested": "a b c d", "query": "a b"}
    async def sg(query, *a, **k):
        return {"success": True, "results": []}
    monkeypatch.setattr(server, "grep_mcp_search", gm)
    monkeypatch.setattr(server, "search_sourcegraph", sg)
    res = await server.handle_call_tool(None, _params("code_search", {"query": "a b c d"}))
    data = json.loads(res.content[0].text)
    assert data["query_relaxed"] is True
    assert data["query_requested"] == "a b c d"
    assert data["queries_used"] == {"grep-mcp": "a b"}
