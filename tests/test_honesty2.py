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
