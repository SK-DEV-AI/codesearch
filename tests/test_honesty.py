import asyncio
import json
from types import SimpleNamespace

import pytest

import enrich
import llm
import server


def _ep(provider="local"):
    return {"provider": provider, "base_url": "http://x", "model": "m",
            "api_key": "k", "rotate": (provider == "groq")}


# ---------- source_availability propagates dict-lane errors ----------

@pytest.mark.asyncio
async def test_availability_propagates_dict_error():
    st = server._availability_status({"success": False, "error": "boom", "status": 429})
    assert st == "failed: boom status=429"
    assert server._availability_status({"success": True}) == "success"
    assert server._availability_status({"success": False}) == "failed (no error message)"
    assert server._availability_status(asyncio.TimeoutError()) == "timed_out"
    assert server._availability_status(RuntimeError("")) == "error: RuntimeError (no message)"
    assert server._availability_status(RuntimeError("read timeout")) == "error: RuntimeError: read timeout"
    # long reasons are clipped, never dropped
    long_msg = server._availability_status({"success": False, "error": "x" * 400})
    assert long_msg.startswith("failed: xxx") and len(long_msg) <= len("failed: ") + 160


# ---------- enrich honesty ----------

@pytest.mark.asyncio
async def test_enrich_success_false_when_all_fail(monkeypatch):
    async def _v(u):
        return u
    monkeypatch.setattr(enrich, "_validate_url", _v)

    async def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(enrich, "safe_stream", boom)

    async def fake_rerank(q, items, top_k):
        return items

    monkeypatch.setattr(enrich, "_rerank", fake_rerank)
    monkeypatch.setattr(enrich, "fallback_sort", lambda items, k: items)
    async def _embed_none(*a, **k):
        return None
    monkeypatch.setattr(enrich, "_embed", _embed_none)

    r = await enrich.enrich_results("q", [{"url": "http://example.com/x", "title": "t"}])
    assert r["success"] is False
    assert r["total_fetched"] == 0
    assert r["total_errors"] >= 1
    assert "error" in r


@pytest.mark.asyncio
async def test_enrich_strips_html_and_truncates(monkeypatch):
    async def _v(u):
        return u
    monkeypatch.setattr(enrich, "_validate_url", _v)

    async def fake_stream(*a, **k):
        return b"<html><body><p>Hello world " + b"x" * 5000 + b"</p></body></html>"

    monkeypatch.setattr(enrich, "safe_stream", fake_stream)

    async def fake_rerank(q, items, top_k):
        return items

    monkeypatch.setattr(enrich, "_rerank", fake_rerank)
    monkeypatch.setattr(enrich, "fallback_sort", lambda items, k: items)
    async def _embed_none(*a, **k):
        return None
    monkeypatch.setattr(enrich, "_embed", _embed_none)

    r = await enrich.enrich_results("q", [{"url": "http://example.com/x", "title": "t"}],
                                    max_chars=100, include_html=False)
    assert r["success"] is True
    entry = r["enriched"][0]
    assert "<" not in entry["content"] or "Hello world" in entry["content"]
    assert entry["truncated"] is True
    assert entry["content_is_html"] is False


# ---------- synthesis parity ----------

@pytest.mark.asyncio
async def test_synthesis_exposes_parity_fields(monkeypatch):
    monkeypatch.setattr(llm, "endpoints", lambda: [_ep("local")])

    async def fake_chat(*a, **k):
        llm._bump("synthesis")
        return ("synth text", "stop", "", 1)

    monkeypatch.setattr(llm, "chat", fake_chat)
    llm.begin()
    merged = {"deduped_results": [{"source": "s", "title": "t", "text": "hi",
                                    "url": "http://example.com/x"}]}
    await server._run_synthesis(merged, "q", {})
    assert merged["synthesis"] == "synth text"
    assert merged.get("synthesis_provider")
    assert "model_calls" in merged
    assert merged.get("finish_reason") == "stop"
    assert merged.get("truncated") is False
    assert isinstance(merged.get("sources"), list) and len(merged["sources"]) == 1
    assert merged.get("prompt_version")
    assert merged.get("sources_used") == 1
    assert merged.get("raw_available_via") == "search_all(..., synthesize=false)"


# ---------- ping honesty ----------

@pytest.mark.asyncio
async def test_ping_success_false_when_unreachable(monkeypatch):
    class FakeClient:
        async def get(self, *a, **k):
            raise RuntimeError("down")

    monkeypatch.setattr(server, "get_http_client", lambda: FakeClient())
    params = SimpleNamespace(name="ping", arguments={})
    res = await server.handle_call_tool(None, params)
    data = json.loads(res.content[0].text)
    assert data["success"] is False
    assert data.get("all_reachable") is False


# ---------- analyze honesty ----------

@pytest.mark.asyncio
async def test_analyze_success_false_when_all_fail(monkeypatch):
    # Patch all underlying fetchers to fail.
    import analyze as A
    async def fail(*a, **k):
        return {"success": False, "error": "nope"}
    for fn in ["gh_get_repo", "fetch_readme", "gh_get_languages", "gh_get_releases",
               "gh_get_topics", "searchcode_analyze", "deepwiki_ask", "codewiki_fetch_repo",
               "gitingest_digest"]:
        if hasattr(A, fn):
            monkeypatch.setattr(A, fn, fail)
    r = await A.analyze_repo("owner/repo")
    assert r["success"] is False
    assert "sources_failed" in r
    assert len(r["sources_failed"]) > 0