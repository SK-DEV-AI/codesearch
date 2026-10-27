import asyncio
import os

import pytest

import llm
import code_expand
import server


# ---------- helpers ----------

def _ep(provider="local"):
    return {"provider": provider, "base_url": "http://x", "model": "m",
            "api_key": "k", "rotate": (provider == "groq")}


# ---------- 1. endpoints local first by default ----------

def test_endpoints_local_first_by_default(monkeypatch):
    monkeypatch.setattr(llm, "local_endpoint", lambda: _ep("local"))
    monkeypatch.setenv("GROQ_API_KEYS", "g1,g2")
    monkeypatch.delenv("CODESEARCH_SYNTH_PROVIDER", raising=False)
    monkeypatch.delenv("CODESEARCH_MODEL_FALLBACK", raising=False)
    eps = llm.endpoints()
    assert [e["provider"] for e in eps] == ["local", "groq"]


# ---------- 2. strict mode drops fallback ----------

def test_endpoints_strict_mode_drops_fallback(monkeypatch):
    monkeypatch.setattr(llm, "local_endpoint", lambda: _ep("local"))
    monkeypatch.setenv("GROQ_API_KEYS", "g1")
    monkeypatch.setenv("CODESEARCH_MODEL_FALLBACK", "off")
    monkeypatch.delenv("CODESEARCH_SYNTH_PROVIDER", raising=False)
    eps = llm.endpoints()
    assert len(eps) == 1
    assert eps[0]["provider"] == "local"


# ---------- 3. chat no endpoint returns error ----------

@pytest.mark.asyncio
async def test_chat_no_endpoint_returns_error(monkeypatch):
    monkeypatch.setattr(llm, "local_endpoint", lambda: {})
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    llm.begin()
    text, finish, err, att = await llm.chat([{"role": "user", "content": "x"}])
    assert text == ""
    assert "no model endpoint" in err
    assert att == 0
    assert llm.counts() == {}


# ---------- 4. chat falls back on 429 ----------

class _FakeResp:
    def __init__(self, status, body=None):
        self.status_code = status
        self.text = "rate limited" if status == 429 else ""
        self._body = body or {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}

    def json(self):
        return self._body


class _SeqClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    async def post(self, *a, **kw):
        st = self.statuses[self.calls]
        self.calls += 1
        if st == 429:
            return _FakeResp(429)
        return _FakeResp(200)


@pytest.mark.asyncio
async def test_chat_falls_back_on_429(monkeypatch):
    monkeypatch.setattr(llm, "local_endpoint", lambda: _ep("local"))
    monkeypatch.setenv("GROQ_API_KEYS", "g1")
    monkeypatch.delenv("CODESEARCH_SYNTH_PROVIDER", raising=False)
    monkeypatch.delenv("CODESEARCH_MODEL_FALLBACK", raising=False)

    # force two endpoints: local first, groq second
    def eps():
        return [_ep("local"), _ep("groq")]
    monkeypatch.setattr(llm, "endpoints", eps)

    client = _SeqClient([429, 200])
    llm.begin()
    text, finish, err, att = await llm.chat(
        [{"role": "user", "content": "x"}], client=client, attempts=1,
    )
    assert text == "answer"
    assert err == ""
    assert att >= 1


# ---------- 5. empty content escalates budget ----------

class _EmptyThenFullClient:
    def __init__(self):
        self.budgets = []

    async def post(self, *a, **kw):
        self.budgets.append(kw.get("json", {}).get("max_tokens"))
        if len(self.budgets) == 1:
            return _FakeResp(200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
        return _FakeResp(200, {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})


@pytest.mark.asyncio
async def test_chat_empty_content_escalates_budget(monkeypatch):
    monkeypatch.setattr(llm, "local_endpoint", lambda: _ep("local"))
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.delenv("CODESEARCH_SYNTH_PROVIDER", raising=False)
    monkeypatch.delenv("CODESEARCH_MODEL_FALLBACK", raising=False)

    def eps():
        return [_ep("local")]
    monkeypatch.setattr(llm, "endpoints", eps)

    client = _EmptyThenFullClient()
    llm.begin()
    text, finish, err, att = await llm.chat(
        [{"role": "user", "content": "x"}], client=client, attempts=2, max_tokens=64,
    )
    assert text == "answer"
    assert len(client.budgets) == 2
    assert client.budgets[1] > client.budgets[0]
    assert client.budgets[0] >= llm.MIN_USEFUL_TOKENS


# ---------- 6. counts are per context ----------

@pytest.mark.asyncio
async def test_counts_are_per_context():
    async def two_requests():
        async def one(kind):
            llm.begin()
            await asyncio.sleep(0)
            llm._bump(kind)
            return llm.counts()

        return await asyncio.gather(one("synthesis"), one("verify"))

    a, b = await two_requests()
    assert a == {"synthesis": 1}
    assert b == {"verify": 1}
    assert llm.counts() == {}


# ---------- 7. expand uses chat and parses variants ----------

@pytest.mark.asyncio
async def test_expand_uses_chat_and_parses_variants(monkeypatch):
    monkeypatch.setattr(code_expand, "endpoints", lambda: [_ep("local")])

    async def fake_chat(*a, **k):
        return ("foo bar baz\nqux quux corge", "stop", "", 1)

    monkeypatch.setattr(code_expand, "chat", fake_chat)
    res = await code_expand.expand_code_query("some query about code")
    assert res[0] == "some query about code"
    assert len(res) >= 2

    # empty -> [query]
    async def empty_chat(*a, **k):
        return ("", "", "no content", 1)

    monkeypatch.setattr(code_expand, "chat", empty_chat)
    res2 = await code_expand.expand_code_query("another query here")
    assert res2 == ["another query here"]


# ---------- 8. synthesis records provider ----------

@pytest.mark.asyncio
async def test_synthesis_records_provider(monkeypatch):
    monkeypatch.setattr(llm, "endpoints", lambda: [_ep("local")])

    async def fake_chat(*a, **k):
        llm._bump("synthesis")
        return ("synth text", "stop", "", 1)

    monkeypatch.setattr(llm, "chat", fake_chat)
    llm.begin()
    merged = {"deduped_results": [{"source": "s", "title": "t", "text": "hi"}]}
    await server._run_synthesis(merged, "q", {})
    assert merged["synthesis"] == "synth text"
    assert merged.get("synthesis_provider")
    assert "model_calls" in merged


# ---------- 9. expand does not raise on transport error ----------

@pytest.mark.asyncio
async def test_expand_does_not_raise_on_transport_error(monkeypatch, caplog):
    monkeypatch.setattr(code_expand, "endpoints", lambda: [_ep("local")])

    async def raising_chat(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(code_expand, "chat", raising_chat)
    res = await code_expand.expand_code_query("some query about code")
    assert res == ["some query about code"]
