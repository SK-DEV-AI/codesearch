import asyncio
from types import SimpleNamespace

import httpx
import pytest

import config
import hackernews
import httpretry
import libraries_io
import semantic_scholar
import sofa


class FakeResponse:
    def __init__(self, status_code=200, headers=None, body=b"{}"):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    @property
    def text(self):
        return self._body.decode("utf-8", "replace")

    def json(self):
        import json
        return json.loads(self._body or b"{}")

    @property
    def reason_phrase(self):
        return ""


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def get(self, *a, **k):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    async def post(self, *a, **k):
        return await self.get(*a, **k)


def _patch(monkeypatch, client):
    monkeypatch.setattr(config, "get_http_client", lambda: client)
    # avoid real sleeps
    async def _sleep(x):
        return None
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    # disable caching to keep tests independent
    async def _cached(k):
        return None
    async def _set(k, v, ttl=None):
        return None
    monkeypatch.setattr(libraries_io, "_cached", _cached)
    monkeypatch.setattr(libraries_io, "_set_cache", _set)
    monkeypatch.setattr(hackernews, "_cached", _cached)
    monkeypatch.setattr(hackernews, "_set_cache", _set)


@pytest.mark.asyncio
async def test_retry_on_503_then_200_libraries(monkeypatch):
    body = b'[{"name": "x", "platform": "npm"}]'
    client = FakeClient([FakeResponse(503), FakeResponse(200, body=body)])
    _patch(monkeypatch, client)
    monkeypatch.setattr(libraries_io, "LI_KEY", "key")
    # bypass key rotator
    async def _next():
        return "key"
    monkeypatch.setattr(libraries_io, "_next_li_key", _next)
    r = await libraries_io.libraries_io_search("q")
    assert r["success"] is True
    assert r["retries"] == 1
    assert client.calls == 2


@pytest.mark.asyncio
async def test_no_retry_on_400_libraries(monkeypatch):
    client = FakeClient([FakeResponse(400, body=b'{"error":"bad"}')])
    _patch(monkeypatch, client)
    monkeypatch.setattr(libraries_io, "LI_KEY", "key")
    async def _next():
        return "key"
    monkeypatch.setattr(libraries_io, "_next_li_key", _next)
    r = await libraries_io.libraries_io_search("q")
    assert r["success"] is False
    assert r["retries"] == 0
    assert client.calls == 1


@pytest.mark.asyncio
async def test_exhausted_retries_surfaces_error(monkeypatch):
    client = FakeClient([FakeResponse(503), FakeResponse(503), FakeResponse(503)])
    _patch(monkeypatch, client)
    monkeypatch.setattr(libraries_io, "LI_KEY", "key")
    async def _next():
        return "key"
    monkeypatch.setattr(libraries_io, "_next_li_key", _next)
    r = await libraries_io.libraries_io_search("q")
    assert r["success"] is False
    assert r["retries"] == 2
    assert "503" in r["error"]
    assert client.calls == 3


@pytest.mark.asyncio
async def test_retry_after_honoured(monkeypatch):
    sleeps = []
    async def _sleep(x):
        sleeps.append(x)
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    body = b'{"hits": [], "nbHits": 0, "page": 0, "nbPages": 0}'
    client = FakeClient([
        FakeResponse(429, headers={"Retry-After": "2"}, body=b""),
        FakeResponse(200, body=body),
    ])
    monkeypatch.setattr(config, "get_http_client", lambda: client)
    # avoid cache
    async def _cached(k):
        return None
    async def _set(k, v, ttl=None):
        return None
    monkeypatch.setattr(hackernews, "_cached", _cached)
    monkeypatch.setattr(hackernews, "_set_cache", _set)
    r = await hackernews.search_hn("q")
    assert r["success"] is True
    assert r["retries"] == 1
    assert any(s == 2 or s == 2.0 for s in sleeps)


@pytest.mark.asyncio
async def test_400_no_retry_hn(monkeypatch):
    client = FakeClient([FakeResponse(400, body=b"")])
    monkeypatch.setattr(config, "get_http_client", lambda: client)
    async def _cached(k):
        return None
    async def _set(k, v, ttl=None):
        return None
    monkeypatch.setattr(hackernews, "_cached", _cached)
    monkeypatch.setattr(hackernews, "_set_cache", _set)
    async def _sleep(x):
        return None
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    r = await hackernews.search_hn("q")
    assert r["success"] is False
    assert r["retries"] == 0
    assert client.calls == 1


@pytest.mark.asyncio
async def test_httpretry_transient_statuses():
    # direct helper test
    client = FakeClient([FakeResponse(503), FakeResponse(200, body=b"{}")])
    resp, retries = await httpretry.retry_request("GET", "http://x", client=client, timeout=5)
    assert resp.status_code == 200
    assert retries == 1


@pytest.mark.asyncio
async def test_httpretry_4xx_no_retry():
    client = FakeClient([FakeResponse(404)])
    resp, retries = await httpretry.retry_request("GET", "http://x", client=client, timeout=5)
    assert resp.status_code == 404
    assert retries == 0
    assert client.calls == 1