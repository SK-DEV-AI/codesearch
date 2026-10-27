"""One place that decides which model answers, and counts the calls.

Adapted from websearch/llm.py for codesearch's smaller surface. Every caller
goes through `chat()`, which:

* prefers the in-tool model (local, free) and falls back to Groq (configurable
  with CODESEARCH_SYNTH_PROVIDER, same knob as synthesis),
* retries transient failures and escalates the token budget when a reasoning
  model returns empty content,
* counts the calls it makes, so a search response can report what it cost.

The counter is a contextvar so concurrent requests do not attribute each
other's calls.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import sys
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# A reasoning model emits reasoning_content BEFORE content, so a budget sized for
# the answer alone comes back as 200 + empty content. Floor every request at a
# budget that survives the reasoning preamble.
MIN_USEFUL_TOKENS = int(os.environ.get("CODESEARCH_MIN_MODEL_TOKENS") or 512)

_COUNTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_counts", default=None)

_LAST: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_last", default=None)

_JCODE_CONFIG = os.path.expanduser("~/.jcode/config.toml")
_DEFAULT_BASE_URL = "http://127.0.0.1:11435/v1"
_DEFAULT_MODEL = "reflection/beam:free"


def begin() -> None:
    """Start a fresh count for the current request context."""
    _COUNTS.set({})
    _LAST.set(None)


def counts() -> dict[str, int]:
    d = _COUNTS.get()
    return dict(d) if d else {}


def last_provider() -> str:
    d = _LAST.get()
    return (d or {}).get("provider", "") or ""


def last_model() -> str:
    d = _LAST.get()
    return (d or {}).get("model", "") or ""


def _bump(kind: str) -> None:
    d = _COUNTS.get()
    if d is None:
        return
    d[kind] = d.get(kind, 0) + 1


def groq_keys() -> list[str]:
    raw = os.environ.get("GROQ_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _config_key() -> str:
    """Read the local endpoint key from ~/.jcode/config.toml if present."""
    try:
        with open(_JCODE_CONFIG, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:  # noqa: BLE001
        return ""
    block = re.search(r"\[providers\.cmdcode2api\](.*?)(?:\n\[|$)", text, re.S)
    if not block:
        return ""
    m = re.search(r'^\s*api_key\s*=\s*"([^"]+)"', block.group(1), re.M)
    return m.group(1) if m else ""


def local_endpoint() -> dict:
    """The in-tool model as an endpoint, or {} when it is not configured."""
    base_url = (os.environ.get("DISTILL_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("DISTILL_MODEL") or _DEFAULT_MODEL
    key = (os.environ.get("DISTILL_API_KEY") or "").strip()
    if not key:
        key = _config_key()
    if not key:
        return {}
    return {"provider": "local", "base_url": base_url,
            "model": model, "api_key": key, "rotate": False}


def endpoints() -> list[dict]:
    """Candidate endpoints, best first.

    CODESEARCH_SYNTH_PROVIDER: local (default) | groq | auto
    CODESEARCH_MODEL_FALLBACK: on (default) | off
      off - only the preferred endpoint is tried.
    """
    prefer = (os.environ.get("CODESEARCH_SYNTH_PROVIDER") or "local").strip().lower()
    keys = groq_keys()
    groq = ({"provider": "groq", "base_url": GROQ_BASE_URL,
             "model": os.environ.get("GROQ_SMART_MODEL") or "openai/gpt-oss-120b",
             "api_key": "", "rotate": True}
            if keys else None)
    local = local_endpoint() or None
    order = [groq, local] if prefer == "groq" else [local, groq]
    eps = [e for e in order if e]
    strict = (os.environ.get("CODESEARCH_MODEL_FALLBACK") or "on").strip().lower()
    if strict in ("off", "none", "0", "false", "no"):
        return eps[:1]
    return eps


def _next_groq_key(offset: int = 0) -> str:
    keys = groq_keys()
    if not keys:
        return ""
    return keys[offset % len(keys)]


async def chat(messages: list[dict], *, max_tokens: int = 1200,
               attempts: int = 2, timeout: float = 30,
               kind: str = "chat", temperature: float = 0.0,
               model: str | None = None,
               client: Any = None) -> tuple[str, str, str, int]:
    """One OpenAI-compatible completion, with fallback across endpoints.

    Returns (text, finish_reason, error, attempts).
    """
    eps = endpoints()
    if not eps:
        return "", "", "no model endpoint configured", 0

    last = "no attempt made"
    total = 0
    for ep in eps:
        if model:
            ep = {**ep, "model": model}
        text, finish, err, att = await _one(ep, messages, max_tokens=max_tokens,
                                           attempts=attempts, timeout=timeout,
                                           temperature=temperature,
                                           client=client)
        total += att
        if text:
            _LAST.set({"provider": ep.get("provider", ""),
                       "model": ep.get("model", "")})
            _bump(kind)
            return text, finish, "", total
        last = err
        logger.warning("llm: %s call failed on %s: %s", kind, ep.get("provider"), err)
    return "", "", last, total


async def _one(ep: dict, messages: list[dict], *, max_tokens: int,
               attempts: int, timeout: float,
               temperature: float, client: Any = None) -> tuple[str, str, str, int]:
    url = ep["base_url"].rstrip("/") + "/chat/completions"
    last = "no attempt made"
    budget = max(max_tokens, MIN_USEFUL_TOKENS)
    ceiling = min(max(budget * 4, 4096), 32_000)
    for attempt in range(1, max(1, attempts) + 1):
        key = (_next_groq_key(attempt - 1) if ep.get("rotate")
               else ep.get("api_key", ""))
        try:
            if client is not None:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={"model": ep["model"], "messages": messages,
                          "temperature": temperature, "max_tokens": budget},
                    timeout=timeout)
            else:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    resp = await c.post(
                        url,
                        headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"},
                        json={"model": ep["model"], "messages": messages,
                              "temperature": temperature, "max_tokens": budget})
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (429, 500, 502, 503, 504):
                continue
            return "", "", last, attempt
        try:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            finish = choice.get("finish_reason") or ""
        except Exception as e:  # noqa: BLE001
            last = f"unparseable response: {type(e).__name__}: {e}"
            continue
        if text:
            return text, finish, "", attempt
        last = (f"empty content (finish_reason={finish}, max_tokens={budget})"
                if finish else f"empty content (max_tokens={budget})")
        if budget < ceiling:
            budget = min(budget * 2, ceiling)
            logger.info("llm: empty content, raising max_tokens to %d", budget)
    return "", "", last, max(1, attempts)


def _selfcheck() -> None:
    """Offline self-check: ordering, counting, failure classification."""
    ok = 0
    bad = []

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    saved = {k: os.environ.get(k) for k in
             ("CODESEARCH_SYNTH_PROVIDER", "GROQ_API_KEYS", "DISTILL_API_KEY",
              "CODESEARCH_MODEL_FALLBACK")}
    orig_local = globals()["local_endpoint"]
    try:
        globals()["local_endpoint"] = lambda: {"provider": "local", "base_url": "x",
                                               "model": "m", "api_key": "k",
                                               "rotate": False}
        os.environ.pop("CODESEARCH_SYNTH_PROVIDER", None)
        os.environ["GROQ_API_KEYS"] = "g1,g2"
        eps = endpoints()
        check("local first by default", [e["provider"] for e in eps] == ["local", "groq"])
        os.environ["CODESEARCH_SYNTH_PROVIDER"] = "groq"
        eps = endpoints()
        check("provider=groq promotes groq", [e["provider"] for e in eps] == ["groq", "local"])
        os.environ.pop("GROQ_API_KEYS")
        os.environ["CODESEARCH_SYNTH_PROVIDER"] = "auto"
        eps = endpoints()
        check("auto without groq yields local only", [e["provider"] for e in eps] == ["local"])
        os.environ["GROQ_API_KEYS"] = "g1"
        globals()["local_endpoint"] = lambda: {"provider": "local", "base_url": "x",
                                               "model": "m", "api_key": "k",
                                               "rotate": False}
        os.environ["CODESEARCH_MODEL_FALLBACK"] = "off"
        check("strict mode drops the fallback",
              [e["provider"] for e in endpoints()] == ["local"])
        os.environ["CODESEARCH_SYNTH_PROVIDER"] = "groq"
        check("strict mode keeps only the preferred lane",
              [e["provider"] for e in endpoints()] == ["groq"])
        os.environ["CODESEARCH_MODEL_FALLBACK"] = "on"
        check("fallback returns when re-enabled",
              len(endpoints()) == 2)
        os.environ.pop("CODESEARCH_MODEL_FALLBACK", None)

        os.environ.pop("GROQ_API_KEYS", None)
        globals()["local_endpoint"] = lambda: {}
        check("no endpoint yields empty list", endpoints() == [])

        async def no_ep() -> tuple[str, str, str, int]:
            begin()
            text, _f, err, _a = await chat([{"role": "user", "content": "x"}])
            return text, _f, err, _a

        text, _f, err, _a = asyncio.run(no_ep())
        check("no endpoint is reported", text == "" and "no model endpoint" in err)

        async def success() -> dict:
            begin()
            _bump("synthesis")
            _bump("synthesis")
            _bump("verify")
            return counts()

        c = asyncio.run(success())
        check("counts label kinds", c == {"synthesis": 2, "verify": 1})
        check("fresh context starts empty", counts() == {})

        async def two_requests() -> tuple[dict, dict]:
            async def one(kind: str) -> dict:
                begin()
                await asyncio.sleep(0)
                _bump(kind)
                return counts()
            return await asyncio.gather(one("synthesis"), one("verify"))

        a, b = asyncio.run(two_requests())
        check("concurrent requests do not share counts",
              a == {"synthesis": 1} and b == {"verify": 1})

        class FakeResp:
            status_code = 429
            text = "rate limited"

        class FakeClient:
            def __init__(self, resp): self.resp = resp
            async def post(self, *a, **kw): return self.resp

        class RaisingClient:
            async def post(self, *a, **kw): raise OSError("connection reset")

        async def fail_once(client, ep, budget, attempts=1) -> tuple[str, str, str, int]:
            begin()
            return await _one(ep, [{"role": "user", "content": "x"}],
                              max_tokens=budget, attempts=attempts, timeout=1,
                              temperature=0.0, client=client)

        ep = {"provider": "local", "base_url": "http://x", "model": "m",
              "api_key": "k", "rotate": False}
        _t, _f, err, att = asyncio.run(fail_once(FakeClient(FakeResp()), ep, 100))
        check("429 is reported with its status", err.startswith("HTTP 429") and att == 1)
        _t, _f, err, _a = asyncio.run(fail_once(RaisingClient(), ep, 100))
        check("transport error is named", "OSError" in err)

        class EmptyThenFull:
            def __init__(self): self.budgets = []
            async def post(self, url, headers=None, json=None, timeout=None):
                self.budgets.append(json["max_tokens"])
                class R:
                    def __init__(self, body): self.status_code = 200; self._b = body
                    def json(self): return self._b
                if len(self.budgets) == 1:
                    return R({"choices": [{"message": {"content": ""},
                                           "finish_reason": "length"}]})
                return R({"choices": [{"message": {"content": "answer"},
                                       "finish_reason": "stop"}]})

        c2 = EmptyThenFull()
        text, _f, err, att = asyncio.run(fail_once(c2, ep, 100, attempts=2))
        check("empty content recovers with a bigger budget", text == "answer")
        check("budget was raised, not reused", c2.budgets[1] > c2.budgets[0])
        check("a tiny budget is raised to the usable floor",
              c2.budgets[0] >= MIN_USEFUL_TOKENS)

        class AlwaysEmpty:
            def __init__(self): self.budgets = []
            async def post(self, url, headers=None, json=None, timeout=None):
                self.budgets.append(json["max_tokens"])
                class R:
                    status_code = 200
                    def json(self): return {"choices": [{"message": {"content": ""},
                                                         "finish_reason": "length"}]}
                return R()

        c4 = AlwaysEmpty()
        text, _f, err, att = asyncio.run(fail_once(c4, ep, 64, attempts=3))
        check("empty content is reported, not returned as success", text == "")
        check("the failure names the budget", "max_tokens" in err)
        check("budget grows on each retry", c4.budgets == sorted(c4.budgets) and c4.budgets[-1] > c4.budgets[0])

        os.environ["GROQ_API_KEYS"] = "a,b,c"
        keys = {_next_groq_key(i) for i in range(3)}
        check("groq keys rotate", keys == {"a", "b", "c"})

        async def counted_ep() -> dict:
            begin()
            p = {"provider": "local", "base_url": "http://x", "model": "m",
                 "api_key": "k", "rotate": False}
            await chat([{"role": "user", "content": "x"}],
                       client=FakeClient(type("R", (), {"status_code": 200,
                                                        "json": lambda self: {"choices": [
                                                            {"message": {"content": "hi"}}]}})()),
                       kind="synthesis")
            return counts()

        globals()["endpoints"] = lambda: [{"provider": "local", "base_url": "http://x",
                                           "model": "m", "api_key": "k", "rotate": False}]
        c3 = asyncio.run(counted_ep())
        check("successful chat bumps its kind", c3 == {"synthesis": 1})
    finally:
        globals()["local_endpoint"] = orig_local
        globals().pop("endpoints", None)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if bad:
        print(f"llm selfcheck: {ok}/{ok + len(bad)} FAILED: {', '.join(bad)}")
        raise SystemExit(1)
    print(f"llm selfcheck: {ok}/{ok} OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()