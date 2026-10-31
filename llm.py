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
import time
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

# The same Beam model is served by the local opencode proxy as well. When the
# cmdcode2api gateway's daily quota is spent it answers 429 for the rest of the
# day, so the proxy is the endpoint that keeps summarisation working. Both are
# the in-tool model; only the gateway differs.
_PROXY_BASE_URL = "http://127.0.0.1:8787/v1"
_PROXY_MODEL = "beam-free"
_PROXY_DEFAULT_KEY = "local-dev-key"  # the local proxy's dev key

# A 429 that says the QUOTA is spent will not clear on the next attempt, so an
# endpoint that rate-limits us is retired for a while instead of being retried
# on every single call.
_COOLDOWN: dict[str, float] = {}
_COOLDOWN_DEFAULT = 600.0


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


def proxy_endpoint() -> dict:
    """The same Beam model behind the local opencode proxy, or {} when off.

    DISTILL_PROXY=off disables it; DISTILL_PROXY_BASE_URL / _MODEL / _API_KEY
    override the defaults.
    """
    if (os.environ.get("DISTILL_PROXY") or "on").strip().lower() in (
            "off", "none", "0", "false", "no"):
        return {}
    base_url = (os.environ.get("DISTILL_PROXY_BASE_URL")
                or _PROXY_BASE_URL).rstrip("/")
    model = os.environ.get("DISTILL_PROXY_MODEL") or _PROXY_MODEL
    key = ((os.environ.get("DISTILL_PROXY_API_KEY") or "").strip()
           or _PROXY_DEFAULT_KEY)
    return {"provider": "proxy", "base_url": base_url, "model": model,
            "api_key": key, "rotate": False}


def _ep_id(ep: dict) -> str:
    return f"{ep.get('base_url', '')}|{ep.get('model', '')}"


def _cooldown_secs() -> float:
    raw = os.environ.get("DISTILL_RATE_LIMIT_COOLDOWN", "")
    try:
        return max(0.0, float(raw)) if str(raw).strip() else _COOLDOWN_DEFAULT
    except (TypeError, ValueError):
        return _COOLDOWN_DEFAULT


def mark_rate_limited(ep: dict) -> None:
    """Retire an endpoint for a while after it answers 429."""
    secs = _cooldown_secs()
    if secs > 0:
        _COOLDOWN[_ep_id(ep)] = time.monotonic() + secs


def cooled_down(ep: dict) -> bool:
    until = _COOLDOWN.get(_ep_id(ep), 0.0)
    if until and time.monotonic() < until:
        return True
    if until:
        _COOLDOWN.pop(_ep_id(ep), None)
    return False


def reset_cooldowns() -> None:
    """Forget every cooldown (tests, and an explicit re-probe)."""
    _COOLDOWN.clear()


def endpoints() -> list[dict]:
    """Candidate endpoints, best first.

    CODESEARCH_SYNTH_PROVIDER: local (default) | groq | auto
    CODESEARCH_MODEL_FALLBACK: on (default) | off
      off - only the preferred endpoint is tried.
    DISTILL_ORDER: comma-separated preference, e.g. "proxy,local,groq".
    DISTILL_PROXY: on (default) | off - serve Beam from the local proxy too.

    Endpoints that recently answered 429 are moved to the back, and dropped
    entirely when at least one endpoint is still fresh: a spent daily quota is
    not worth a round trip on every call. If everything is cooling down the
    full list is returned, so a mis-set cooldown cannot silence the model.
    """
    prefer = (os.environ.get("CODESEARCH_SYNTH_PROVIDER") or "local").strip().lower()
    keys = groq_keys()
    groq = ({"provider": "groq", "base_url": GROQ_BASE_URL,
             "model": os.environ.get("GROQ_SMART_MODEL") or "openai/gpt-oss-120b",
             "api_key": "", "rotate": True}
            if keys else None)
    local = local_endpoint() or None
    proxy = proxy_endpoint() or None
    chain = {"local": local, "proxy": proxy, "groq": groq}
    default_order = ["local", "proxy", "groq"]
    if prefer == "groq":
        default_order = ["groq", "local", "proxy"]
    elif prefer == "proxy":
        default_order = ["proxy", "local", "groq"]
    wanted = [n.strip().lower() for n in
              (os.environ.get("DISTILL_ORDER") or "").split(",") if n.strip()]
    order = list(wanted or default_order)
    # A preference reorders the chain, it does not truncate it: anything not
    # named keeps its usual place at the end.
    order += [n for n in ("local", "proxy", "groq") if n not in order]
    eps = [chain[n] for n in order if chain.get(n)]
    fresh = [e for e in eps if not cooled_down(e)]
    if fresh:
        eps = fresh + [e for e in eps if cooled_down(e)]
    strict = (os.environ.get("CODESEARCH_MODEL_FALLBACK") or "on").strip().lower()
    if strict in ("off", "none", "0", "false", "no"):
        if prefer == "groq":
            return eps[:1]
        return [e for e in eps if e.get("provider") != "groq"]
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
        if "HTTP 429" in err:
            # Retire it so the next call does not pay this round trip again.
            mark_rate_limited(ep)
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
            # An exception with no message (httpx.ReadTimeout) used to render
            # as "ReadTimeout: " - a dangling colon the caller learns nothing
            # from. Name the failure instead.
            detail = str(e).strip()
            last = (f"{type(e).__name__}: {detail}" if detail
                    else f"{type(e).__name__} (no message)")
            continue
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (429, 500, 502, 503, 504):
                # A 429 is usually a spent quota rather than a hiccup: one
                # retry here, then chat() moves on to the next endpoint.
                if resp.status_code == 429 and attempt >= 2:
                    break
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
              "CODESEARCH_MODEL_FALLBACK", "DISTILL_PROXY", "DISTILL_ORDER",
              "DISTILL_RATE_LIMIT_COOLDOWN")}
    orig_local = globals()["local_endpoint"]
    orig_proxy = globals()["proxy_endpoint"]
    reset_cooldowns()
    try:
        # Keep the proxy out of the way until its own checks below.
        globals()["proxy_endpoint"] = lambda: {}
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

        # The proxy is a second gateway for the SAME model, not a different
        # model: it exists so a spent quota on one gateway cannot stop
        # summarisation.
        globals()["proxy_endpoint"] = orig_proxy
        os.environ.pop("DISTILL_PROXY", None)
        os.environ.pop("DISTILL_ORDER", None)
        os.environ.pop("CODESEARCH_SYNTH_PROVIDER", None)
        check("proxy sits behind local by default",
              [e["provider"] for e in endpoints()] == ["local", "proxy", "groq"])
        os.environ["DISTILL_ORDER"] = "proxy,local"
        check("DISTILL_ORDER overrides preference",
              [e["provider"] for e in endpoints()] == ["proxy", "local", "groq"])
        os.environ["DISTILL_PROXY"] = "off"
        check("DISTILL_PROXY=off removes it",
              "proxy" not in [e["provider"] for e in endpoints()])
        os.environ.pop("DISTILL_PROXY", None)
        os.environ.pop("DISTILL_ORDER", None)
        proxy_ep = [e for e in endpoints() if e["provider"] == "proxy"][0]
        mark_rate_limited(proxy_ep)
        check("a 429'd endpoint moves to the back",
              [e["provider"] for e in endpoints()] == ["local", "groq", "proxy"])
        os.environ["DISTILL_RATE_LIMIT_COOLDOWN"] = "0"
        reset_cooldowns()
        mark_rate_limited(proxy_ep)
        check("a zero cooldown retires nothing", _COOLDOWN == {})
        os.environ.pop("DISTILL_RATE_LIMIT_COOLDOWN", None)
        reset_cooldowns()
        globals()["proxy_endpoint"] = lambda: {}

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

        class SilentTimeout:
            async def post(self, *a, **kw): raise TimeoutError()

        _t, _f, err, _a = asyncio.run(fail_once(SilentTimeout(), ep, 100))
        check("a message-less exception is still legible",
              "TimeoutError" in err and "no message" in err
              and not err.rstrip().endswith(":"))

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
        globals()["proxy_endpoint"] = orig_proxy
        reset_cooldowns()
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
