"""Code-specific query expansion via Groq — generates pseudo-code + API calls."""

from __future__ import annotations

import asyncio
import re

from config import _KeyRotator, get_http_client

_groq_rotator = _KeyRotator("GROQ_API_KEYS")
_next_key = _groq_rotator.next


CODE_KEYWORDS = re.compile(
    r"(function|class|method|api|library|sdk|import|export|module|"
    r"package|syntax|error|exception|type|interface|async|await|"
    r"callback|promise|stream|io|parse|validate|serialize|deploy)", re.I
)


async def expand_code_query(query: str) -> list[str]:
    """Expand a code/natural-language query into pseudo-code + API variations.

    Returns [original, variant1, variant2, ...] up to 4 total.
    Skips expansion for exact tokens (function names, error codes, paths).
    """
    stripped = query.strip()
    if not stripped:
        return []

    # Don't expand exact API/symbol queries — they need precision, not breadth
    if re.match(r'^[a-zA-Z_][\w.]*(::[\w.]+)*\(?\)?$', stripped):
        return [stripped]
    if stripped.startswith("pkg:") or stripped.startswith("CVE-") or stripped.startswith("GHSA-"):
        return [stripped]

    key = await _next_key()
    if not key:
        return [stripped]

    has_code = bool(CODE_KEYWORDS.search(stripped))
    lang_hint = "code and programming" if has_code else "developer tools and libraries"

    prompt = (
        f"You are a code search optimizer. Given a {lang_hint} query, generate "
        "2-3 alternative search queries that capture different angles:\n"
        "- One with specific API/library function names (e.g. 'dict.get()' for 'safe dictionary access')\n"
        "- One with the most common/widely-used library or framework for this task\n"
        "- One with minimal/broad terms for maximum recall\n\n"
        "Rules:\n"
        "- Keep each query under 80 chars\n"
        "- Do NOT include the original query\n"
        "- One query per line, plain text\n"
        "- Do NOT number or prefix\n"
        "- Be specific — prefer 'parse json in rust serde_json' over 'how to parse json'\n\n"
        f"Query: {stripped}"
    )

    try:
        import httpx
        c = get_http_client()
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 150,
            },
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            lines = []
            for q in text.split("\n"):
                raw = q.strip()
                if not raw or len(raw) <= 5 or raw.lower() == stripped.lower():
                    continue
                # Only strip leading markers (dash, asterisk, number prefix), not mid-word digits
                cleaned = re.sub(r'^[\s*\-•·>]+|^[\d]+[\.\)]\s*', '', raw).strip().strip('"\'[]')
                if not cleaned or len(cleaned) <= 5:
                    cleaned = raw
                if any(kw in cleaned.lower() for kw in [
                    "here", "variation", "query:", "---", "original",
                    "broad", "specific", "alternative",
                ]):
                    continue
                lines.append(cleaned)
            seen = {stripped.lower()}
            unique = []
            for q in lines:
                ql = q.lower()
                if ql not in seen and len(q) > 5:
                    seen.add(ql)
                    unique.append(q)
            if unique:
                return [stripped] + unique[:3]
    except Exception:
        pass
    return [stripped]
