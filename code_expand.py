"""Code-specific query expansion via LLM — generates pseudo-code + API calls."""

from __future__ import annotations

import logging
import re

from llm import chat, endpoints

logger = logging.getLogger(__name__)


CODE_KEYWORDS = re.compile(
    r"(function|class|method|api|library|sdk|import|export|module|"
    r"package|syntax|error|exception|type|interface|async|await|"
    r"callback|promise|stream|io|parse|validate|serialize|deploy)", re.I
)


async def expand_code_query(query: str) -> list[str]:
    """Expand a code/natural-language query into pseudo-code + API variations.

    Returns [original, variant1, variant2, ...] up to 4 total.
    Skips expansion for exact tokens (function names, error codes, paths).
    No-op (returns [query]) when no model endpoint is configured.
    """
    stripped = query.strip()
    if not stripped:
        return []

    # Don't expand exact API/symbol queries — they need precision, not breadth
    if re.match(r'^[a-zA-Z_][\w.]*(::[\w.]+)*\(?\)?$', stripped):
        return [stripped]
    if stripped.startswith("pkg:") or stripped.startswith("CVE-") or stripped.startswith("GHSA-"):
        return [stripped]

    if not endpoints():
        return [stripped]

    has_code = bool(CODE_KEYWORDS.search(stripped))
    lang_hint = "code and programming" if has_code else "developer tools and libraries"

    prompt = (
        f"Given a {lang_hint} query, output exactly 2 alternative keyword-only "
        "search queries. Each query is one line of 3-6 keywords. "
        "No questions. No sentences. No numbering. No prefixes. No explanation.\n\n"
        f"Query: {stripped}"
    )

    try:
        text, _finish, err, _att = await chat(
            [{"role": "system", "content": "Generate exactly 2 keyword-only search queries from the user's input. One query per line. No numbering. No prefixes. No explanations."},
             {"role": "user", "content": prompt}],
            max_tokens=128, attempts=2, timeout=15, kind="expand", temperature=0.7,
        )
        if not text:
            if err:
                logger.warning("expand_code_query: %s", err)
            return [stripped]
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
    except Exception as e:  # noqa: BLE001
        logger.warning("expand_code_query failed: %s", e)
    return [stripped]
