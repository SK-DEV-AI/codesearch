"""Progressive query relaxation for the global code-search lanes.

Both keyless global lanes AND their terms (grep.app's `searchGitHub` and
Sourcegraph's `context:global` search), so a natural-language query from an
agent — "httpx vs aiohttp benchmark", "sqlite fts5 bm25 ranking tuning" —
matches nothing even though each term on its own is common. Measured
2026-10-31 against both lanes: the 4-5 word forms returned 0 hits while the
2-3 word forms returned 5-10.

`relax_queries` therefore yields shorter variants, in order, so a lane can
retry once or twice before reporting an honest empty result. The lane keeps
`query_requested` alongside the relaxed query it actually ran, so the caller
can see the substitution rather than guessing why results look loose.
"""

from __future__ import annotations

import re

# Filler that a code index does not contain as identifiers, plus question
# words an agent tends to leave in. Stripped before shortening, never used to
# replace a term.
STOPWORDS = frozenset("""
a an the and or of for to in on with about vs versus how what why when which
best good top using use example examples code please find search show give me
""".split())

_PUNCT = "`'\",;:()[]{}!?"


def _clean(tok: str) -> str:
    return tok.strip(_PUNCT).lower()


def relax_queries(query: str, max_terms: int = 3) -> list[str]:
    """Return shorter variants of `query`, best first. Original never returned.

    Empty when the query is already two words or fewer (nothing to drop), which
    is also the shape both lanes answer reliably.
    """
    toks = [t for t in re.split(r"\s+", (query or "").strip()) if t]
    if len(toks) < 3:
        return []
    content = [t for t in toks if _clean(t) not in STOPWORDS] or toks
    seen = {" ".join(toks).lower()}
    out: list[str] = []
    # 3 content terms, then the leading 2. Two attempts max: each lane call
    # costs a round trip, and a 2-term query is where both lanes answer.
    for n in (max_terms, 2):
        cand = " ".join(content[:n])
        key = cand.lower()
        if len(cand.split()) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out
