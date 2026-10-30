from __future__ import annotations

import asyncio

from config import get_http_client, api_error
from embed import _embed, _dedup_rank, _hybrid_rank
from reranker import rerank as _rerank, fallback_sort, killswitch_active
from security import SecurityError, safe_fetch, safe_stream, validate_url as _validate_url

def _html_to_text(raw: str) -> str:
    """Convert HTML to readable text. Prefer trafilatura; fall back to tag stripping."""
    if "<" not in raw:
        return raw
    try:
        import trafilatura
        extracted = trafilatura.extract(raw, include_comments=False, include_tables=True,
                                       output_format="markdown")
        if extracted and extracted.strip():
            return extracted
    except Exception:
        pass
    # Fallback: strip tags crudely.
    import re as _re
    text = _re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=_re.S | _re.I)
    text = _re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=_re.S | _re.I)
    text = _re.sub(r"<[^>]+>", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text
# JS app shell (needs a browser to render). codesearch has no CDP/browser by
# design (websearch owns port 9222) — flag it honestly instead of feeding the
# shell to ranking as if it were content.
_SHELL_MARKERS = ("__NEXT_DATA__", "gtm.start", "__NUXT__", "id=\"root\"", "id='root'")


def _is_js_shell(body: bytes) -> bool:
    """True when the body looks like an unrendered JS app shell: tiny text
    with app-bootstrap markers in the head. Same signal websearch uses to
    escalate to CDP (raw_html[:2000] marker check)."""
    if len(body) > 20000:
        return False
    try:
        head = body[:2000].decode("utf-8", errors="replace")
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return False
    # strip tags crudely: if visible text is tiny but markers present, shell
    import re as _re
    visible = _re.sub(r"<[^>]+>", " ", text)
    visible = _re.sub(r"\s+", " ", visible).strip()
    return len(visible) < 500 and any(m in head for m in _SHELL_MARKERS)


async def enrich_results(
    query: str,
    results: list[dict],
    top_k: int = 20,
    max_fetch_size: int | None = 50,
    max_urls: int = 5,
    max_chars: int = 4000,
    include_html: bool = False,
) -> dict:
    """Fetch full content for search results, deduplicate by embedding, rerank by relevance."""
    if not results:
        return {"success": True, "results": [], "reason": "no results to enrich"}

    # Effective fetch cap: honour both legacy max_fetch_size and the new max_urls cap.
    fetch_cap = len(results)
    if max_fetch_size is not None:
        fetch_cap = min(fetch_cap, max(1, int(max_fetch_size)))
    fetch_cap = min(fetch_cap, max(1, int(max_urls)))

    fetcher = get_http_client()
    sem = asyncio.Semaphore(8)
    deadline = asyncio.get_running_loop().time() + 20

    async def _fetch_one(r: dict) -> dict:
        url = str(r.get("url", ""))
        if not url:
            r["__fetch_error"] = "no url"
            return r
        # SSRF validation
        try:
            url = await _validate_url(url)
        except SecurityError as e:
            r["__fetch_error"] = str(e)
            return r
        async with sem:
            if asyncio.get_running_loop().time() > deadline:
                r["__fetch_error"] = "deadline"
                return r
            try:
                # M4: stream + hard cap via safe_stream (per-hop SSRF kept)
                body = await safe_stream(
                    fetcher, url, max_bytes=100000 if include_html else 50000,
                    timeout=10)
                if _is_js_shell(body):
                    # Not a fetch failure — the page needs a browser (websearch
                    # fetch renders via CDP). Flag it so callers know.
                    r["__fetch_error"] = ("js-shell: page needs browser rendering "
                        "(use websearch fetch); plain-HTTP body kept in full_content")
                raw = body.decode("utf-8", errors="replace")
                if include_html:
                    r["full_content"] = raw
                    r["__content_is_html"] = True
                else:
                    r["full_content"] = _html_to_text(raw)
                    r["__content_is_html"] = False
            except Exception as e:
                r["__fetch_error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return r

    fetch_tasks = [asyncio.create_task(_fetch_one(r)) for r in results[:fetch_cap]]
    fetched = await asyncio.gather(*fetch_tasks)
    fetched += results[fetch_cap:]

    # Dedup by NIM embedding
    texts = []
    for r in fetched:
        t = r.get("full_content") or r.get("text") or r.get("snippet") or r.get("title", "")
        texts.append(t)
    q_emb = await _embed([query], "query")
    if q_emb:
        p_emb = await _embed(texts, "passage")
        if p_emb:
            for r, emb in zip(fetched, p_emb):
                r["_embedding"] = emb
            deduped = _dedup_rank(fetched, q_emb[0])
            deduped = _hybrid_rank(deduped, query)
        else:
            deduped = fetched
    else:
        deduped = fetched

    # Rerank
    try:
        reranked = await _rerank(query, deduped, top_k=min(top_k, len(deduped)))
    except Exception:
        reranked = fallback_sort(deduped, top_k)

    result_list: list[dict] = []
    for r in reranked:
        raw_content = (r.get("full_content", "") or r.get("text", "") or "")
        truncated = len(raw_content) > max_chars
        content = raw_content[:max_chars]
        entry: dict = {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": content,
            "truncated": truncated,
        }
        if include_html or r.get("__content_is_html"):
            entry["content_is_html"] = True
        else:
            entry["content_is_html"] = False
        source = r.get("source") or r.get("engine") or ""
        if source:
            entry["source"] = source
        fetch_err = r.get("__fetch_error")
        if fetch_err:
            entry["fetch_error"] = fetch_err
        score = r.get("_rerank") or r.get("_relevance") or r.get("_hybrid")
        if score:
            entry["relevance_score"] = round(score, 4)
        result_list.append(entry)

    total_fetched = sum(1 for r in fetched if r.get("full_content") and not r.get("__fetch_error"))
    total_errors = sum(1 for r in fetched if r.get("__fetch_error"))
    # Honest success: at least one fetch succeeded (or no input). All-failed => false.
    success = (total_fetched > 0) or (len(results) == 0)

    result: dict = {
        "success": success,
        "total_input": len(results),
        "total_enriched": len(result_list),
        "total_fetched": total_fetched,
        "total_errors": total_errors,
        "max_chars_per_url": max_chars,
        "max_urls": fetch_cap,
        "enriched": result_list,
    }
    if not success and total_errors > 0:
        result["error"] = "all fetches failed (see fetch_error per entry)"
    if killswitch_active():
        result["reranker"] = "disabled — results are engine-ranked only (not reranked). " \
            "Enable with: rm ~/.local/share/reranker-rust/disabled"
    return result
