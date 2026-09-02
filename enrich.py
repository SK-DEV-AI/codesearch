from __future__ import annotations

import asyncio

from config import get_http_client, api_error
from embed import _embed, _dedup_rank, _hybrid_rank
from reranker import rerank as _rerank, fallback_sort, killswitch_active
from security import SecurityError, safe_fetch, safe_stream, validate_url as _validate_url


async def enrich_results(
    query: str,
    results: list[dict],
    top_k: int = 20,
    max_fetch_size: int | None = 50,
    include_html: bool = False,
) -> dict:
    """Fetch full content for search results, deduplicate by embedding, rerank by relevance."""
    if not results:
        return {"success": True, "results": [], "reason": "no results to enrich"}

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
                r["full_content"] = body.decode("utf-8", errors="replace")
            except Exception as e:
                r["__fetch_error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return r

    limit = max_fetch_size or len(results)
    fetch_tasks = [asyncio.create_task(_fetch_one(r)) for r in results[:limit]]
    fetched = await asyncio.gather(*fetch_tasks)
    fetched += results[limit:]

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
        entry: dict = {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": (r.get("full_content", "") or r.get("text", "") or "")[:10000],
        }
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

    result: dict = {
        "success": True,
        "total_input": len(results),
        "total_enriched": len(result_list),
        "total_fetched": sum(1 for r in fetched if r.get("full_content") and not r.get("__fetch_error")),
        "total_errors": sum(1 for r in fetched if r.get("__fetch_error")),
        "enriched": result_list,
    }
    if killswitch_active():
        result["reranker"] = "disabled — results are engine-ranked only (not reranked). " \
            "Enable with: rm ~/.local/share/reranker-rust/disabled"
    return result
