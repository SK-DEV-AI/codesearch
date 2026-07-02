from __future__ import annotations

import asyncio

from config import get_http_client
from embed import _embed, _dedup_rank, _hybrid_rank
from reranker import rerank as _rerank


async def enrich_results(
    query: str,
    results: list[dict],
    top_k: int = 20,
    max_fetch_size: int | None = 50,
    include_html: bool = False,
) -> dict:
    """Fetch full content for search results, deduplicate by embedding, rerank by relevance."""
    if not results:
        return {"success": False, "error": "no results to enrich", "enriched": []}

    fetcher = get_http_client()
    sem = asyncio.Semaphore(8)
    deadline = asyncio.get_event_loop().time() + 20

    async def _fetch_one(r: dict) -> dict:
        url = str(r.get("url", ""))
        if not url:
            r["__fetch_error"] = "no url"
            return r
        async with sem:
            if asyncio.get_event_loop().time() > deadline:
                r["__fetch_error"] = "deadline"
                return r
            try:
                resp = await fetcher.get(url, timeout=10, follow_redirects=True)
                if resp.status_code == 200:
                    text = resp.text
                    r["full_content"] = text[:100000]
                    if not include_html:
                        r["full_content"] = text[:50000]
                else:
                    r["__fetch_error"] = f"HTTP {resp.status_code}"
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
        reranked = deduped[:top_k]

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
        score = r.get("_relevance") or r.get("_hybrid")
        if score:
            entry["relevance_score"] = round(score, 4)
        result_list.append(entry)

    return {
        "success": True,
        "total_input": len(results),
        "total_enriched": len(result_list),
        "total_fetched": sum(1 for r in fetched if r.get("full_content") and not r.get("__fetch_error")),
        "total_errors": sum(1 for r in fetched if r.get("__fetch_error")),
        "enriched": result_list,
    }
