from __future__ import annotations

import asyncio
import math
import re
from collections import Counter, defaultdict

import httpx

from config import NV_BASE, NV_EMBED_MODEL, NV_KEY, _cached, _set_cache, _KeyRotator, get_http_client

_nv_rotator = _KeyRotator("NV_KEY")
_next_nv_key = _nv_rotator.next


async def _embed(texts: list[str], input_type: str = "passage") -> list[list[float]] | None:
    if not _nv_rotator.has_keys:
        return None
    results: list[list[float] | None] = [None] * len(texts)
    uncached = []
    uncached_idx = []
    for i, t in enumerate(texts):
        k = f"emb:{input_type}:{t[:200]}"
        c = await _cached(k)
        if c is not None:
            results[i] = c
        else:
            uncached.append(t)
            uncached_idx.append(i)
    if uncached:
        nv_key = await _next_nv_key()
        try:
            c = get_http_client()
            r = await c.post(f"{NV_BASE}/embeddings", json={
                "model": NV_EMBED_MODEL, "input": uncached, "input_type": input_type, "encoding_format": "float",
                "truncate": "END",
            }, headers={"Authorization": f"Bearer {nv_key}"})
            if r.status_code == 200:
                data = r.json()
                for idx, row in zip(range(len(uncached)), data.get("data", [])):
                    emb = row.get("embedding")
                    if emb:
                        orig_idx = uncached_idx[idx]
                        results[orig_idx] = emb
                        await _set_cache(f"emb:{input_type}:{uncached[idx][:200]}", emb)
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    final = [r for r in results if r is not None]
    return final if len(final) == len(texts) else None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a))
    nb = math.sqrt(sum(bi * bi for bi in b))
    return dot / (na * nb + 1e-10)


def _cluster_by_source(results: list[dict]) -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        src = r.get("source", "unknown")
        clusters[src].append(r)
    return dict(clusters)


def _dedup_within_cluster(items: list[dict], sim_threshold: float = 0.85) -> list[dict]:
    if not items:
        return items
    deduped: list[dict] = []
    kept_embeds: list[list[float]] = []
    for r in items:
        emb = r.get("_embedding")
        if emb is None:
            key = str(r.get("title", r.get("full_name", r.get("file", ""))))[:100]
            if key and not any(str(d.get("title", d.get("full_name", d.get("file", ""))))[:100] == key for d in deduped):
                deduped.append(r)
            continue
        is_dup = False
        for ke in kept_embeds:
            if _cosine_sim(emb, ke) > sim_threshold:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)
            kept_embeds.append(emb)
    return deduped


# ── BM25 hybrid for exact keyword/syntax matching ──

_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]\w*|[(){}\[\];:.,<>!?=+\-*/&|^~%#@`]", text.lower())


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], doc_len: int, avg_dl: float, idf_cache: dict[str, float]) -> float:
    doc_freq: Counter[str] = Counter(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        tf = doc_freq.get(qt, 0)
        if tf == 0:
            continue
        idf = idf_cache.get(qt, 0.0)
        if idf <= 0:
            continue
        score += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * doc_len / (avg_dl or 1)))
    return score


def _hybrid_rank(results: list[dict], query: str, emb_weight: float = 0.4, bm25_weight: float = 0.6) -> list[dict]:
    """Combine embedding relevance with BM25 keyword score via min-max normalized weighted sum."""
    if not results or not query:
        return results
    q_tokens = _tokenize(query)
    if not q_tokens:
        return results
    N = len(results)

    doc_tokens_list: list[list[str]] = []
    doc_lengths: list[int] = []
    for r in results:
        doc_text = r.get("title", "") + " " + r.get("text", r.get("snippet", ""))
        tokens = _tokenize(doc_text)
        doc_tokens_list.append(tokens)
        doc_lengths.append(len(tokens))

    avg_dl = sum(doc_lengths) / max(N, 1)
    q_set = set(q_tokens)

    df_counts: dict[str, int] = {}
    for tokens in doc_tokens_list:
        seen = set()
        for t in tokens:
            if t in q_set and t not in seen:
                df_counts[t] = df_counts.get(t, 0) + 1
                seen.add(t)

    idf_cache: dict[str, float] = {}
    for qt in q_set:
        df = df_counts.get(qt, 0)
        idf_cache[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1.0) if df > 0 else 0.0

    raw_scores: list[float] = []
    for i, r in enumerate(results):
        score = _bm25_score(q_tokens, doc_tokens_list[i], doc_lengths[i], avg_dl, idf_cache)
        r["_bm25"] = score
        raw_scores.append(score)

    min_bm = min(raw_scores) if raw_scores else 0.0
    max_bm = max(raw_scores) if raw_scores else 0.0
    bm_range = max_bm - min_bm if max_bm > min_bm else 1.0

    for r in results:
        rel = r.get("_relevance", 0) or 0
        bm = r.get("_bm25", 0)
        bm_norm = (bm - min_bm) / bm_range
        r["_hybrid"] = rel * emb_weight + bm_norm * bm25_weight

    results.sort(key=lambda x: x.get("_hybrid", 0), reverse=True)
    return results


def _dedup_rank(results: list[dict], query_embed: list[float] | None, sim_threshold: float = 0.85) -> list[dict]:
    if not results:
        return results
    nv_embeds = [r.get("_embedding") for r in results]
    has_embeds = query_embed is not None and all(e is not None for e in nv_embeds)

    if has_embeds:
        clusters = _cluster_by_source(results)
        deduped: list[dict] = []
        for src, items in clusters.items():
            cluster_deduped = _dedup_within_cluster(items, sim_threshold)
            deduped.extend(cluster_deduped)

        for r in deduped:
            emb = r.get("_embedding")
            if emb is not None and query_embed is not None:
                r["_relevance"] = round(_cosine_sim(query_embed, emb), 4)

        deduped.sort(key=lambda x: x.get("_relevance", 0), reverse=True)
    else:
        seen_titles: set[str] = set()
        deduped = []
        for r in results:
            key = str(r.get("title", r.get("full_name", r.get("file", ""))))[:100]
            if key and key not in seen_titles:
                seen_titles.add(key)
                deduped.append(r)

    for r in deduped:
        r.pop("_embedding", None)
    return deduped
