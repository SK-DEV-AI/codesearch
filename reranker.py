"""Shared reranker bridge — connects to Unix socket managed by websearch."""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

_SOCKET_PATH = "/tmp/reranker_worker.sock"
_RERANKER_KILLSWITCH = os.path.expanduser("~/.local/share/reranker-rust/disabled")


def killswitch_active() -> bool:
    """True when the reranker killswitch file exists — reranking is off (frees
    GPU VRAM). Enable with: rm ~/.local/share/reranker-rust/disabled"""
    return os.path.exists(_RERANKER_KILLSWITCH)

_LOCK = asyncio.Lock()
_READER = None
_WRITER = None


async def _close_connection():
    """Close the current connection so the worker can accept new ones."""
    global _READER, _WRITER
    if _READER is not None:
        try:
            _READER.feed_eof()
        except Exception:
            pass
        _READER = None
    if _WRITER is not None:
        try:
            _WRITER.close()
            await _WRITER.wait_closed()
        except Exception:
            pass
        _WRITER = None


async def _is_healthy() -> bool:
    """Stat-based health check — no wire ping, so a busy worker (40-120s
    rerank) never stalls other agents' checks or queues stale ping lines."""
    global _WRITER, _READER
    return (
        _WRITER is not None
        and _READER is not None
        and os.path.exists(_SOCKET_PATH)
    )


async def _ensure_worker():
    global _READER, _WRITER
    await _close_connection()
    for _ in range(50):
        try:
            _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH, limit=2**20)
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            await asyncio.sleep(0.1)
    return False


async def warmup() -> bool:
    if killswitch_active():
        return False
    async with _LOCK:
        global _READER, _WRITER
        if await _is_healthy():
            return True
        if not await _ensure_worker():
            return False
        try:
            req = json.dumps({"query": "warmup", "passages": [{"snippet": "warmup"}], "top_k": 1})
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
            result = json.loads(r)
            return not result.get("error")
        except Exception:
            await _close_connection()
            return False


def fallback_sort(passages: list[dict], top_k: int) -> list[dict]:
    """Sort passages by best available relevance score when reranker fails."""
    scored = []
    for p in passages:
        score = p.get("_rel") or p.get("_relevance") or p.get("_hybrid") or 0
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


async def rerank(query: str, passages: list[dict], top_k: int = 20) -> list[dict]:
    if not passages:
        return []
    if killswitch_active():
        return fallback_sort(passages, top_k)
    async with _LOCK:
        global _READER, _WRITER
        if not await _is_healthy() and not await _ensure_worker():
            return fallback_sort(passages, top_k)
        normalized = []
        for p in passages:
            item = dict(p)
            text = item.get("snippet") or item.get("text") or item.get("content") or item.get("full_content") or ""
            item["snippet"] = text[:32768]
            normalized.append(item)
        req = json.dumps({"query": query, "passages": normalized, "top_k": top_k})
        try:
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
        except (BrokenPipeError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: write failed: {e}")
            await _close_connection()
            return fallback_sort(passages, top_k)
        try:
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: read failed: {e}")
            await _close_connection()
            return fallback_sort(passages, top_k)
        try:
            result = json.loads(r)
        except json.JSONDecodeError:
            logger.warning("reranker: invalid JSON response from worker")
            await _close_connection()
            return fallback_sort(passages, top_k)
        if result.get("error"):
            logger.warning(f"reranker error: {result['error']}")
            return fallback_sort(passages, top_k)
        scored = result.get("scores", [])
        for s in scored:
            if "score" in s:
                s["_rerank"] = s.pop("score")
        return scored
