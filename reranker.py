"""Shared reranker bridge — connects to Unix socket managed by free-websearch."""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

_SOCKET_PATH = "/tmp/reranker_worker.sock"

_LOCK = asyncio.Lock()
_READER = None
_WRITER = None


async def _ensure_worker():
    global _READER, _WRITER
    try:
        _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False


async def warmup() -> bool:
    async with _LOCK:
        global _READER, _WRITER
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
            return False


async def rerank(query: str, passages: list[dict], top_k: int = 20) -> list[dict]:
    if not passages:
        return []
    async with _LOCK:
        global _READER, _WRITER
        if not await _ensure_worker():
            return passages[:top_k]
        normalized = []
        for p in passages:
            item = dict(p)
            text = item.get("snippet") or item.get("text") or item.get("content") or ""
            item["snippet"] = text[:8000]
            normalized.append(item)
        req = json.dumps({"query": query, "passages": normalized, "top_k": top_k})
        try:
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
        except (BrokenPipeError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: write failed: {e}")
            _WRITER = None
            return passages[:top_k]
        try:
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: read failed: {e}")
            _WRITER = None
            return passages[:top_k]
        try:
            result = json.loads(r)
        except json.JSONDecodeError:
            return passages[:top_k]
    if result.get("error"):
        logger.warning(f"reranker error: {result['error']}")
        return passages[:top_k]
    scored = result.get("scores", [])
    for s in scored:
        if "score" in s:
            s["_rerank"] = s.pop("score")
    return scored
