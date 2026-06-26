from __future__ import annotations

import asyncio
import json
import urllib.parse
from typing import Any

import httpx

from config import get_http_client

CODEWIKI_URL = "https://codewiki.google/_/BoqAngularSdlcAgentsUi/data/batchexecute"
MAX_CODEWIKI_RETRIES = 3
BASE_DELAY = 1.0


def _collect_wrb_frames(node: Any, out: list[dict]) -> None:
    if not isinstance(node, list):
        return
    if len(node) >= 3 and node[0] == "wrb.fr" and isinstance(node[1], str):
        raw = node[2]
        payload = json.loads(raw) if isinstance(raw, str) else raw
        out.append({"rpcId": node[1], "payload": payload})
    for child in node:
        _collect_wrb_frames(child, out)


def _extract_wrb_frames(text: str) -> list[dict]:
    trimmed = text.lstrip()
    if trimmed.startswith(")]}'"):
        trimmed = trimmed[4:].lstrip()
    frames: list[dict] = []
    for line in trimmed.split("\n"):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                _collect_wrb_frames(json.loads(line), frames)
            except json.JSONDecodeError:
                continue
    return frames


async def codewiki_rpc(rpc_id: str, payload: list, source_path: str = "/") -> dict:
    last_err = None
    for attempt in range(MAX_CODEWIKI_RETRIES):
        try:
            body_obj = [[[rpc_id, json.dumps(payload), None, "generic"]]]
            body = f"f.req={urllib.parse.quote(json.dumps(body_obj))}&"
            params = {"rpcids": rpc_id, "rt": "c", "source-path": source_path}
            c = get_http_client()
            r = await c.post(
                CODEWIKI_URL, params=params, content=body,
                headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            )
            if r.status_code != 200:
                last_err = f"CodeWiki HTTP {r.status_code}"
                if attempt < MAX_CODEWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                return {"success": False, "error": last_err}
            frames = _extract_wrb_frames(r.text)
            for frame in frames:
                if frame["rpcId"] == rpc_id:
                    return {"success": True, "payload": frame["payload"]}
            available = [f["rpcId"] for f in frames]
            return {"success": False, "error": f"RPC {rpc_id} not found (available: {', '.join(available) if available else 'none'})"}
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            last_err = str(e)
            if attempt < MAX_CODEWIKI_RETRIES - 1:
                await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                continue
    return {"success": False, "error": last_err or "unknown error"}


async def codewiki_search_repos(query: str, limit: int = 5, offset: int = 0) -> dict:
    r = await codewiki_rpc("vyWDAf", [query, limit, query, offset], source_path="/")
    if not r.get("success"):
        return r
    payload = r["payload"]
    rows = payload[0] if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], list) else []
    results = []
    for item in rows:
        if not isinstance(item, list):
            continue
        full_name = item[0] if isinstance(item[0], str) else "unknown/unknown"
        url = (item[3][1] if isinstance(item[3], list) and len(item[3]) > 1
               and isinstance(item[3][1], str) else None)
        desc = (item[5][0] if isinstance(item[5], list)
                and len(item[5]) > 0 else None)
        results.append({"full_name": full_name, "url": url, "description": desc})
    return {"success": True, "results": results, "offset": offset, "has_more": len(results) == limit}


async def codewiki_fetch_repo(owner: str, repo: str) -> dict:
    repo_url = f"https://github.com/{owner}/{repo}"
    source_path = f"/{owner}/{repo}"
    r = await codewiki_rpc("VSX6ub", [repo_url], source_path=source_path)
    if not r.get("success"):
        return r
    payload = r["payload"]
    if not isinstance(payload, list) or not payload or payload[0] is None:
        return {"success": True, "source": "codewiki", "sections": [], "note": f"no wiki found for {owner}/{repo}"}
    primary = payload[0]
    sections_raw = primary[1] if isinstance(primary, list) and len(primary) > 1 and isinstance(primary[1], list) else []
    sections = []
    for item in sections_raw:
        if not isinstance(item, list):
            continue
        title = item[0] if isinstance(item[0], str) else "Untitled"
        markdown = item[5] if isinstance(item[5], str) else (item[4] if isinstance(item[4], str) else "")
        sections.append({"title": title, "markdown": markdown[:3000]})
    canonical_url = payload[1][0][1] if len(payload) > 1 and isinstance(payload[1], list) and payload[1] and isinstance(payload[1][0], list) and len(payload[1][0]) > 1 and isinstance(payload[1][0][1], str) else None
    commit = primary[0][1] if isinstance(primary, list) and len(primary) > 0 and isinstance(primary[0], list) and len(primary[0]) > 1 and isinstance(primary[0][1], str) else None
    return {"success": True, "source": "codewiki", "sections": sections, "canonical_url": canonical_url, "commit": commit}


async def codewiki_ask_repo(owner: str, repo: str, question: str, history: list | None = None) -> dict:
    repo_url = f"https://github.com/{owner}/{repo}"
    source_path = f"/{owner}/{repo}"
    messages = (history or []) + [[question, "user"]]
    cvt = [[m[0], "model" if m[1] == "assistant" else "user"] for m in messages]
    r = await codewiki_rpc("EgIxfe", [cvt, [None, repo_url]], source_path=source_path)
    if not r.get("success"):
        return r
    payload = r["payload"]
    if payload is None:
        return {"success": False, "error": f"CodeWiki has no Q&A index for {owner}/{repo}", "source": "codewiki"}
    answer = payload[0] if isinstance(payload, list) and isinstance(payload[0], str) else str(payload)
    return {"success": True, "answer": answer, "source": "codewiki"}
