from __future__ import annotations

import asyncio
import json

import httpx

from config import DEEPWIKI_MCP, get_http_client, api_error

MAX_DEEPWIKI_RETRIES = 3
BASE_DELAY = 1.0


def _parse_mcp_sse(text: str) -> dict | None:
    # L10: SSE frames can split one JSON payload across multiple `data:` lines.
    # Collect the current frame (data lines) and try each, not just the first.
    buffered: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            buffered.append(line[6:])
            try:
                return json.loads("\n".join(buffered))
            except json.JSONDecodeError:
                continue
        elif line == "":
            buffered = []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def deepwiki_fetch(owner: str, repo: str, wiki_name: str = "") -> dict:
    repo_label = f"{owner}/{repo}"
    last_err = None
    c = get_http_client()
    for attempt in range(MAX_DEEPWIKI_RETRIES):
        try:
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            init = await c.post(DEEPWIKI_MCP, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "codesearch", "version": "1.0"}}
            }, headers=headers)
            if init.status_code != 200:
                last_err = api_error("init failed", init)
                if attempt < MAX_DEEPWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                break
            struct_args: dict[str, str] = {"repoName": repo_label}
            if wiki_name:
                struct_args["wikiName"] = wiki_name
            struct = await c.post(DEEPWIKI_MCP, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "read_wiki_structure", "arguments": struct_args}
            }, headers=headers)
            if struct.status_code != 200:
                last_err = api_error("structure", struct)
                if attempt < MAX_DEEPWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                break
            sdata = _parse_mcp_sse(struct.text)
            if not sdata or "result" not in sdata:
                last_err = "no result from structure"
                if attempt < MAX_DEEPWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                break
            sections = []
            for item in sdata["result"].get("content", []):
                if item.get("type") == "text":
                    sections.append(item["text"][:3000])
            detail_args: dict[str, str] = {"repoName": repo_label}
            if wiki_name:
                detail_args["wikiName"] = wiki_name
            detail_data = await c.post(DEEPWIKI_MCP, json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "read_wiki_contents", "arguments": detail_args}
            }, headers=headers)
            detail = ""
            if detail_data.status_code == 200:
                ddata = _parse_mcp_sse(detail_data.text)
                if ddata and "result" in ddata:
                    texts = []
                    for item in ddata["result"].get("content", []):
                        if item.get("type") == "text":
                            texts.append(item["text"])
                    detail = "\n\n".join(texts)[:10000]
            return {"success": True, "source": "deepwiki_mcp", "repo": repo_label, "structure": sections[:5], "detail": detail}
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            last_err = str(e)
            if attempt < MAX_DEEPWIKI_RETRIES - 1:
                await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                continue
    try:
        r = await c.get(f"https://deepwiki.com/{owner}/{repo}", headers={"User-Agent": "mcp-codesearch/1.0"})
        if r.status_code != 200:
            return {"success": False, "error": api_error("deepwiki", r)}
        return {"success": True, "source": "deepwiki_html", "content": r.text[:10000]}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}


async def deepwiki_ask(owner: str = "", repo: str = "", question: str = "",
                       wiki_name: str = "", repos: list[str] | None = None) -> dict:
    repo_label = repos if repos else f"{owner}/{repo}"
    last_err = None
    c = get_http_client()
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    for attempt in range(MAX_DEEPWIKI_RETRIES):
        try:
            init = await c.post(DEEPWIKI_MCP, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "codesearch", "version": "1.0"}}
            }, headers=headers)
            if init.status_code != 200:
                last_err = api_error("init failed", init)
                if attempt < MAX_DEEPWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                return {"success": False, "error": last_err}
            ask_args: dict = {"repoName": repo_label, "question": question}
            if wiki_name:
                ask_args["wikiName"] = wiki_name
            r = await c.post(DEEPWIKI_MCP, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "ask_question", "arguments": ask_args}
            }, headers=headers)
            if r.status_code != 200:
                last_err = api_error("DeepWiki", r)
                if attempt < MAX_DEEPWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                return {"success": False, "error": last_err}
            data = _parse_mcp_sse(r.text)
            if not data or "result" not in data:
                last_err = "no result from DeepWiki"
                if attempt < MAX_DEEPWIKI_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                return {"success": False, "error": last_err}
            answer = ""
            for item in data["result"].get("content", []):
                if item.get("type") == "text":
                    answer += item["text"] + "\n"
            return {"success": True, "answer": answer.strip(), "source": "deepwiki"}
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            last_err = str(e)
            if attempt < MAX_DEEPWIKI_RETRIES - 1:
                await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                continue
    return {"success": False, "error": last_err}
