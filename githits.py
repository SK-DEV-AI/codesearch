from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

from config import GITHITS_API_TOKEN

_GITHITS_BIN = shutil.which("githits") or os.path.expanduser("~/.npm-global/bin/githits")


def _build_env() -> dict[str, str]:
    env = {**os.environ, "GITHITS_API_TOKEN": GITHITS_API_TOKEN}
    env.pop("COLUMNS", None)
    return env


async def _run_githits(args: list[str], timeout: int = 60) -> dict[str, Any]:
    """Run a githits CLI command and return parsed JSON result."""
    if not GITHITS_API_TOKEN:
        return {"error": "GITHITS_API_TOKEN not configured"}
    cmd = [_GITHITS_BIN, *args, "--json"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=_build_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {"error": f"githits command timed out after {timeout}s: {' '.join(args)}"}
    if proc.returncode != 0:
        return {"error": f"githits failed (code {proc.returncode}): {stderr.decode().strip()}"}
    try:
        return json.loads(stdout.decode())
    except json.JSONDecodeError:
        return {"error": f"githits returned invalid JSON: {stdout.decode()[:500]}"}


async def get_example(query: str, language: str = "") -> dict[str, Any]:
    """Find canonical open-source code examples matching a natural-language description."""
    args = ["example", query]
    if language:
        args.extend(["--lang", language])
    result = await _run_githits(args, timeout=90)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    if isinstance(result, dict):
        return {"success": True, **result}
    return {"success": True, "result": result}


async def search(query: str, target: str = "", source: str = "",
                 lang: str = "", limit: int = 10) -> dict[str, Any]:
    """Search code, docs, and symbols across indexed dependencies."""
    args = ["search", query]
    if target:
        args.extend(["--in", target])
    if source:
        args.extend(["--source", source])
    if lang:
        args.extend(["--lang", lang])
    args.extend(["--limit", str(limit)])
    result = await _run_githits(args, timeout=90)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "results": result.get("results", result)}


async def code_files(spec: str, path_prefix: str = "") -> dict[str, Any]:
    """List files in an indexed dependency without needing a GitHub URL."""
    args = ["code", "files", spec]
    if path_prefix:
        args.append(path_prefix)
    result = await _run_githits(args, timeout=30)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "files": result.get("files", result)}


async def code_read(spec: str, path: str) -> dict[str, Any]:
    """Read a file from an indexed dependency by package-scoped path."""
    args = ["code", "read", spec, path]
    result = await _run_githits(args, timeout=30)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "content": result.get("content", result)}


async def code_grep(spec: str, pattern: str, path_prefix: str = "") -> dict[str, Any]:
    """Grep through indexed dependency source for a pattern."""
    args = ["code", "grep", spec, pattern]
    if path_prefix:
        args.append(path_prefix)
    result = await _run_githits(args, timeout=60)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "matches": result.get("matches", result)}


async def pkg_deps(spec: str) -> dict[str, Any]:
    """Analyze transitive dependencies for a package with conflict detection."""
    args = ["pkg", "deps", spec]
    result = await _run_githits(args, timeout=30)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "dependencies": result.get("dependencies", result)}
