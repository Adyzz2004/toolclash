"""Read tool definitions in any of the three common formats into a ToolSet.

Supported inputs:

* MCP — the JSON a server returns for ``tools/list``. Either bare
  ``{"tools": [...]}``, wrapped in JSON-RPC ``{"result": {"tools": [...]}}``,
  or a bare list of tool objects. Items use ``inputSchema``.
* OpenAI — a list of function tools. Either the Chat Completions shape
  ``{"type": "function", "function": {"name": ..., "parameters": ...}}`` or the
  flat Responses-API shape ``{"type": "function", "name": ..., "parameters": ...}``.
* Anthropic — a list of ``{"name": ..., "description": ..., "input_schema": {...}}``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import SourceFormat, Tool, ToolSet


class LoadError(ValueError):
    """Raised when the input is not a tool list we understand."""


def load_tools(path: str | Path, fmt: SourceFormat | None = None) -> ToolSet:
    """Read a JSON file and parse it. ``fmt`` overrides auto-detection."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise LoadError(f"{path} is not valid JSON: {e}") from e
    return parse_tools(data, fmt)


def parse_tools(data: Any, fmt: SourceFormat | None = None) -> ToolSet:
    """Parse already-loaded JSON data. ``fmt`` overrides auto-detection."""
    fmt = fmt or detect_format(data)
    if fmt not in _PARSERS:
        raise LoadError(f"unknown format {fmt!r}; expected one of {sorted(_PARSERS)}")
    return _PARSERS[fmt](data)


def detect_format(data: Any) -> SourceFormat:
    """Guess which provider's shape ``data`` is in, from its top-level keys."""
    if isinstance(data, dict):
        if "tools" in data:
            return "mcp"
        result = data.get("result")
        if isinstance(result, dict) and "tools" in result:
            return "mcp"
        raise LoadError("JSON object has no 'tools' key; expected MCP tools/list output")

    if isinstance(data, list):
        if not data:
            raise LoadError("tool list is empty")
        first = data[0]
        if not isinstance(first, dict):
            raise LoadError("tool list items must be JSON objects")
        if "function" in first or ("type" in first and "parameters" in first):
            return "openai"
        if "input_schema" in first:
            return "anthropic"
        if "inputSchema" in first:
            return "mcp"
        raise LoadError(f"could not recognise tool format from keys {sorted(first)}")

    raise LoadError(f"expected a JSON object or list, got {type(data).__name__}")


# --- per-format parsers ---------------------------------------------------------


def _from_mcp(data: Any) -> ToolSet:
    if isinstance(data, dict):
        raw = data.get("tools")
        if raw is None:
            raw = data.get("result", {}).get("tools")
    else:
        raw = data
    _check_list(raw, "mcp")
    return ToolSet(tools=[_tool(item, "inputSchema") for item in raw], source_format="mcp")


def _from_openai(data: Any) -> ToolSet:
    _check_list(data, "openai")
    # Nested shape keeps the real definition under "function"; flat shape is the item itself.
    tools = [_tool(item.get("function", item), "parameters") for item in data]
    return ToolSet(tools=tools, source_format="openai")


def _from_anthropic(data: Any) -> ToolSet:
    _check_list(data, "anthropic")
    return ToolSet(tools=[_tool(item, "input_schema") for item in data], source_format="anthropic")


_PARSERS: dict[str, Callable[[Any], ToolSet]] = {
    "mcp": _from_mcp,
    "openai": _from_openai,
    "anthropic": _from_anthropic,
}


# --- helpers --------------------------------------------------------------------


def _check_list(raw: Any, fmt: str) -> None:
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise LoadError(f"expected a list of tool objects for format {fmt!r}")


def _tool(item: dict[str, Any], schema_key: str) -> Tool:
    """Build a Tool from one raw entry. ``schema_key`` is where that format keeps the schema."""
    name = item.get("name")
    if not name:
        raise LoadError(f"tool entry has no name: {item}")
    return Tool(
        name=name,
        description=item.get("description") or "",
        parameters=item.get(schema_key) or {},
    )
