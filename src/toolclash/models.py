"""Core data types.

OpenAI, Anthropic and MCP each describe tools in a slightly different JSON shape.
The loaders turn all of them into the two classes below, so the rest of the
project (question generation, bench runner, reports) only ever deals with one shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .fileio import read_text, write_text

SourceFormat = Literal["mcp", "openai", "anthropic"]


class Tool(BaseModel):
    """One callable tool, in a provider-neutral shape."""

    name: str
    description: str = ""
    # A JSON Schema object describing the tool's arguments, e.g.
    # {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def parameter_names(self) -> list[str]:
        return list(self.parameters.get("properties", {}))

    @property
    def required_parameters(self) -> list[str]:
        return list(self.parameters.get("required", []))

    def schema_only(self) -> Tool:
        """A copy of this tool with its description blanked out.

        The question generator (milestone 2) must build questions from the schema,
        not the description. Otherwise the bench only measures "can the model match
        words from the description", which is not the real-world task.
        """
        return self.model_copy(update={"description": ""})


class ToolSet(BaseModel):
    """An ordered collection of tools, as one agent would see them."""

    tools: list[Tool]
    source_format: SourceFormat | None = None

    @model_validator(mode="after")
    def _names_must_be_unique(self) -> ToolSet:
        seen: set[str] = set()
        dupes: set[str] = set()
        for tool in self.tools:
            if tool.name in seen:
                dupes.add(tool.name)
            seen.add(tool.name)
        if dupes:
            raise ValueError(f"duplicate tool names: {sorted(dupes)}")
        return self

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.tools)

    def save(self, path: str | Path) -> None:
        write_text(path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ToolSet:
        return cls.model_validate_json(read_text(path, legacy=True))
