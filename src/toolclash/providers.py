"""Model providers: the only place in the project that talks to an LLM API.

Every provider does two jobs:

1. ``choose_tool(question, tools)`` — show the model the tool list and one user
   request, and return the name of the tool it decided to call (or ``None`` if
   it called nothing). This is what the bench measures.
2. ``complete(prompt)`` — plain text in, plain text out. Used by the question
   generator, which must run on a *different* model than the one being benched.

``FakeProvider`` needs no network or API key. It is used by the tests and by
``--provider fake`` so the whole pipeline can be run end to end for free.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import Tool, ToolSet

SYSTEM_PROMPT = (
    "You are an assistant that can call tools. Read the user's request and call the "
    "single most appropriate tool with sensible arguments. If none of the tools fits "
    "the request, do not call any tool; reply with one short sentence instead."
)

# Providers reject an empty schema; this is the smallest valid "no arguments" schema.
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class Usage:
    """Token counts for one provider, summed over every call. Cost = tokens x price."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.calls += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class Choice:
    """What the model picked for one question. ``tool`` is None when it called nothing."""

    tool: str | None
    raw: Any = None  # the provider's own response object, kept for debugging


class Provider(Protocol):
    name: str
    usage: Usage

    def choose_tool(self, question: str, tools: ToolSet) -> Choice: ...

    def complete(self, prompt: str) -> str: ...


# --- ToolSet -> provider wire format (the reverse of loaders.py) --------------------


def to_openai_tools(tools: ToolSet) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or _EMPTY_SCHEMA,
            },
        }
        for t in tools.tools
    ]


def to_anthropic_tools(tools: ToolSet) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters or _EMPTY_SCHEMA,
        }
        for t in tools.tools
    ]


# --- real providers -------------------------------------------------------------------


class OpenAIProvider:
    """Chat Completions API with function tools. Reads OPENAI_API_KEY from the environment.

    Also serves Gemini and Ollama: both expose an OpenAI-compatible endpoint, so they
    are this same class pointed at a different ``base_url`` (see ``get_provider``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        label: str = "openai",
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'toolclash[openai]' to use the openai provider") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.name = f"{label}:{model}"
        self.usage = Usage()

    def _kwargs(self) -> dict[str, Any]:
        # Some newer OpenAI models reject a temperature argument, so only send it when set.
        return {"temperature": self.temperature} if self.temperature is not None else {}

    def choose_tool(self, question: str, tools: ToolSet) -> Choice:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=to_openai_tools(tools),
            tool_choice="auto",
            **self._kwargs(),
        )
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        calls = resp.choices[0].message.tool_calls or []
        return Choice(tool=calls[0].function.name if calls else None, raw=resp)

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self._kwargs(),
        )
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""


class AnthropicProvider:
    """Messages API with tools. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str, temperature: float | None = None, max_tokens: int = 1024):
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install 'toolclash[anthropic]' to use the anthropic provider"
            ) from e
        self.client = Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        self.usage = Usage()

    def _kwargs(self) -> dict[str, Any]:
        return {"temperature": self.temperature} if self.temperature is not None else {}

    def choose_tool(self, question: str, tools: ToolSet) -> Choice:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
            tools=to_anthropic_tools(tools),
            **self._kwargs(),
        )
        self.usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        return Choice(tool=tool_blocks[0].name if tool_blocks else None, raw=resp)

    def complete(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max(self.max_tokens, 2048),
            messages=[{"role": "user", "content": prompt}],
            **self._kwargs(),
        )
        self.usage.add(resp.usage.input_tokens, resp.usage.output_tokens)
        return "".join(b.text for b in resp.content if b.type == "text")


# --- fake provider ----------------------------------------------------------------------


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    """Lower-case words, underscores split, plurals crudely stripped ("docs" -> "doc")."""
    words = _WORD_RE.findall(text.lower().replace("_", " "))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def _tool_words(tool: Tool) -> set[str]:
    return _words(tool.name) | {w for p in tool.parameter_names for w in _words(p)}


@dataclass
class FakeProvider:
    """A stand-in model that needs no API key.

    ``choose_tool`` picks the tool whose name and parameter words overlap most with
    the question. Ties are broken at random, which is the fake's version of
    "confusion". ``flip_rate`` adds extra random wrong picks so reports look
    realistic in demos.

    ``complete`` only understands the prompts that ``questions.py`` builds: it reads
    the JSON inside ``<request>...</request>`` and fills in templates.
    """

    flip_rate: float = 0.0
    seed: int = 0
    name: str = "fake"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_tool(self, question: str, tools: ToolSet) -> Choice:
        self.usage.add(len(question) // 4, 8)  # rough token estimate, so cost math has data
        q = _words(question)
        scores = {t.name: len(q & _tool_words(t)) for t in tools.tools}
        best = max(scores.values(), default=0)
        if best == 0:
            return Choice(tool=None)
        pick = self.rng.choice(sorted(n for n, s in scores.items() if s == best))
        if self.rng.random() < self.flip_rate:
            pick = self.rng.choice(tools.names())
        return Choice(tool=pick)

    def complete(self, prompt: str) -> str:
        self.usage.add(len(prompt) // 4, 64)
        match = re.search(r"<request>\s*(\{.*\})\s*</request>", prompt, re.DOTALL)
        if not match:
            return "[]"
        req = json.loads(match.group(1))
        task, n = req["task"], int(req["n"])
        if task == "direct":
            lines = _fake_direct(req["tool"], n)
        elif task == "ambiguous":
            lines = _fake_ambiguous(req["target"], req["distractor"], n)
        else:
            lines = _FAKE_NO_TOOL[:n]
        return json.dumps(lines)


def _verb(tool: dict[str, Any]) -> str:
    return tool["name"].replace("_", " ")


def _args(tool: dict[str, Any], i: int) -> str:
    props = list((tool.get("parameters") or {}).get("properties", {}))
    required = (tool.get("parameters") or {}).get("required") or props[:1]
    return ", ".join(f"{p} is value{i}" for p in required) or "no arguments"


_DIRECT_TEMPLATES = [
    "Please {verb}: {args}.",
    "I need to {verb}; {args}.",
    "Could you {verb}? {args}.",
    "Go ahead and {verb}, where {args}.",
    "Help me {verb} — {args}.",
    "{verb} for me, {args}.",
]


def _fake_direct(tool: dict[str, Any], n: int) -> list[str]:
    return [
        _DIRECT_TEMPLATES[i % len(_DIRECT_TEMPLATES)].format(verb=_verb(tool), args=_args(tool, i))
        for i in range(n)
    ]


def _fake_ambiguous(target: dict[str, Any], distractor: dict[str, Any], n: int) -> list[str]:
    return [
        f"I want to {_verb(target)} (something like {_verb(distractor)}), {_args(target, i)}."
        for i in range(n)
    ]


_FAKE_NO_TOOL = [
    "What is the capital of France?",
    "How are you doing today?",
    "Explain why the sky is blue.",
    "Give me a one-line summary of the French Revolution.",
    "What is 17 times 23?",
    "Tell me a fun fact about octopuses.",
    "Translate 'good morning' into Spanish.",
    "Who wrote Pride and Prejudice?",
]


# --- factory -----------------------------------------------------------------------------


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

PROVIDER_KINDS = ("fake", "openai", "anthropic", "gemini", "ollama")


def get_provider(spec: str) -> Provider:
    """Build a provider from a spec string.

    * ``fake`` or ``fake:0.2`` (random flip rate) — no network, no key.
    * ``openai:<model>`` — needs OPENAI_API_KEY.
    * ``anthropic:<model>`` — needs ANTHROPIC_API_KEY.
    * ``gemini:<model>`` — needs GEMINI_API_KEY; uses Google's OpenAI-compatible endpoint.
    * ``ollama:<model>`` — local Ollama at localhost:11434 (override with OLLAMA_HOST).
    """
    kind, _, rest = spec.partition(":")
    if kind == "fake":
        return FakeProvider(flip_rate=float(rest) if rest else 0.0)
    if kind not in PROVIDER_KINDS:
        raise ValueError(f"unknown provider {kind!r}; expected one of {', '.join(PROVIDER_KINDS)}")
    if not rest:
        raise ValueError(f"provider spec {spec!r} needs a model, e.g. '{kind}:<model-name>'")
    if kind == "openai":
        return OpenAIProvider(rest)
    if kind == "anthropic":
        return AnthropicProvider(rest)
    if kind == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        return OpenAIProvider(rest, base_url=GEMINI_BASE_URL, api_key=key, label="gemini")
    host = os.environ.get("OLLAMA_HOST", OLLAMA_BASE_URL).rstrip("/")
    if not host.endswith("/v1"):
        host += "/v1"
    return OpenAIProvider(rest, base_url=host, api_key="ollama", label="ollama")
