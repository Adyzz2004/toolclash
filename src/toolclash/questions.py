"""Generate the test questions the bench will ask.

Anti-cheating rules, all enforced here:

* Questions are generated from each tool's **name and argument schema only**. The
  description is blanked out before the generator sees it (``Tool.schema_only``),
  so the bench measures whether the model can route real requests, not whether it
  can match words from the description.
* The generator should be a **different model** from the one being benched
  (that is the caller's job: pass a different provider to ``generate_questions``
  and ``run_bench``).
* **Ambiguous** questions are written for the pairs of tools whose schemas look
  most alike: "route to A, but easy to mistake for B".
* **No-tool** questions are requests none of the tools can handle; picking any
  tool for them is over-triggering.
* Every question is assigned to a ``dev`` or ``heldout`` split by a stable hash of
  its id. Project 3 optimises on ``dev`` and is scored on ``heldout`` only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .fileio import read_text, write_text
from .models import Tool, ToolSet
from .providers import Provider, _tool_words

Kind = Literal["direct", "ambiguous", "no_tool"]
Split = Literal["dev", "heldout"]


class Question(BaseModel):
    id: str
    text: str
    expected: str | None  # tool name, or None when no tool should be called
    kind: Kind
    distractor: str | None = None  # for ambiguous questions: the tool it is easy to confuse with
    split: Split


class QuestionSet(BaseModel):
    generator: str  # provider name that wrote the questions
    tool_names: list[str]
    questions: list[Question]

    def subset(self, split: str | None = None) -> list[Question]:
        if split in (None, "all"):
            return list(self.questions)
        return [q for q in self.questions if q.split == split]

    def save(self, path: str | Path) -> None:
        write_text(path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> QuestionSet:
        return cls.model_validate_json(read_text(path, legacy=True))


class GenerationError(RuntimeError):
    """The generator model did not return a usable JSON list."""


# --- public entry point ---------------------------------------------------------------


def generate_questions(
    tools: ToolSet,
    provider: Provider,
    *,
    per_tool: int = 5,
    pairs: int = 3,
    per_pair: int = 3,
    no_tool: int = 5,
    heldout_fraction: float = 0.3,
) -> QuestionSet:
    questions: list[Question] = []

    for tool in tools.tools if per_tool > 0 else []:
        for i, text in enumerate(_ask(provider, _direct_prompt(tool, per_tool), per_tool)):
            questions.append(_make(f"{tool.name}-direct-{i}", text, tool.name, "direct", None, heldout_fraction))

    for a, b in similar_pairs(tools, pairs) if per_pair > 0 else []:
        for target, other in ((a, b), (b, a)):
            prompt = _ambiguous_prompt(target, other, per_pair)
            for i, text in enumerate(_ask(provider, prompt, per_pair)):
                qid = f"{target.name}-vs-{other.name}-{i}"
                questions.append(_make(qid, text, target.name, "ambiguous", other.name, heldout_fraction))

    if no_tool:
        for i, text in enumerate(_ask(provider, _no_tool_prompt(tools, no_tool), no_tool)):
            questions.append(_make(f"none-{i}", text, None, "no_tool", None, heldout_fraction))

    return QuestionSet(generator=provider.name, tool_names=tools.names(), questions=questions)


def similar_pairs(tools: ToolSet, k: int) -> list[tuple[Tool, Tool]]:
    """The k pairs of tools whose names and parameter names overlap most (Jaccard)."""
    scored = []
    for a, b in itertools.combinations(tools.tools, 2):
        wa, wb = _tool_words(a), _tool_words(b)
        score = len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
        scored.append((score, a, b))
    scored.sort(key=lambda s: (-s[0], s[1].name, s[2].name))
    return [(a, b) for _, a, b in scored[:k]]


# --- prompts ----------------------------------------------------------------------------


def _schema_view(tool: Tool) -> dict[str, Any]:
    """What the generator is allowed to see: name + schema, never the description."""
    stripped = tool.schema_only()
    return {"name": stripped.name, "parameters": stripped.parameters}


def _wrap(instructions: str, request: dict[str, Any]) -> str:
    return (
        f"{instructions}\n\n<request>\n{json.dumps(request, indent=2)}\n</request>\n\n"
        "Reply with a JSON array of strings only. No prose, no numbering, no code fences."
    )


def _direct_prompt(tool: Tool, n: int) -> str:
    return _wrap(
        f"Below is the name and argument schema of a tool an AI assistant can call. "
        f"Its description is hidden on purpose. Write {n} different, realistic requests "
        f"a user might type that should make a careful assistant call this tool. Vary the "
        f"wording and length, mention concrete values for the arguments, and do not use "
        f"the tool's name in the request.",
        {"task": "direct", "n": n, "tool": _schema_view(tool)},
    )


def _ambiguous_prompt(target: Tool, distractor: Tool, n: int) -> str:
    return _wrap(
        f"Below are two tools an AI assistant can call, shown as name and argument schema "
        f"only. Write {n} user requests that a careful assistant should route to the "
        f"TARGET tool, but that are easy to mistake for the DISTRACTOR tool. Do not use "
        f"either tool's name in the request.",
        {
            "task": "ambiguous",
            "n": n,
            "target": _schema_view(target),
            "distractor": _schema_view(distractor),
        },
    )


def _no_tool_prompt(tools: ToolSet, n: int) -> str:
    return _wrap(
        f"Below are the tools an AI assistant can call, shown as name and argument schema "
        f"only. Write {n} user requests on nearby topics that NONE of these tools can "
        f"handle, so the assistant should answer directly without calling any tool.",
        {"task": "no_tool", "n": n, "tools": [_schema_view(t) for t in tools.tools]},
    )


# --- helpers ------------------------------------------------------------------------------


def _ask(provider: Provider, prompt: str, n: int) -> list[str]:
    text = provider.complete(prompt)
    items = parse_string_list(text)
    if not items:
        raise GenerationError(f"generator returned no questions. Raw reply:\n{text[:500]}")
    return items[:n]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_string_list(text: str) -> list[str]:
    """Pull a JSON array of strings out of a model reply, tolerating fences and chatter."""
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in data if isinstance(s, str) and s.strip()]


def _make(
    qid: str, text: str, expected: str | None, kind: Kind, distractor: str | None, heldout: float
) -> Question:
    return Question(
        id=qid, text=text, expected=expected, kind=kind, distractor=distractor,
        split=assign_split(qid, heldout),
    )


def assign_split(qid: str, heldout_fraction: float) -> Split:
    """Stable: the same id always lands in the same split, on any machine, any run."""
    digest = hashlib.sha1(qid.encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "heldout" if bucket < heldout_fraction else "dev"
