"""The bench runner: ask the model which tool it would call, many times, and record it.

Each question is asked ``samples`` times. One answer tells you whether the model
was right; several answers tell you whether it was *sure*. A tool that is picked
correctly 4 times out of 5 is a problem waiting to happen, and only repeated
sampling reveals it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from .models import ToolSet
from .providers import Provider
from .questions import Kind, QuestionSet


class Pick(BaseModel):
    question_id: str
    kind: Kind
    expected: str | None
    picked: str | None
    sample: int


class BenchRun(BaseModel):
    """Everything the report needs. Saved to JSON so reports can be re-run offline."""

    provider: str
    generator: str
    samples: int
    split: str
    tool_names: list[str]
    picks: list[Pick]
    usage: dict[str, int]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BenchRun:
        return cls.model_validate_json(Path(path).read_text())


ProgressFn = Callable[[int, int], None]  # (done, total)


def run_bench(
    tools: ToolSet,
    questions: QuestionSet,
    provider: Provider,
    *,
    samples: int = 5,
    split: str = "dev",
    progress: ProgressFn | None = None,
) -> BenchRun:
    if set(questions.tool_names) - set(tools.names()):
        missing = sorted(set(questions.tool_names) - set(tools.names()))
        raise ValueError(f"questions mention tools not in the tool set: {missing}")

    selected = questions.subset(split)
    total = len(selected) * samples
    picks: list[Pick] = []
    done = 0
    for q in selected:
        for i in range(samples):
            choice = provider.choose_tool(q.text, tools)
            picks.append(
                Pick(question_id=q.id, kind=q.kind, expected=q.expected, picked=choice.tool, sample=i)
            )
            done += 1
            if progress:
                progress(done, total)

    return BenchRun(
        provider=provider.name,
        generator=questions.generator,
        samples=samples,
        split=split,
        tool_names=tools.names(),
        picks=picks,
        usage=provider.usage.as_dict(),
    )
