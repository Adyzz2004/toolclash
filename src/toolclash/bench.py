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

from .fileio import read_text, write_text
from .models import ToolSet
from .providers import Provider, ProviderError
from .questions import Kind, QuestionSet

SPLITS = ("dev", "heldout", "all")


class Pick(BaseModel):
    question_id: str
    kind: Kind
    expected: str | None
    picked: str | None
    sample: int
    # Set when the reply had no usable answer ("truncated", "refused"); see providers.Choice.
    # Such samples are left out of every score instead of counting as "called no tool".
    error: str | None = None


class BenchRun(BaseModel):
    """Everything the report needs. Saved to JSON so reports can be re-run offline."""

    provider: str
    generator: str
    samples: int
    split: str
    tool_names: list[str]
    picks: list[Pick]
    usage: dict[str, int]
    # True when cut-off and refused replies are marked with Pick.error. Runs saved by
    # earlier versions lack the field: they recorded those replies as "no tool called".
    tracks_errors: bool = False

    def save(self, path: str | Path) -> None:
        write_text(path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BenchRun:
        return cls.model_validate_json(read_text(path, legacy=True))


class BenchAborted(RuntimeError):
    """The run stopped part way because the provider raised an error.

    ``partial`` holds every pick made before the failure, so the calls already paid
    for are not lost; ``cause`` is the provider's exception (a ``ProviderError`` for
    API problems). Ctrl-C is not turned into this: the ``KeyboardInterrupt`` is
    re-raised as is, with the partial run attached as its ``partial`` attribute.
    """

    def __init__(self, partial: BenchRun, cause: Exception):
        reason = str(cause) if isinstance(cause, ProviderError) else f"{type(cause).__name__}: {cause}"
        super().__init__(reason)
        self.partial = partial
        self.cause = cause


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
    names = set(tools.names())
    mentioned = set(questions.tool_names)
    for q in questions.questions:
        mentioned.update(t for t in (q.expected, q.distractor) if t is not None)
    if mentioned - names:
        raise ValueError(f"questions mention tools not in the tool set: {sorted(mentioned - names)}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of: {', '.join(SPLITS)}")
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")

    selected = questions.subset(split)
    if not selected:
        raise ValueError(f"the questions file has no questions in the {split!r} split")

    total = len(selected) * samples
    picks: list[Pick] = []

    def result() -> BenchRun:
        return BenchRun(
            provider=provider.name,
            generator=questions.generator,
            samples=samples,
            split=split,
            tool_names=tools.names(),
            picks=picks,
            usage=provider.usage.as_dict(),
            tracks_errors=True,
        )

    try:
        for q in selected:
            for i in range(samples):
                choice = provider.choose_tool(q.text, tools)
                picks.append(
                    Pick(
                        question_id=q.id, kind=q.kind, expected=q.expected, picked=choice.tool,
                        sample=i, error=getattr(choice, "error", None),
                    )
                )
                if progress:
                    progress(len(picks), total)
    except KeyboardInterrupt as e:
        e.partial = result()  # type: ignore[attr-defined]
        raise
    except Exception as e:
        raise BenchAborted(result(), e) from e

    return result()
