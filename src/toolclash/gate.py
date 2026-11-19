"""The CI gate: did a change to the tool set make an existing tool worse?

Run the bench before the change (baseline) and after it (candidate), on the same
questions. For every tool present in both runs, compare accuracy and stability.
Any drop past the threshold is a regression, and the build should fail.
"""

from __future__ import annotations

from pydantic import BaseModel

from .report import Summary


class Regression(BaseModel):
    tool: str
    metric: str  # "accuracy" or "stability"
    before: float
    after: float

    @property
    def drop(self) -> float:
        return self.before - self.after

    def describe(self, new_tools: list[str]) -> str:
        cause = f"adding {', '.join(new_tools)} " if new_tools else ""
        return (
            f"{cause}made {self.tool} {self.drop * 100:.0f} points worse "
            f"on {self.metric} ({self.before * 100:.0f}% → {self.after * 100:.0f}%)"
        )


class GateResult(BaseModel):
    regressions: list[Regression]
    new_tools: list[str]
    removed_tools: list[str]

    @property
    def passed(self) -> bool:
        return not self.regressions


def compare(
    baseline: Summary,
    candidate: Summary,
    *,
    max_accuracy_drop: float = 0.10,
    max_stability_drop: float = 0.10,
) -> GateResult:
    before, after = set(baseline.tools), set(candidate.tools)
    regressions: list[Regression] = []
    for name in sorted(before & after):
        b, c = baseline.tools[name], candidate.tools[name]
        if b.accuracy - c.accuracy > max_accuracy_drop:
            regressions.append(Regression(tool=name, metric="accuracy", before=b.accuracy, after=c.accuracy))
        if b.stability - c.stability > max_stability_drop:
            regressions.append(
                Regression(tool=name, metric="stability", before=b.stability, after=c.stability)
            )
    return GateResult(
        regressions=regressions,
        new_tools=sorted(after - before),
        removed_tools=sorted(before - after),
    )
