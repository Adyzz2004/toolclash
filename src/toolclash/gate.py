"""The CI gate: did a change to the tool set make an existing tool worse?

Run the bench before the change (baseline) and after it (candidate), on the same
questions. For every tool present in both runs, compare accuracy and stability.
Any drop past the threshold is a regression, and the build should fail.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .report import Summary

# Accuracy and stability are ratios of small counts, so without a tolerance float noise
# decides drops that sit exactly on the threshold: 0.8 - 0.7 > 0.1 is True in floating
# point while 0.9 - 0.8 > 0.1 is False. A drop equal to the threshold always passes.
_EPSILON = 1e-9


class Regression(BaseModel):
    tool: str
    metric: str  # "accuracy", "stability" or "usable answers"
    before: float
    after: float

    @property
    def drop(self) -> float:
        return self.before - self.after

    def describe(self, new_tools: list[str], *, ascii_only: bool = False) -> str:
        cause = f"adding {', '.join(new_tools)} " if new_tools else ""
        arrow = "->" if ascii_only else "→"
        return (
            f"{cause}made {self.tool} {self.drop * 100:.0f} points worse "
            f"on {self.metric} ({self.before * 100:.0f}% {arrow} {self.after * 100:.0f}%)"
        )


class GateResult(BaseModel):
    regressions: list[Regression]
    new_tools: list[str]
    removed_tools: list[str]
    compared: list[str] = Field(default_factory=list)  # tools with data in both runs
    skipped: list[str] = Field(default_factory=list)  # in both runs, but no data in one of them
    # False when a run came from an earlier version, which recorded cut-off and refused
    # replies as "no tool called": usable answers cannot be compared then.
    usable_checked: bool = True

    @property
    def passed(self) -> bool:
        # A gate that compared nothing must not pass: it usually means the wrong
        # baseline file, a renamed tool set or a mistyped --split.
        return bool(self.compared) and not self.regressions


def compare(
    baseline: Summary,
    candidate: Summary,
    *,
    max_accuracy_drop: float = 0.10,
    max_stability_drop: float = 0.10,
) -> GateResult:
    before, after = set(baseline.tools), set(candidate.tools)
    regressions: list[Regression] = []
    compared: list[str] = []
    skipped: list[str] = []
    check_usable = baseline.tracks_errors and candidate.tracks_errors
    for name in sorted(before & after):
        b, c = baseline.tools[name], candidate.tools[name]
        b_asked, c_asked = b.samples + b.excluded, c.samples + c.excluded
        if not b_asked or not c_asked:  # not asked about in one of the runs: nothing to compare
            skipped.append(name)
            continue
        compared.append(name)
        if check_usable:
            # Cut-off and refused replies are left out of accuracy, so check them on their
            # own: a model that stopped giving usable answers for a tool has regressed too.
            b_usable, c_usable = b.samples / b_asked, c.samples / c_asked
            if b_usable - c_usable > max_accuracy_drop + _EPSILON:
                regressions.append(
                    Regression(tool=name, metric="usable answers", before=b_usable, after=c_usable)
                )
            b_acc = b.accuracy if b.samples else None
            c_acc = c.accuracy if c.samples else None
        else:
            # One run is from an earlier version, which counted cut-off and refused replies
            # as "no tool called" (so as wrong). Compare both runs on that basis.
            b_acc, c_acc = b.correct / b_asked, c.correct / c_asked
        if b_acc is not None and c_acc is not None and b_acc - c_acc > max_accuracy_drop + _EPSILON:
            regressions.append(Regression(tool=name, metric="accuracy", before=b_acc, after=c_acc))
        if not b.samples or not c.samples:
            continue
        if b.stability - c.stability > max_stability_drop + _EPSILON:
            regressions.append(
                Regression(tool=name, metric="stability", before=b.stability, after=c.stability)
            )
    return GateResult(
        regressions=regressions,
        new_tools=sorted(after - before),
        removed_tools=sorted(before - after),
        compared=compared,
        skipped=skipped,
        usable_checked=check_usable,
    )
