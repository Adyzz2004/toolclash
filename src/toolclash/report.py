"""Turn raw picks into the numbers people act on.

* **Confusion matrix** — rows are the tool that *should* have been called, columns
  the tool that *was* called. The diagonal is correct; everything else is a
  confusion. ``(none)`` is a real row and column: the no-tool questions live
  there, and so does the model declining to call anything.
* **Per-tool accuracy** — share of samples where the right tool was picked.
* **Stability** — average share of samples per question that agree with the
  majority pick. 1.0 means every repeat gave the same answer.
* **Flip rate** — share of a tool's questions where the repeats did *not* all
  agree. "Correct but shaky" shows up here even when accuracy is high.
* **Mutual confusion** for a pair (A, B) — (A→B + B→A) / (all picks for A and B).
  High values are the pairs whose descriptions need work.
* **Over-trigger rate** — share of no-tool samples where some tool was called.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from .bench import BenchRun

NONE = "(none)"


class ToolStats(BaseModel):
    tool: str
    questions: int
    samples: int
    correct: int
    accuracy: float
    stability: float
    flip_rate: float


class PairStats(BaseModel):
    a: str
    b: str
    a_to_b: int
    b_to_a: int
    mutual_confusion: float


class Summary(BaseModel):
    provider: str
    generator: str
    samples_per_question: int
    split: str
    labels: list[str]  # tool names + NONE, in matrix order
    matrix: dict[str, dict[str, int]]  # expected -> picked -> count
    tools: dict[str, ToolStats]
    pairs: list[PairStats]  # sorted, most confused first
    overall_accuracy: float
    over_trigger_rate: float | None  # None when there were no no-tool questions
    usage: dict[str, int]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Summary:
        return cls.model_validate_json(Path(path).read_text())


def _label(name: str | None) -> str:
    return NONE if name is None else name


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def summarize(run: BenchRun) -> Summary:
    labels = [*run.tool_names, NONE]

    matrix: dict[str, dict[str, int]] = {e: {p: 0 for p in labels} for e in labels}
    by_question: dict[str, list[str]] = defaultdict(list)
    expected_of: dict[str, str] = {}
    for pick in run.picks:
        exp, got = _label(pick.expected), _label(pick.picked)
        if got not in matrix[exp]:  # a name we never heard of (model hallucinated a tool)
            for row in matrix.values():
                row.setdefault(got, 0)
            if got not in labels:
                labels.append(got)
        matrix[exp][got] += 1
        by_question[pick.question_id].append(got)
        expected_of[pick.question_id] = exp

    tools: dict[str, ToolStats] = {}
    for name in run.tool_names:
        qids = [q for q, e in expected_of.items() if e == name]
        n_samples = sum(matrix[name].values())
        correct = matrix[name][name]
        majority_shares, flips = [], 0
        for q in qids:
            answers = by_question[q]
            top = Counter(answers).most_common(1)[0][1]
            majority_shares.append(top / len(answers))
            flips += top < len(answers)
        tools[name] = ToolStats(
            tool=name,
            questions=len(qids),
            samples=n_samples,
            correct=correct,
            accuracy=correct / n_samples if n_samples else 0.0,
            stability=sum(majority_shares) / len(majority_shares) if majority_shares else 0.0,
            flip_rate=flips / len(qids) if qids else 0.0,
        )

    pairs = []
    for a, b in combinations(run.tool_names, 2):
        a_to_b, b_to_a = matrix[a][b], matrix[b][a]
        total = sum(matrix[a].values()) + sum(matrix[b].values())
        pairs.append(
            PairStats(
                a=a, b=b, a_to_b=a_to_b, b_to_a=b_to_a,
                mutual_confusion=(a_to_b + b_to_a) / total if total else 0.0,
            )
        )
    pairs.sort(key=lambda p: (-p.mutual_confusion, p.a, p.b))

    tool_samples = sum(sum(matrix[t].values()) for t in run.tool_names)
    tool_correct = sum(matrix[t][t] for t in run.tool_names)
    none_samples = sum(matrix[NONE].values())
    over_trigger = (none_samples - matrix[NONE][NONE]) / none_samples if none_samples else None

    return Summary(
        provider=run.provider,
        generator=run.generator,
        samples_per_question=run.samples,
        split=run.split,
        labels=labels,
        matrix=matrix,
        tools=tools,
        pairs=pairs,
        overall_accuracy=tool_correct / tool_samples if tool_samples else 0.0,
        over_trigger_rate=over_trigger,
        usage=run.usage,
    )


# --- printing ------------------------------------------------------------------------


def estimate_cost(usage: dict[str, int], price_in: float, price_out: float) -> float:
    """Cost in dollars, given prices per **million** tokens (as providers list them)."""
    return (
        usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out
    ) / 1_000_000


def print_summary(
    s: Summary,
    console: Console | None = None,
    max_pairs: int = 10,
    prices: tuple[float, float] | None = None,
) -> None:
    console = console or Console()
    pct = _pct

    cost = ""
    if prices:
        cost = f"  est. cost ${estimate_cost(s.usage, *prices):.4f}"
    console.print(
        f"[bold]{s.provider}[/bold]  split={s.split}  samples/question={s.samples_per_question}  "
        f"calls={s.usage.get('calls', 0)}  tokens={s.usage.get('input_tokens', 0)}in/"
        f"{s.usage.get('output_tokens', 0)}out{cost}"
    )
    over = "n/a" if s.over_trigger_rate is None else pct(s.over_trigger_rate)
    console.print(f"overall accuracy {pct(s.overall_accuracy)}   over-trigger rate {over}\n")

    m = Table(title="Confusion matrix (rows: expected, columns: picked)")
    m.add_column("expected \\ picked", style="bold")
    for col in s.labels:
        m.add_column(col, justify="right")
    for row in s.labels:
        cells = []
        for col in s.labels:
            n = s.matrix[row].get(col, 0)
            if n == 0:
                cells.append("[dim]·[/dim]")
            elif row == col:
                cells.append(f"[green]{n}[/green]")
            else:
                cells.append(f"[red]{n}[/red]")
        m.add_row(row, *cells)
    console.print(m)

    t = Table(title="Per tool")
    for col in ("tool", "questions", "accuracy", "stability", "flip rate"):
        t.add_column(col, justify="right" if col != "tool" else "left")
    for st in s.tools.values():
        t.add_row(st.tool, str(st.questions), pct(st.accuracy), pct(st.stability), pct(st.flip_rate))
    console.print(t)

    p = Table(title=f"Most confused pairs (top {max_pairs})")
    for col in ("A", "B", "A→B", "B→A", "mutual confusion"):
        p.add_column(col, justify="right" if col not in ("A", "B") else "left")
    for ps in s.pairs[:max_pairs]:
        p.add_row(ps.a, ps.b, str(ps.a_to_b), str(ps.b_to_a), pct(ps.mutual_confusion))
    console.print(p)
