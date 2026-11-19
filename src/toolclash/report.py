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

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .bench import BenchRun
from .fileio import read_text, write_text

NONE = "(none)"


class ToolStats(BaseModel):
    tool: str
    questions: int
    samples: int
    correct: int
    accuracy: float
    stability: float
    flip_rate: float
    excluded: int = 0  # samples for this tool with no usable answer (not in the numbers above)


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
    # Samples with no usable answer (reason -> count), left out of every number above.
    excluded: dict[str, int] = Field(default_factory=dict)
    tracks_errors: bool = False  # see BenchRun.tracks_errors

    def save(self, path: str | Path) -> None:
        write_text(path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Summary:
        return cls.model_validate_json(read_text(path, legacy=True))


def _label(name: str | None) -> str:
    return NONE if name is None else name


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def summarize(run: BenchRun) -> Summary:
    labels = [*run.tool_names, NONE]

    matrix: dict[str, dict[str, int]] = {e: {p: 0 for p in labels} for e in labels}
    by_question: dict[str, list[str]] = defaultdict(list)
    expected_of: dict[str, str] = {}
    excluded: Counter[str] = Counter()
    excluded_for: Counter[str] = Counter()
    for pick in run.picks:
        if pick.error:  # truncated / refused: no decision was made, so it is not a pick at all
            excluded[pick.error] += 1
            excluded_for[_label(pick.expected)] += 1
            continue
        exp, got = _label(pick.expected), _label(pick.picked)
        for name in (exp, got):
            if name not in matrix:  # a name outside the tool list, e.g. a hallucinated tool
                labels.append(name)
                for row in matrix.values():
                    row[name] = 0
                matrix[name] = {p: 0 for p in labels}
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
            excluded=excluded_for[name],
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
        excluded=dict(excluded),
        tracks_errors=run.tracks_errors,
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
    *,
    ascii_only: bool = False,
) -> None:
    """Print the report. ``ascii_only`` draws tables, arrows and dots in plain ASCII.

    The CLI asks for that when output goes to a file or pipe: Windows PowerShell 5.1
    decodes redirected output with an old code page that garbles box-drawing characters.
    """
    console = console or Console()
    pct = _pct
    arrow, dot = ("->", ".") if ascii_only else ("→", "·")
    table_box = box.ASCII if ascii_only else box.HEAVY_HEAD

    def show(table: Table) -> None:
        if ascii_only:  # rich would shorten long names with a "…"; wrap them instead
            for column in table.columns:
                column.overflow = "fold"
        console.print(table)

    cost = ""
    if prices:
        cost = f"  est. cost ${estimate_cost(s.usage, *prices):.4f}"
    console.print(
        f"[bold]{escape(s.provider)}[/bold]  split={s.split}  samples/question={s.samples_per_question}  "
        f"calls={s.usage.get('calls', 0)}  tokens={s.usage.get('input_tokens', 0)}in/"
        f"{s.usage.get('output_tokens', 0)}out{cost}"
    )
    if s.excluded:
        detail = ", ".join(f"{reason} {n}" for reason, n in sorted(s.excluded.items()))
        console.print(
            f"[yellow]{sum(s.excluded.values())} samples had no usable answer ({detail}); "
            "they are left out of every number below[/yellow]"
        )
    has_tool_data = any(st.samples for st in s.tools.values())
    overall = pct(s.overall_accuracy) if has_tool_data else "n/a"
    over = "n/a" if s.over_trigger_rate is None else pct(s.over_trigger_rate)
    console.print(f"overall accuracy {overall}   over-trigger rate {over}\n")

    m = Table(title="Confusion matrix (rows: expected, columns: picked)", box=table_box)
    m.add_column("expected \\ picked", style="bold")
    for col in s.labels:
        m.add_column(escape(col), justify="right")
    for row in s.labels:
        cells = []
        for col in s.labels:
            n = s.matrix.get(row, {}).get(col, 0)
            if n == 0:
                cells.append(f"[dim]{dot}[/dim]")
            elif row == col:
                cells.append(f"[green]{n}[/green]")
            else:
                cells.append(f"[red]{n}[/red]")
        m.add_row(escape(row), *cells)
    show(m)

    t = Table(title="Per tool", box=table_box)
    for col in ("tool", "questions", "accuracy", "stability", "flip rate"):
        t.add_column(col, justify="right" if col != "tool" else "left")
    for st in s.tools.values():
        if st.samples:
            t.add_row(
                escape(st.tool), str(st.questions), pct(st.accuracy), pct(st.stability),
                pct(st.flip_rate),
            )
        else:  # no data is not the same as 0% accurate
            t.add_row(escape(st.tool), str(st.questions), "n/a", "n/a", "n/a")
    show(t)

    p = Table(title=f"Most confused pairs (top {max_pairs})", box=table_box)
    for col in ("A", "B", f"A{arrow}B", f"B{arrow}A", "mutual confusion"):
        p.add_column(col, justify="right" if col not in ("A", "B") else "left")
    for ps in s.pairs[:max_pairs]:
        p.add_row(escape(ps.a), escape(ps.b), str(ps.a_to_b), str(ps.b_to_a), pct(ps.mutual_confusion))
    show(p)
