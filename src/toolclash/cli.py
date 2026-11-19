"""Command-line entry point. Run ``toolclash --help``.

Pipeline:  inspect → generate → run → report → compare
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from .bench import BenchRun, run_bench
from .gate import compare as compare_summaries
from .loaders import load_tools
from .providers import get_provider
from .questions import QuestionSet, generate_questions
from .report import print_summary, summarize

app = typer.Typer(
    help="Measure which of your agent's tools the model confuses with each other.",
    no_args_is_help=True,
)
console = Console()
err = Console(stderr=True)

ToolsArg = Annotated[Path, typer.Argument(help="Tool definitions: MCP, OpenAI or Anthropic JSON.")]
ProviderOpt = Annotated[
    str,
    typer.Option(
        "--provider", "-p",
        help="fake, fake:<flip-rate>, openai:<model>, anthropic:<model>, gemini:<model>, ollama:<model>",
    ),
]


def _fail(message: str) -> NoReturn:
    err.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """toolclash: tool-selection confusion benchmark."""


@app.command()
def inspect(
    path: ToolsArg,
    fmt: Annotated[
        str | None, typer.Option("--format", "-f", help="Force a format: mcp, openai or anthropic.")
    ] = None,
) -> None:
    """Load a tool file and print what toolclash sees in it."""
    try:
        toolset = load_tools(path, fmt)  # type: ignore[arg-type]
    except (ValueError, FileNotFoundError) as e:
        _fail(str(e))

    table = Table(title=f"{len(toolset)} tools  (format: {toolset.source_format})")
    table.add_column("name", style="bold")
    table.add_column("params")
    table.add_column("required")
    table.add_column("description")
    for tool in toolset.tools:
        table.add_row(
            tool.name,
            ", ".join(tool.parameter_names) or "-",
            ", ".join(tool.required_parameters) or "-",
            _one_line(tool.description, 60),
        )
    console.print(table)


@app.command()
def generate(
    tools: ToolsArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write questions.json")],
    provider: ProviderOpt = "fake",
    per_tool: Annotated[int, typer.Option(help="Direct questions per tool.")] = 5,
    pairs: Annotated[int, typer.Option(help="How many similar tool pairs get ambiguous questions.")] = 3,
    per_pair: Annotated[int, typer.Option(help="Ambiguous questions per direction per pair.")] = 3,
    no_tool: Annotated[int, typer.Option(help="Questions that no tool should answer.")] = 5,
    heldout: Annotated[float, typer.Option(help="Fraction of questions kept for the held-out set.")] = 0.3,
) -> None:
    """Generate test questions from the tool schemas (descriptions are hidden from the generator)."""
    try:
        toolset = load_tools(tools)
        gen = get_provider(provider)
        qs = generate_questions(
            toolset, gen, per_tool=per_tool, pairs=pairs, per_pair=per_pair,
            no_tool=no_tool, heldout_fraction=heldout,
        )
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as e:
        _fail(str(e))
    qs.save(out)
    kinds = {k: sum(q.kind == k for q in qs.questions) for k in ("direct", "ambiguous", "no_tool")}
    held = sum(q.split == "heldout" for q in qs.questions)
    console.print(
        f"wrote {len(qs.questions)} questions to {out}  "
        f"(direct {kinds['direct']}, ambiguous {kinds['ambiguous']}, no-tool {kinds['no_tool']}; "
        f"{held} held out)  generator={gen.name}  calls={gen.usage.calls}"
    )


@app.command()
def run(
    tools: ToolsArg,
    questions: Annotated[Path, typer.Argument(help="questions.json from `generate`.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write run.json")],
    provider: ProviderOpt = "fake",
    samples: Annotated[int, typer.Option("--samples", "-n", help="Times to ask each question.")] = 5,
    split: Annotated[str, typer.Option(help="dev, heldout or all.")] = "dev",
) -> None:
    """Ask the model which tool it would call for every question, several times each."""
    try:
        toolset = load_tools(tools)
        qs = QuestionSet.load(questions)
        model = get_provider(provider)
        if model.name == qs.generator and not model.name.startswith("fake"):
            err.print(
                "[yellow]warning:[/yellow] benching with the same model that wrote the questions; "
                "results will be flattering."
            )
        with Progress(console=console, transient=True) as bar:
            task = bar.add_task("asking", total=None)
            result = run_bench(
                toolset, qs, model, samples=samples, split=split,
                progress=lambda done, total: bar.update(task, completed=done, total=total),
            )
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as e:
        _fail(str(e))
    result.save(out)
    console.print(f"wrote {len(result.picks)} picks to {out}  provider={model.name}  usage={result.usage}")


@app.command()
def report(
    run_file: Annotated[Path, typer.Argument(help="run.json from `run`.")],
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write the summary as JSON.")] = None,
    top: Annotated[int, typer.Option(help="How many confused pairs to show.")] = 10,
    price_in: Annotated[
        float | None, typer.Option(help="Provider's input price per million tokens, for a cost estimate.")
    ] = None,
    price_out: Annotated[
        float | None, typer.Option(help="Provider's output price per million tokens.")
    ] = None,
) -> None:
    """Print the confusion matrix, per-tool stats and most confused pairs."""
    try:
        summary = summarize(BenchRun.load(run_file))
    except (ValueError, FileNotFoundError) as e:
        _fail(str(e))
    prices = (price_in, price_out) if price_in is not None and price_out is not None else None
    print_summary(summary, console, max_pairs=top, prices=prices)
    if json_out:
        summary.save(json_out)
        console.print(f"wrote summary to {json_out}")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="run.json before the change.")],
    candidate: Annotated[Path, typer.Argument(help="run.json after the change.")],
    max_accuracy_drop: Annotated[float, typer.Option(help="Fail if accuracy drops more than this.")] = 0.10,
    max_stability_drop: Annotated[float, typer.Option(help="Fail if stability drops more than this.")] = 0.10,
) -> None:
    """CI gate: exit 1 if any existing tool got worse between two runs."""
    try:
        before = summarize(BenchRun.load(baseline))
        after = summarize(BenchRun.load(candidate))
    except (ValueError, FileNotFoundError) as e:
        _fail(str(e))
    result = compare_summaries(
        before, after, max_accuracy_drop=max_accuracy_drop, max_stability_drop=max_stability_drop
    )
    if result.new_tools:
        console.print(f"new tools: {', '.join(result.new_tools)}")
    if result.removed_tools:
        console.print(f"removed tools: {', '.join(result.removed_tools)}")
    if result.passed:
        console.print("[green]PASS[/green] no existing tool regressed")
        return
    for r in result.regressions:
        console.print(f"[red]FAIL[/red] {r.describe(result.new_tools)}")
    raise typer.Exit(code=1)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
