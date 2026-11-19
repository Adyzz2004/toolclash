"""Command-line entry point. Run ``toolclash --help``.

Pipeline:  inspect → generate → run → report → compare
"""

from __future__ import annotations

import codecs
import os
import stat
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress
from rich.table import Table

from .bench import SPLITS, BenchAborted, BenchRun, run_bench
from .fileio import check_file_name
from .gate import compare as compare_summaries
from .loaders import load_tools
from .providers import ProviderError, get_provider
from .questions import QuestionSet, generate_questions
from .report import Summary, print_summary, summarize

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
    err.print(f"[red]error:[/red] {escape(message)}")
    raise typer.Exit(code=1)


def _utf8_output() -> None:
    """Switch stdout and stderr to UTF-8 when they are not already.

    Each command calls this; the app callback does not, because it also runs for
    ``<command> --help``, whose output is best left as it is (plain ASCII when redirected).

    Reports print "→" and tool names can be in any language. A Windows console
    already gets UTF-8 from Python, but redirected or piped output (``> out.txt``,
    CI logs) defaults to a legacy code page that cannot encode them, and printing
    crashed half way through the report.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if codecs.lookup(stream.encoding).name != "utf-8":
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, LookupError, OSError, TypeError, ValueError):
            pass  # not a regular text stream (a test harness, say): leave it as it is


def _ascii_only() -> bool:
    """Plain ASCII tables and arrows when output goes to a file or pipe.

    Text from the tool files still goes out as UTF-8, but toolclash's own decorations
    stay readable even where redirected output is decoded with an old code page
    (Windows PowerShell 5.1 does that).
    """
    return not console.is_terminal


def _check_output_path(out: Path) -> None:
    """Catch an unusable -o before any paid model call is made. Leaves nothing behind."""
    if out.is_dir():
        raise ValueError(f"{out} is a folder; give a file name, e.g. {out / 'run.json'}")
    check_file_name(out)
    try:
        if not stat.S_ISREG(os.stat(out).st_mode):
            return  # a device such as NUL or CON: written to directly
        existed = True
    except OSError:
        existed = False
    missing = [p for p in [out.parent, *out.parent.parents] if not p.exists()]
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "ab"):  # changes nothing in an existing file
            pass
        if not existed:
            out.unlink()
    except (FileExistsError, NotADirectoryError):
        raise ValueError(f"cannot write {out}: part of that path is a file, not a folder") from None
    except PermissionError:
        why = "it is read-only or open in another program" if existed else "permission denied"
        raise ValueError(f"cannot write {out}: {why}") from None
    except OSError as e:
        raise ValueError(f"cannot write {out}: {e.strerror or e}") from None
    finally:
        for folder in missing:  # deepest first; save() creates them again when it writes
            try:
                folder.rmdir()
            except OSError:
                break


def _sibling(out: Path, tag: str) -> Path:
    """run.json -> run.<tag>.json (next to the output file)."""
    return out.with_name(f"{out.stem}.{tag}{out.suffix or '.json'}")


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
    _utf8_output()
    try:
        toolset = load_tools(path, fmt)  # type: ignore[arg-type]
    except (ValueError, OSError) as e:
        _fail(str(e))

    ascii_only = _ascii_only()
    table = Table(
        title=f"{len(toolset)} tools  (format: {toolset.source_format})",
        box=box.ASCII if ascii_only else box.HEAVY_HEAD,
    )
    table.add_column("name", style="bold")
    table.add_column("params")
    table.add_column("required")
    table.add_column("description")
    for tool in toolset.tools:
        # escape(): descriptions are shown as written, not read as rich markup
        table.add_row(
            escape(tool.name),
            escape(", ".join(tool.parameter_names) or "-"),
            escape(", ".join(tool.required_parameters) or "-"),
            escape(_one_line(tool.description, 60, ellipsis="..." if ascii_only else "…")),
        )
    if ascii_only:  # wrap long names instead of shortening them with a "…"
        for column in table.columns:
            column.overflow = "fold"
    console.print(table)


@app.command()
def generate(
    tools: ToolsArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write questions.json")],
    provider: ProviderOpt = "fake",
    per_tool: Annotated[int, typer.Option(min=0, help="Direct questions per tool.")] = 5,
    pairs: Annotated[
        int, typer.Option(min=0, help="How many similar tool pairs get ambiguous questions.")
    ] = 3,
    per_pair: Annotated[int, typer.Option(min=1, help="Ambiguous questions per direction per pair.")] = 3,
    no_tool: Annotated[int, typer.Option(min=0, help="Questions that no tool should answer.")] = 5,
    heldout: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Fraction of questions kept for the held-out set.")
    ] = 0.3,
) -> None:
    """Generate test questions from the tool schemas (descriptions are hidden from the generator)."""
    _utf8_output()
    try:
        _check_output_path(out)
        toolset = load_tools(tools)
        gen = get_provider(provider)
        qs = generate_questions(
            toolset, gen, per_tool=per_tool, pairs=pairs, per_pair=per_pair,
            no_tool=no_tool, heldout_fraction=heldout,
        )
        if not qs.questions:  # (not _fail here: typer.Exit is a RuntimeError, caught below)
            raise ValueError("nothing to generate: --per-tool, --pairs and --no-tool are all 0")
    except (ValueError, OSError, RuntimeError, ImportError) as e:
        _fail(str(e))
    _save_or_keep(qs, out)
    kinds = {k: sum(q.kind == k for q in qs.questions) for k in ("direct", "ambiguous", "no_tool")}
    held = sum(q.split == "heldout" for q in qs.questions)
    console.print(
        f"wrote {len(qs.questions)} questions to {escape(str(out))}  "
        f"(direct {kinds['direct']}, ambiguous {kinds['ambiguous']}, no-tool {kinds['no_tool']}; "
        f"{held} held out)  generator={escape(gen.name)}  calls={gen.usage.calls}"
    )


@app.command()
def run(
    tools: ToolsArg,
    questions: Annotated[Path, typer.Argument(help="questions.json from `generate`.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write run.json")],
    provider: ProviderOpt = "fake",
    samples: Annotated[int, typer.Option("--samples", "-n", min=1, help="Times to ask each question.")] = 5,
    split: Annotated[str, typer.Option(help=f"{', '.join(SPLITS[:-1])} or {SPLITS[-1]}.")] = "dev",
) -> None:
    """Ask the model which tool it would call for every question, several times each."""
    _utf8_output()
    try:
        _check_output_path(out)
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
    except KeyboardInterrupt as e:
        partial = getattr(e, "partial", None)
        err.print(f"[red]interrupted[/red]{escape(_save_partial(partial, out)) if partial else ''}")
        raise typer.Exit(code=130) from None
    except BenchAborted as e:
        if not isinstance(e.cause, ProviderError):  # a bug, not an API problem: keep the details
            traceback.print_exception(e.cause)
        _fail(f"{e}{_save_partial(e.partial, out)}")
    except (ValueError, OSError, RuntimeError, ImportError) as e:
        _fail(str(e))
    _save_or_keep(result, out)
    console.print(
        f"wrote {len(result.picks)} picks to {escape(str(out))}  provider={escape(model.name)}  "
        f"usage={result.usage}"
    )
    problems = Counter(p.error for p in result.picks if p.error)
    if problems:
        detail = ", ".join(f"{reason} {n}" for reason, n in sorted(problems.items()))
        err.print(
            f"[yellow]warning:[/yellow] {sum(problems.values())} of {len(result.picks)} samples got "
            f"no usable answer ({detail}); report leaves them out of every score."
        )


def _save_somewhere(result: BenchRun | QuestionSet, out: Path, tag: str) -> Path | None:
    """Save next to -o as run.<tag>.json, or in the temp folder if that fails too."""
    near = _sibling(out, tag)
    for path in (near, Path(tempfile.gettempdir()) / f"toolclash.{near.name}"):
        try:
            result.save(path)
            return path
        except (OSError, ValueError):
            continue
    return None


def _save_partial(partial: BenchRun, out: Path) -> str:
    """Keep the picks a failed run already paid for; return a note for the error message."""
    if not partial.picks:
        return ""
    path = _save_somewhere(partial, out, "partial")
    if path is None:
        return f" (the {len(partial.picks)} picks made so far could not be saved anywhere)"
    return f"\nthe {len(partial.picks)} picks made before it stopped were saved to {path}"


def _save_or_keep(result: BenchRun | QuestionSet, out: Path) -> None:
    """Save to -o; if that fails after the paid calls, keep the result in another file."""
    try:
        result.save(out)
        return
    except OSError as e:
        problem = f"could not write {out}: {e.strerror or e}"
    except ValueError as e:
        problem = str(e)
    fallback = _save_somewhere(result, out, "unsaved")
    if fallback is None:
        _fail(f"{problem}; the results could not be saved anywhere")
    _fail(f"{problem}\nthe results were saved to {fallback} instead")


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
    _utf8_output()
    try:
        if json_out:
            _check_output_path(json_out)
        summary = summarize(BenchRun.load(run_file))
    except (ValueError, OSError) as e:
        _fail(str(e))
    prices = (price_in, price_out) if price_in is not None and price_out is not None else None
    print_summary(summary, console, max_pairs=top, prices=prices, ascii_only=_ascii_only())
    if json_out:
        try:
            summary.save(json_out)
        except (OSError, ValueError) as e:
            _fail(f"could not write {json_out}: {getattr(e, 'strerror', None) or e}")
        console.print(f"wrote summary to {escape(str(json_out))}")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="run.json before the change.")],
    candidate: Annotated[Path, typer.Argument(help="run.json after the change.")],
    max_accuracy_drop: Annotated[float, typer.Option(help="Fail if accuracy drops more than this.")] = 0.10,
    max_stability_drop: Annotated[float, typer.Option(help="Fail if stability drops more than this.")] = 0.10,
) -> None:
    """CI gate: exit 1 if any existing tool got worse between two runs."""
    _utf8_output()
    try:
        before = summarize(BenchRun.load(baseline))
        after = summarize(BenchRun.load(candidate))
    except (ValueError, OSError) as e:
        _fail(str(e))
    result = compare_summaries(
        before, after, max_accuracy_drop=max_accuracy_drop, max_stability_drop=max_stability_drop
    )
    if result.new_tools:
        console.print(f"new tools: {escape(', '.join(result.new_tools))}")
    if result.removed_tools:
        console.print(f"removed tools: {escape(', '.join(result.removed_tools))}")
    for label, s in (("baseline", before), ("candidate", after)):
        if s.excluded:
            console.print(f"{label}: {_excluded_note(s)}")
    if result.skipped:
        console.print(f"not compared (not asked about in one of the runs): {escape(', '.join(result.skipped))}")
    if not result.compared:
        _fail(
            "no tool was asked about in both runs, so there is nothing to compare; check that "
            "both runs used the same questions file and --split"
        )
    if not result.usable_checked:
        console.print(
            "note: a run from an earlier toolclash version counted cut-off and refused replies "
            "as 'no tool', so usable answers were not compared; re-run it to include that check"
        )
    if result.passed:
        console.print("[green]PASS[/green] no existing tool regressed")
        return
    ascii_only = _ascii_only()
    for r in result.regressions:
        console.print(f"[red]FAIL[/red] {escape(r.describe(result.new_tools, ascii_only=ascii_only))}")
    raise typer.Exit(code=1)


def _excluded_note(s: Summary) -> str:
    detail = ", ".join(f"{reason} {n}" for reason, n in sorted(s.excluded.items()))
    return f"{sum(s.excluded.values())} samples had no usable answer ({detail})"


def _one_line(text: str, limit: int, ellipsis: str = "…") -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - len(ellipsis)] + ellipsis
