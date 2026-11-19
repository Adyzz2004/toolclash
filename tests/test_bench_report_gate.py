import pytest
from rich.console import Console

from toolclash.bench import BenchRun, Pick, run_bench
from toolclash.gate import compare
from toolclash.models import ToolSet
from toolclash.providers import FakeProvider
from toolclash.report import NONE, Summary, print_summary, summarize

# --- bench ----------------------------------------------------------------------------


def test_run_bench_asks_every_dev_question_n_times(toolset, questions):
    run = run_bench(toolset, questions, FakeProvider(), samples=3, split="dev")
    assert len(run.picks) == 3 * len(questions.subset("dev"))
    assert run.samples == 3 and run.split == "dev" and run.provider == "fake"
    assert run.usage["calls"] == len(run.picks)


def test_run_bench_all_split(toolset, questions):
    run = run_bench(toolset, questions, FakeProvider(), samples=1, split="all")
    assert len(run.picks) == len(questions.questions)


def test_run_bench_rejects_questions_for_missing_tools(toolset, questions):
    smaller = ToolSet(tools=toolset.tools[:2])
    with pytest.raises(ValueError, match="not in the tool set"):
        run_bench(smaller, questions, FakeProvider())


def test_run_bench_progress_callback(toolset, questions):
    seen = []
    run_bench(toolset, questions, FakeProvider(), samples=2, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == seen[-1][1] == 2 * len(questions.subset("dev"))


def test_bench_roundtrip(toolset, questions, tmp_path):
    run = run_bench(toolset, questions, FakeProvider(), samples=2)
    run.save(tmp_path / "run.json")
    assert BenchRun.load(tmp_path / "run.json") == run


# --- report -----------------------------------------------------------------------------


def _pick(qid, expected, picked, sample=0, kind=None):
    kind = kind or ("no_tool" if expected is None else "direct")
    return Pick(question_id=qid, kind=kind, expected=expected, picked=picked, sample=sample)


def _run(picks, tools=("a", "b")):
    return BenchRun(
        provider="fake", generator="fake", samples=2, split="dev",
        tool_names=list(tools), picks=picks, usage={"calls": len(picks)},
    )


def test_summary_numbers_by_hand():
    picks = [
        # question a1: both samples right
        _pick("a1", "a", "a", 0), _pick("a1", "a", "a", 1),
        # question a2: one right, one confused with b -> shaky
        _pick("a2", "a", "a", 0), _pick("a2", "a", "b", 1),
        # question b1: both wrong, confused with a
        _pick("b1", "b", "a", 0), _pick("b1", "b", "a", 1),
        # no-tool question: one over-trigger
        _pick("n1", None, None, 0), _pick("n1", None, "b", 1),
    ]
    s = summarize(_run(picks))

    assert s.matrix["a"] == {"a": 3, "b": 1, NONE: 0}
    assert s.matrix["b"] == {"a": 2, "b": 0, NONE: 0}
    assert s.matrix[NONE] == {"a": 0, "b": 1, NONE: 1}

    a, b = s.tools["a"], s.tools["b"]
    assert (a.questions, a.samples, a.correct) == (2, 4, 3)
    assert a.accuracy == 0.75
    assert a.stability == pytest.approx((1.0 + 0.5) / 2)
    assert a.flip_rate == 0.5
    assert (b.accuracy, b.stability, b.flip_rate) == (0.0, 1.0, 0.0)  # wrong, but confidently

    assert s.overall_accuracy == 3 / 6
    assert s.over_trigger_rate == 0.5
    pair = s.pairs[0]
    assert (pair.a, pair.b, pair.a_to_b, pair.b_to_a) == ("a", "b", 1, 2)
    assert pair.mutual_confusion == 3 / 6


def test_summary_handles_hallucinated_tool_name():
    s = summarize(_run([_pick("a1", "a", "made_up")]))
    assert "made_up" in s.labels
    assert s.matrix["a"]["made_up"] == 1
    assert s.tools["a"].accuracy == 0.0


def test_summary_with_no_no_tool_questions():
    s = summarize(_run([_pick("a1", "a", "a")]))
    assert s.over_trigger_rate is None


def test_summary_roundtrip_and_print(tmp_path):
    s = summarize(_run([_pick("a1", "a", "a"), _pick("n1", None, None)]))
    s.save(tmp_path / "s.json")
    assert Summary.load(tmp_path / "s.json") == s
    console = Console(record=True, width=120)
    print_summary(s, console)
    out = console.export_text()
    assert "Confusion matrix" in out and "overall accuracy 100%" in out


def test_end_to_end_fake_is_perfect_without_flips(toolset, questions):
    s = summarize(run_bench(toolset, questions, FakeProvider(), samples=3, split="all"))
    assert s.overall_accuracy == 1.0
    assert s.over_trigger_rate == 0.0
    assert all(t.flip_rate == 0.0 for t in s.tools.values())


# --- gate ---------------------------------------------------------------------------------


def _summary_with(acc: dict[str, float], stab: dict[str, float] | None = None) -> Summary:
    stab = stab or dict.fromkeys(acc, 1.0)
    picks = []
    for name, a in acc.items():
        n_right = round(a * 10)
        picks += [_pick(f"{name}-q", name, name, i) for i in range(n_right)]
        picks += [_pick(f"{name}-q", name, "other", i) for i in range(10 - n_right)]
    s = summarize(_run(picks, tools=list(acc) + ["other"]))
    for name, st in stab.items():  # stability is hard to fake from picks; set it directly
        s.tools[name].stability = st
    return s


def test_gate_passes_when_nothing_regressed():
    before = _summary_with({"a": 0.9, "b": 0.9})
    after = _summary_with({"a": 0.85, "b": 0.9, "c": 0.5})
    result = compare(before, after)
    assert result.passed
    assert result.new_tools == ["c"] and result.removed_tools == []


def test_gate_fails_on_accuracy_drop_and_names_the_cause():
    before = _summary_with({"a": 0.9, "b": 0.9})
    after = _summary_with({"a": 0.7, "b": 0.9, "send_slack": 0.5})
    result = compare(before, after, max_accuracy_drop=0.10)
    assert not result.passed
    r = result.regressions[0]
    assert (r.tool, r.metric) == ("a", "accuracy")
    assert "adding send_slack made a 20 points worse" in r.describe(result.new_tools)


def test_gate_fails_on_stability_drop_only():
    before = _summary_with({"a": 0.9}, {"a": 1.0})
    after = _summary_with({"a": 0.9}, {"a": 0.6})
    result = compare(before, after)
    assert [r.metric for r in result.regressions] == ["stability"]


def test_gate_ignores_tools_not_in_both():
    before = _summary_with({"a": 0.9, "old": 0.9})
    after = _summary_with({"a": 0.9})
    result = compare(before, after)
    assert result.passed and result.removed_tools == ["old"]


def test_estimate_cost_uses_per_million_prices():
    from toolclash.report import estimate_cost

    usage = {"input_tokens": 1_000_000, "output_tokens": 500_000}
    assert estimate_cost(usage, price_in=2.0, price_out=8.0) == pytest.approx(2.0 + 4.0)


def test_print_summary_shows_cost_when_prices_given():
    s = summarize(_run([_pick("a1", "a", "a")]))
    s.usage = {"calls": 1, "input_tokens": 1_000_000, "output_tokens": 0}
    console = Console(record=True, width=120)
    print_summary(s, console, prices=(3.0, 15.0))
    assert "est. cost $3.0000" in console.export_text()
