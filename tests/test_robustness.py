"""Real-world failure modes: encodings, provider errors, cut-off replies, bad options."""

import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from toolclash import fileio
from toolclash.bench import BenchAborted, BenchRun, Pick, run_bench
from toolclash.cli import _utf8_output, app
from toolclash.gate import compare
from toolclash.loaders import LoadError, load_tools, parse_tools
from toolclash.providers import (
    FakeProvider,
    ProviderError,
    Usage,
    _describe_api_error,
    get_provider,
    ollama_base_url,
)
from toolclash.questions import QuestionSet, generate_questions
from toolclash.report import print_summary, summarize

from .conftest import FIXTURES
from .test_bench_report_gate import _pick, _run, _summary_with

runner = CliRunner()
TOOLS = str(FIXTURES / "mcp_tools.json")
UNICODE_TEXT = "Look up “cafés” — fast ≥ 5 → 🚀 测试"


@pytest.fixture(autouse=True)
def _independent_of_this_machine(monkeypatch):
    """Tests must not pick up the developer's Anthropic login or send anything anywhere."""
    for var in ("ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")


def _flat(output: str) -> str:
    """CLI output without colours or line wrapping, so assertions don't depend on the terminal."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output).split())


class _Flaky:
    """Answers like the fake provider for ``ok_calls`` calls, then fails like an overloaded API."""

    name = "flaky"

    def __init__(self, ok_calls: int):
        self.ok_calls = ok_calls
        self.usage = Usage()
        self.inner = FakeProvider()

    def choose_tool(self, question, tools):
        if self.usage.calls >= self.ok_calls:
            raise ProviderError("flaky: HTTP 529: overloaded_error: Overloaded")
        self.usage.add(1, 1)
        return self.inner.choose_tool(question, tools)

    def complete(self, prompt):
        return self.inner.complete(prompt)


class _UnicodeWriter(FakeProvider):
    """A generator whose questions contain characters a Windows code page cannot encode."""

    def complete(self, prompt):
        return json.dumps([UNICODE_TEXT, "测试一下"])


# --- files are UTF-8 on every platform ------------------------------------------------


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_tool_files_load_with_any_unicode_encoding(tmp_path, encoding):
    data = {"tools": [{"name": "lookup", "description": UNICODE_TEXT, "inputSchema": {
        "type": "object", "properties": {"kind": {"type": "string", "enum": ["café"]}}}}]}
    path = tmp_path / "tools.json"
    path.write_bytes(json.dumps(data, ensure_ascii=False).encode(encoding))
    tool = load_tools(path).tools[0]
    assert tool.description == UNICODE_TEXT
    assert tool.parameters["properties"]["kind"]["enum"] == ["café"]


def test_tool_file_that_is_not_utf8_is_a_clear_error(tmp_path):
    path = tmp_path / "tools.json"
    path.write_bytes('{"tools": [{"name": "café"}]}'.encode("cp1252"))
    with pytest.raises(LoadError, match="not UTF-8"):
        load_tools(path)


def test_questions_with_any_characters_round_trip_as_utf8(toolset, tmp_path):
    qs = generate_questions(toolset, _UnicodeWriter(), per_tool=1, pairs=0, no_tool=0)
    path = tmp_path / "nested" / "q.json"  # missing folders are created
    qs.save(path)
    assert UNICODE_TEXT in path.read_bytes().decode("utf-8")
    assert QuestionSet.load(path) == qs


def test_cli_generate_writes_unicode_questions(tmp_path, monkeypatch):
    monkeypatch.setattr("toolclash.cli.get_provider", lambda spec: _UnicodeWriter())
    out = tmp_path / "q.json"
    result = runner.invoke(app, ["generate", TOOLS, "-o", str(out), "--pairs", "0", "--no-tool", "0"])
    assert result.exit_code == 0, result.output
    assert any(q.text == UNICODE_TEXT for q in QuestionSet.load(out).questions)


def test_output_switches_to_utf8_when_the_console_uses_a_legacy_code_page(monkeypatch):
    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw_out, encoding="cp1252"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(raw_err, encoding="cp1252"))
    _utf8_output()
    print("A→B", file=sys.stdout)  # UnicodeEncodeError in cp1252
    sys.stdout.flush()
    assert raw_out.getvalue().decode("utf-8").strip() == "A→B"


def test_inspect_shows_descriptions_literally_not_as_markup(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"tools": [{"name": "x", "description": "ends [/b] early"}]}), "utf-8")
    result = runner.invoke(app, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "[/b]" in result.output


# --- loaders --------------------------------------------------------------------------


def test_wrapped_openai_and_anthropic_lists_keep_their_schemas():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    anthropic = parse_tools({"tools": [{"name": "x", "input_schema": schema}]})
    openai = parse_tools({"tools": [{"type": "function", "function": {"name": "x", "parameters": schema}}]})
    assert (anthropic.source_format, anthropic.tools[0].parameters) == ("anthropic", schema)
    assert (openai.source_format, openai.tools[0].parameters) == ("openai", schema)
    # an MCP list whose first tool takes no arguments is still MCP
    mcp = parse_tools({"tools": [{"name": "ping"}, {"name": "x", "inputSchema": schema}]})
    assert mcp.source_format == "mcp" and mcp.get("x").parameters == schema


# --- providers: clean errors, cut-off and refused replies -------------------------------


class _Stub:
    """Stands in for an SDK resource: returns canned responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)

    def create(self, **kwargs):
        return self.responses.pop(0)


def _claude(stop_reason, *blocks):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def test_anthropic_cut_off_or_refused_replies_are_flagged_not_counted_as_no_tool(toolset):
    pytest.importorskip("anthropic")
    from toolclash.providers import AnthropicProvider

    provider = AnthropicProvider("claude-opus-5-5")
    assert provider.max_tokens >= 16000  # room for thinking before the tool call
    provider.client = SimpleNamespace(base_url="stub", messages=_Stub(
        _claude("tool_use", _block("thinking", thinking=""), _block("tool_use", name="get_doc")),
        _claude("end_turn", _block("text", text="No tool fits.")),
        _claude("max_tokens", _block("thinking", thinking="")),
        _claude("refusal"),
    ))
    choices = [provider.choose_tool("q", toolset) for _ in range(4)]
    assert [(c.tool, c.error) for c in choices] == [
        ("get_doc", None), (None, None), (None, "truncated"), (None, "refused"),
    ]
    assert provider.usage.calls == 4


def test_anthropic_cut_off_generator_reply_is_a_clear_error():
    pytest.importorskip("anthropic")
    from toolclash.providers import AnthropicProvider

    provider = AnthropicProvider("claude-sonnet-5-5")
    provider.client = SimpleNamespace(base_url="stub", messages=_Stub(_claude("max_tokens")))
    with pytest.raises(ProviderError, match="truncated"):
        provider.complete("write questions")


@pytest.mark.parametrize("config_dir_set", [False, True])
def test_anthropic_missing_credentials_is_a_clean_error(monkeypatch, tmp_path, toolset, config_dir_set):
    pytest.importorskip("anthropic")
    from toolclash.providers import AnthropicProvider

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    # A machine with no saved login: the SDK's default config folders are empty.
    for var in ("HOME", "USERPROFILE", "APPDATA"):
        monkeypatch.setenv(var, str(tmp_path))
    if config_dir_set:  # pointing the SDK at a folder with no profile fails differently
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # never leaves the machine
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("claude-opus-5-5").choose_tool("search docs", toolset)


def test_anthropic_unreachable_api_is_a_clean_error(toolset):
    anthropic = pytest.importorskip("anthropic")
    from toolclash.providers import AnthropicProvider

    provider = AnthropicProvider("claude-opus-5-5")
    provider.client = anthropic.Anthropic(api_key="test", base_url="http://127.0.0.1:9", max_retries=0)
    with pytest.raises(ProviderError, match="could not reach"):
        provider.choose_tool("search docs", toolset)


def _gpt(finish_reason, tool=None, usage=True):
    calls = [SimpleNamespace(function=SimpleNamespace(name=tool))] if tool else None
    message = SimpleNamespace(tool_calls=calls, content=None, refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5) if usage else None,
    )


def test_openai_cut_off_or_filtered_replies_are_flagged(toolset):
    pytest.importorskip("openai")
    from toolclash.providers import OpenAIProvider

    provider = OpenAIProvider("some-model", api_key="test")
    provider.client = SimpleNamespace(base_url="stub", chat=SimpleNamespace(completions=_Stub(
        _gpt("tool_calls", tool="get_doc"), _gpt("stop"), _gpt("length", usage=False),
        _gpt("content_filter"),
    )))
    choices = [provider.choose_tool("q", toolset) for _ in range(4)]
    assert [(c.tool, c.error) for c in choices] == [
        ("get_doc", None), (None, None), (None, "truncated"), (None, "refused"),
    ]
    assert provider.usage.calls == 4  # a reply without usage still counts as a call


def test_openai_missing_key_is_a_clean_error(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        get_provider("openai:some-model")


def test_openai_unreachable_server_is_a_clean_error(toolset):
    pytest.importorskip("openai")
    from toolclash.providers import OpenAIProvider

    provider = OpenAIProvider("llama3", base_url="http://127.0.0.1:9/v1", api_key="test")
    provider.client = provider.client.with_options(max_retries=0)
    with pytest.raises(ProviderError, match="could not reach"):
        provider.choose_tool("search docs", toolset)


@pytest.mark.parametrize(
    "host,expected",
    [("0.0.0.0:11434", "http://127.0.0.1:11434/v1"), ("box:11434/", "http://box:11434/v1")],
)
def test_ollama_host_without_scheme(monkeypatch, host, expected):
    pytest.importorskip("openai")
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert str(get_provider("ollama:llama3").client.base_url).startswith(expected)


# --- bench: validation and partial runs ---------------------------------------------------


@pytest.mark.parametrize(
    "options,match", [({"split": "test"}, "unknown split"), ({"samples": 0}, "at least 1")]
)
def test_run_bench_rejects_bad_options(toolset, questions, options, match):
    with pytest.raises(ValueError, match=match):
        run_bench(toolset, questions, FakeProvider(), **options)


def test_run_bench_rejects_a_question_for_an_unknown_tool(toolset, questions):
    bad = questions.model_copy(deep=True)
    bad.questions[0].expected = "no_such_tool"
    with pytest.raises(ValueError, match="no_such_tool"):
        run_bench(toolset, bad, FakeProvider(), split="all")


def test_run_bench_keeps_the_picks_made_before_a_failure(toolset, questions):
    with pytest.raises(BenchAborted, match="HTTP 529") as info:
        run_bench(toolset, questions, _Flaky(ok_calls=3), samples=2, split="all")
    assert len(info.value.partial.picks) == 3
    assert info.value.partial.usage["calls"] == 3


def test_cli_run_saves_partial_picks_and_reports_a_clean_error(tmp_path, monkeypatch):
    q, out = tmp_path / "q.json", tmp_path / "run.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    monkeypatch.setattr("toolclash.cli.get_provider", lambda spec: _Flaky(ok_calls=4))
    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(out), "--split", "all"])
    assert result.exit_code == 1
    assert "HTTP 529" in _flat(result.output) and "Traceback" not in result.output
    assert not out.exists()
    assert len(BenchRun.load(tmp_path / "run.partial.json").picks) == 4


def test_cli_run_rejects_an_unknown_split(tmp_path):
    q = tmp_path / "q.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(tmp_path / "r.json"), "--split", "test"])
    assert result.exit_code == 1 and "unknown split" in _flat(result.output)


# --- report -------------------------------------------------------------------------------


def _printed(summary) -> str:
    console = Console(record=True, width=200)
    print_summary(summary, console)
    return console.export_text()


def test_report_prints_a_hallucinated_tool_name():
    out = _printed(summarize(_run([_pick("a1", "a", "made_up"), _pick("a1", "a", "a", 1)])))
    assert "made_up" in out and "overall accuracy 50%" in out


def test_cut_off_and_refused_samples_are_excluded_and_reported():
    picks = [
        _pick("a1", "a", "a", 0),
        Pick(question_id="a1", kind="direct", expected="a", picked=None, sample=1, error="truncated"),
        _pick("n1", None, None, 0),
        Pick(question_id="n1", kind="no_tool", expected=None, picked=None, sample=1, error="refused"),
    ]
    s = summarize(_run(picks))
    assert s.excluded == {"truncated": 1, "refused": 1}
    assert (s.tools["a"].samples, s.tools["a"].accuracy, s.tools["a"].flip_rate) == (1, 1.0, 0.0)
    assert s.over_trigger_rate == 0.0 and sum(s.matrix["(none)"].values()) == 1
    assert "2 samples had no usable answer" in _printed(s)


def test_a_tool_without_data_is_shown_as_n_a_not_zero():
    out = _printed(summarize(_run([_pick("a1", "a", "a")])))  # tool "b" has no questions
    rows = [[cell.strip() for cell in line.split("│") if cell.strip()] for line in out.splitlines()]
    assert ["b", "0", "n/a", "n/a", "n/a"] in rows


# --- gate ---------------------------------------------------------------------------------


@pytest.mark.parametrize("before,after", [(0.9, 0.8), (0.8, 0.7), (0.3, 0.2)])
def test_gate_treats_a_drop_equal_to_the_threshold_the_same_at_every_level(before, after):
    result = compare(_summary_with({"a": before}), _summary_with({"a": after}), max_accuracy_drop=0.10)
    assert result.passed, result.regressions


def test_gate_fails_when_nothing_can_be_compared():
    result = compare(_summary_with({"a": 0.9}), _summary_with({"b": 0.9}))
    assert result.compared == [] and not result.passed


def test_cli_compare_fails_when_the_runs_share_no_tools(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _run([_pick("x1", "x", "x")], tools=("x",)).save(a)
    _run([_pick("y1", "y", "y")], tools=("y",)).save(b)
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 1 and "nothing to compare" in _flat(result.output)


# --- second review round ----------------------------------------------------------------


def test_files_written_by_earlier_versions_in_a_windows_code_page_still_load(
    questions, tmp_path, monkeypatch
):
    monkeypatch.setattr(fileio, "_legacy_encoding", lambda: "cp1252")
    qs = questions.model_copy(deep=True)
    qs.questions[0].text = "Help me search docs — query is “café”"
    path = tmp_path / "old_q.json"
    path.write_bytes(qs.model_dump_json(indent=2).encode("cp1252"))  # how earlier versions wrote it
    assert QuestionSet.load(path) == qs


def test_an_unreadable_file_error_names_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fileio, "_legacy_encoding", lambda: "utf-8")
    path = tmp_path / "broken_run.json"
    path.write_bytes(b'{"provider": "\xff\xfe\xfa"}')
    with pytest.raises(ValueError, match="broken_run.json is not UTF-8"):
        BenchRun.load(path)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "http://127.0.0.1:11434/v1"),
        ("", "http://127.0.0.1:11434/v1"),
        ("0.0.0.0", "http://127.0.0.1:11434/v1"),
        ('"myserver"', "http://myserver:11434/v1"),
        ("myserver:8080", "http://myserver:8080/v1"),
        ("::1", "http://[::1]:11434/v1"),
        ("[::1]:9999", "http://[::1]:9999/v1"),
        ("http://box", "http://box:80/v1"),
        ("https://box", "https://box:443/v1"),
        ("http://box:11434/", "http://box:11434/v1"),
        ("http://box:11434/v1", "http://box:11434/v1"),
        ("box:notaport", "http://box:11434/v1"),
        ("http://user:pw@box:11434", "http://user:pw@box:11434/v1"),
        ("user:pw@box", "http://user:pw@box:11434/v1"),
    ],
)
def test_ollama_host_is_read_like_ollama_reads_it(value, expected):
    assert ollama_base_url(value) == expected


class _FakeAPIError(Exception):
    def __init__(self, message, status=None, body=None, response_text=""):
        super().__init__(message)
        self.message, self.status_code, self.body = message, status, body
        self.response = SimpleNamespace(text=response_text)


def test_api_error_messages_are_one_clean_line():
    html = "<html>\n<head><title>502 Bad Gateway</title></head>\n<body>nginx</body></html>"
    assert _describe_api_error(_FakeAPIError(html, 502, body=html), "u") == "HTTP 502: 502 Bad Gateway"
    plain = _FakeAPIError("Error code: 404", 404, response_text="404 page not found\n")
    assert _describe_api_error(plain, "u") == "HTTP 404: 404 page not found"
    body = {"type": "error", "error": {"type": "not_found_error", "message": "model: x"}}
    assert _describe_api_error(_FakeAPIError("", 404, body=body), "u") == "HTTP 404: not_found_error: model: x"


def test_connection_errors_show_the_cause_but_not_passwords():
    try:
        try:
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        except OSError as cause:
            raise _FakeAPIError("Connection error.") from cause
    except _FakeAPIError as e:
        message = _describe_api_error(e, "http://admin:S3CRET@gateway:8080/v1/")
    assert "S3CRET" not in message and "admin" not in message
    assert "gateway:8080" in message and "CERTIFICATE_VERIFY_FAILED" in message


def test_malformed_replies_are_clean_errors(toolset):
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    from toolclash.providers import AnthropicProvider, OpenAIProvider

    claude = AnthropicProvider("claude-opus-5-5")
    claude.client = SimpleNamespace(base_url="stub", messages=_Stub("<html>oops</html>"))
    with pytest.raises(ProviderError, match="unexpected reply"):
        claude.choose_tool("q", toolset)
    gpt = OpenAIProvider("m", api_key="test")
    gpt.client = SimpleNamespace(base_url="stub", chat=SimpleNamespace(completions=_Stub(
        SimpleNamespace(choices=[], usage=None))))
    with pytest.raises(ProviderError, match="unexpected reply"):
        gpt.choose_tool("q", toolset)


def test_other_anthropic_sdk_errors_are_clean_errors(toolset):
    anthropic = pytest.importorskip("anthropic")
    from toolclash.providers import AnthropicProvider

    class _Raises:
        def create(self, **kwargs):
            raise anthropic.AnthropicError("token refresh failed: invalid_grant")

    provider = AnthropicProvider("claude-opus-5-5")
    provider.client = SimpleNamespace(base_url="stub", messages=_Raises())
    with pytest.raises(ProviderError, match="invalid_grant"):
        provider.choose_tool("q", toolset)


class _Interrupted(_Flaky):
    def choose_tool(self, question, tools):
        if self.usage.calls >= self.ok_calls:
            raise KeyboardInterrupt
        return super().choose_tool(question, tools)


def test_ctrl_c_stays_a_keyboard_interrupt_and_carries_the_partial_run(toolset, questions):
    with pytest.raises(KeyboardInterrupt) as info:
        run_bench(toolset, questions, _Interrupted(ok_calls=2), split="all")
    assert len(info.value.partial.picks) == 2


def test_cli_ctrl_c_saves_the_partial_run(tmp_path, monkeypatch):
    q, out = tmp_path / "q.json", tmp_path / "run.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    monkeypatch.setattr("toolclash.cli.get_provider", lambda spec: _Interrupted(ok_calls=3))
    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(out), "--split", "all"])
    assert result.exit_code == 130
    assert len(BenchRun.load(tmp_path / "run.partial.json").picks) == 3


def test_cli_rejects_a_folder_as_output_before_any_model_call(tmp_path, monkeypatch):
    q = tmp_path / "q.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    provider = _Flaky(ok_calls=1000)
    monkeypatch.setattr("toolclash.cli.get_provider", lambda spec: provider)
    for args in (["run", TOOLS, str(q), "-o", str(tmp_path)], ["generate", TOOLS, "-o", str(tmp_path)]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1 and "is a folder" in _flat(result.output)
    assert provider.usage.calls == 0


def test_cli_keeps_a_finished_run_when_the_output_cannot_be_written(tmp_path, monkeypatch):
    q, out = tmp_path / "q.json", tmp_path / "run.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    n_questions = len(QuestionSet.load(q).questions)
    real_save = BenchRun.save

    def save(self, path):
        if Path(path) == out:
            raise PermissionError(f"cannot write {path}: it is read-only or open in another program")
        real_save(self, path)

    monkeypatch.setattr(BenchRun, "save", save)
    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(out), "-n", "2", "--split", "all"])
    assert result.exit_code == 1 and "could not write" in _flat(result.output)
    assert len(BenchRun.load(tmp_path / "run.unsaved.json").picks) == 2 * n_questions


def test_redirected_report_and_compare_output_is_plain_ascii(tmp_path, monkeypatch):
    monkeypatch.setattr("toolclash.cli._ascii_only", lambda: True)
    q, a, b = tmp_path / "q.json", tmp_path / "a.json", tmp_path / "b.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    runner.invoke(app, ["run", TOOLS, str(q), "-o", str(a), "-n", "3", "--split", "all", "-p", "fake:0.15"])
    runner.invoke(app, ["run", TOOLS, str(q), "-o", str(b), "-n", "3", "--split", "all", "-p", "fake:0.9"])
    for args in (["inspect", TOOLS], ["report", str(a)], ["compare", str(a), str(b)]):
        result = runner.invoke(app, args)
        assert result.output.isascii(), (args, result.output)
    assert "->" in runner.invoke(app, ["compare", str(a), str(b)]).output


def _with_errors(picks_ok, picks_refused, tool="a"):
    picks = [_pick(f"{tool}{i}", tool, tool, 0) for i in range(picks_ok)]
    picks += [
        Pick(question_id=f"{tool}r{i}", kind="direct", expected=tool, picked=None, sample=0, error="refused")
        for i in range(picks_refused)
    ]
    run = _run(picks, tools=(tool, "b"))
    run.tracks_errors = True  # as every run made by this version is
    return summarize(run)


def test_gate_fails_when_the_model_stops_giving_usable_answers():
    result = compare(_with_errors(10, 0), _with_errors(0, 10))
    assert not result.passed
    assert [(r.tool, r.metric, r.after) for r in result.regressions] == [("a", "usable answers", 0.0)]
    assert compare(_with_errors(10, 0), _with_errors(9, 1)).passed  # within the 10-point allowance


def test_gate_skips_the_usable_answers_check_for_a_baseline_from_an_earlier_version():
    old = _with_errors(8, 0)
    old.tracks_errors = False  # it recorded its 2 refusals as "no tool", not as errors
    result = compare(old, _with_errors(8, 2))
    assert result.usable_checked is False
    assert all(r.metric != "usable answers" for r in result.regressions)


def test_writing_to_a_device_such_as_the_null_device_works():
    fileio.write_text(os.devnull, "{}")  # NUL on Windows, /dev/null elsewhere


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file-name rules")
def test_cli_rejects_a_windows_invalid_output_name_before_any_model_call(tmp_path, monkeypatch):
    q = tmp_path / "q.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    provider = _Flaky(ok_calls=1000)
    monkeypatch.setattr("toolclash.cli.get_provider", lambda spec: provider)
    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(tmp_path / "openai:gpt-4o.json")])
    assert result.exit_code == 1 and "not a valid file name" in _flat(result.output)
    assert provider.usage.calls == 0 and sorted(p.name for p in tmp_path.iterdir()) == ["q.json"]


def test_cli_falls_back_to_the_temp_folder_when_the_output_folder_fails(tmp_path, monkeypatch):
    q, out = tmp_path / "q.json", tmp_path / "runs" / "run.json"
    assert runner.invoke(app, ["generate", TOOLS, "-o", str(q)]).exit_code == 0
    real_save = BenchRun.save

    def save(self, path):
        if Path(path).parent == out.parent:  # the folder itself has become unusable
            raise PermissionError(13, "Permission denied")
        real_save(self, path)

    monkeypatch.setattr(BenchRun, "save", save)
    temp = tmp_path / "temp"  # never touch the real temp folder: it may hold a rescued run
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(out), "-n", "1", "--split", "all"])
    # paths may be folded across lines, so compare without any whitespace
    assert result.exit_code == 1 and "toolclash.run.unsaved.json" in "".join(_flat(result.output).split())
    assert BenchRun.load(temp / "toolclash.run.unsaved.json").picks


def test_output_check_leaves_no_folders_behind_when_a_later_step_fails(tmp_path):
    out = tmp_path / "results" / "today" / "run.json"
    result = runner.invoke(app, ["run", TOOLS, str(tmp_path / "missing_q.json"), "-o", str(out)])
    assert result.exit_code == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_file_that_may_be_changed_in_a_folder_closed_to_new_files_is_written_in_place(
    tmp_path, monkeypatch
):
    target = tmp_path / "run.json"
    target.write_text("old", encoding="utf-8")

    def no_new_files(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(fileio, "_create_temp", no_new_files)
    fileio.write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows devices and file-name rules")
def test_windows_console_device_and_bad_json_name(tmp_path):
    from toolclash.cli import _check_output_path

    _check_output_path(Path("CON"))  # a device: accepted, written to directly
    a = tmp_path / "a.json"
    _run([_pick("a1", "a", "a")]).save(a)
    result = runner.invoke(app, ["report", str(a), "--json", str(tmp_path / "openai:x.json")])
    assert result.exit_code == 1 and "not a valid file name" in _flat(result.output)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]


def test_legacy_encoding_without_locale_getencoding(monkeypatch):
    # As on Python 3.10, where locale has no getencoding(). Swap in a stand-in module rather
    # than deleting the function: on 3.11+ locale's own getpreferredencoding() calls it.
    stand_in = SimpleNamespace(getpreferredencoding=lambda do_setlocale=True: "cp1252")
    monkeypatch.setattr(fileio, "locale", stand_in)
    encoding = fileio._legacy_encoding()
    if sys.platform == "win32":
        assert encoding.startswith("cp")  # asked from Windows itself
    else:
        assert encoding == "cp1252"


def test_gate_with_an_old_baseline_still_catches_a_tool_with_no_usable_answers():
    old = _with_errors(10, 0)
    old.tracks_errors = False
    result = compare(old, _with_errors(0, 10))
    assert not result.passed
    assert [(r.tool, r.metric, r.after) for r in result.regressions] == [("a", "accuracy", 0.0)]


def test_openai_reply_that_is_not_json_is_a_clean_error(toolset):
    pytest.importorskip("openai")
    from toolclash.providers import OpenAIProvider

    class _BadJSON:
        def create(self, **kwargs):
            return json.loads("{truncated")

    provider = OpenAIProvider("m", api_key="test")
    provider.client = SimpleNamespace(base_url="stub", chat=SimpleNamespace(completions=_BadJSON()))
    with pytest.raises(ProviderError, match="could not be parsed"):
        provider.choose_tool("q", toolset)


def test_a_bad_openai_base_url_in_the_environment_never_echoes_the_login(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://user:pa#ss@host:1/v1")
    with pytest.raises(ProviderError, match="OPENAI_BASE_URL") as info:
        get_provider("openai:some-model")
    assert "pa#ss" not in str(info.value) and "user" not in str(info.value)


def test_nested_malformed_replies_are_clean_errors(toolset):
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    from toolclash.providers import AnthropicProvider, OpenAIProvider

    claude = AnthropicProvider("claude-opus-5-5")
    claude.client = SimpleNamespace(base_url="stub", messages=_Stub(_claude("end_turn", "just a string")))
    with pytest.raises(ProviderError, match="unexpected reply"):
        claude.choose_tool("q", toolset)
    no_function = SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
        tool_calls=[SimpleNamespace(id="x")], content=None, refusal=None))
    gpt = OpenAIProvider("m", api_key="test")
    gpt.client = SimpleNamespace(base_url="stub", chat=SimpleNamespace(completions=_Stub(
        SimpleNamespace(choices=[no_function], usage=SimpleNamespace(prompt_tokens="7", completion_tokens=1)))))
    with pytest.raises(ProviderError, match="unexpected reply"):
        gpt.choose_tool("q", toolset)
    assert (gpt.usage.calls, gpt.usage.input_tokens) == (1, 0)  # "7" is not a number of tokens


def test_an_unusable_base_url_is_a_clean_error():
    pytest.importorskip("openai")
    from toolclash.providers import OpenAIProvider

    with pytest.raises(ProviderError, match="ollama:llama3"):
        OpenAIProvider("llama3", base_url="http://box:11434:80/v1", api_key="x", label="ollama")


@pytest.mark.skipif(os.name != "posix", reason="file modes are a POSIX thing")
def test_written_files_get_normal_permissions(tmp_path):
    umask = os.umask(0o022)
    try:
        fresh = tmp_path / "fresh.json"
        fileio.write_text(fresh, "{}")
        assert fresh.stat().st_mode & 0o777 == 0o644
        shared = tmp_path / "shared.json"
        shared.write_text("{}")
        shared.chmod(0o664)
        fileio.write_text(shared, "{}")
        assert shared.stat().st_mode & 0o777 == 0o664
    finally:
        os.umask(umask)


def test_write_falls_back_to_overwriting_when_windows_will_not_replace(tmp_path, monkeypatch):
    target = tmp_path / "held.json"
    target.write_text("old", encoding="utf-8")

    def refuse(src, dst):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(fileio.os, "replace", refuse)
    monkeypatch.setattr(fileio, "_on_windows", lambda: True)
    monkeypatch.setattr(fileio.time, "sleep", lambda seconds: None)
    fileio.write_text(target, "new → text")
    assert target.read_text(encoding="utf-8") == "new → text"
    assert [p.name for p in tmp_path.iterdir()] == ["held.json"]  # no temp file left behind
