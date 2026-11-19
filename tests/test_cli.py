import json

from typer.testing import CliRunner

from toolclash.cli import app

from .conftest import FIXTURES

runner = CliRunner()
TOOLS = str(FIXTURES / "openai_tools.json")


def test_inspect():
    result = runner.invoke(app, ["inspect", TOOLS])
    assert result.exit_code == 0, result.output
    assert "search_docs" in result.output and "openai" in result.output


def test_inspect_missing_file(tmp_path):
    result = runner.invoke(app, ["inspect", str(tmp_path / "missing.json")])
    assert result.exit_code == 1


def test_full_pipeline_with_fake_provider(tmp_path):
    q, r, s = tmp_path / "q.json", tmp_path / "run.json", tmp_path / "summary.json"

    result = runner.invoke(app, ["generate", TOOLS, "-o", str(q), "--per-tool", "2", "--pairs", "1"])
    assert result.exit_code == 0, result.output
    assert "wrote" in result.output and q.exists()

    result = runner.invoke(app, ["run", TOOLS, str(q), "-o", str(r), "-n", "2", "--split", "all"])
    assert result.exit_code == 0, result.output
    assert r.exists()

    result = runner.invoke(app, ["report", str(r), "--json", str(s)])
    assert result.exit_code == 0, result.output
    assert "Confusion matrix" in result.output
    assert json.loads(s.read_text())["overall_accuracy"] == 1.0


def test_compare_passes_and_fails(tmp_path):
    q, base, shaky = tmp_path / "q.json", tmp_path / "base.json", tmp_path / "shaky.json"
    runner.invoke(app, ["generate", TOOLS, "-o", str(q), "--per-tool", "3"])
    runner.invoke(app, ["run", TOOLS, str(q), "-o", str(base), "-n", "3", "--split", "all"])
    runner.invoke(app, ["run", TOOLS, str(q), "-o", str(shaky), "-n", "3", "--split", "all", "-p", "fake:0.9"])

    ok = runner.invoke(app, ["compare", str(base), str(base)])
    assert ok.exit_code == 0 and "PASS" in ok.output

    bad = runner.invoke(app, ["compare", str(base), str(shaky)])
    assert bad.exit_code == 1 and "FAIL" in bad.output


def test_generate_rejects_bad_provider(tmp_path):
    result = runner.invoke(app, ["generate", TOOLS, "-o", str(tmp_path / "q.json"), "-p", "gemini:x"])
    assert result.exit_code == 1
