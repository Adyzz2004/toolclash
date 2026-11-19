from pathlib import Path

import pytest
from typer.testing import CliRunner

from toolclash.cli import app
from toolclash.loaders import LoadError, detect_format, load_tools, parse_tools
from toolclash.models import Tool, ToolSet

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_FILES = ["mcp_tools.json", "openai_tools.json", "anthropic_tools.json"]
EXPECTED_NAMES = ["search_docs", "get_doc", "send_slack_message", "create_ticket"]


# --- loading each supported format ----------------------------------------------


@pytest.mark.parametrize(
    "filename,fmt",
    [
        ("mcp_tools.json", "mcp"),
        ("openai_tools.json", "openai"),
        ("anthropic_tools.json", "anthropic"),
    ],
)
def test_load_each_format(filename, fmt):
    toolset = load_tools(FIXTURES / filename)
    assert toolset.source_format == fmt
    assert toolset.names() == EXPECTED_NAMES


def test_all_formats_normalise_to_the_same_tools():
    """The whole point of the loader: three input shapes, one output shape."""
    dumped = [
        [t.model_dump() for t in load_tools(FIXTURES / f).tools] for f in FIXTURE_FILES
    ]
    assert dumped[0] == dumped[1] == dumped[2]


def test_mcp_jsonrpc_wrapper():
    data = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]},
    }
    toolset = parse_tools(data)
    assert toolset.source_format == "mcp"
    assert toolset.names() == ["ping"]


def test_mcp_bare_list():
    toolset = parse_tools([{"name": "ping", "inputSchema": {"type": "object"}}])
    assert toolset.source_format == "mcp"


def test_openai_flat_responses_shape():
    data = [{"type": "function", "name": "ping", "parameters": {"type": "object"}}]
    toolset = parse_tools(data)
    assert toolset.source_format == "openai"
    assert toolset.get("ping").parameters == {"type": "object"}


def test_explicit_format_overrides_detection():
    data = [{"name": "ping", "input_schema": {"type": "object"}}]
    assert parse_tools(data, fmt="anthropic").source_format == "anthropic"


def test_missing_description_and_schema_default_to_empty():
    tool = parse_tools([{"name": "ping", "input_schema": None}]).get("ping")
    assert tool.description == ""
    assert tool.parameters == {}


# --- rejecting bad input ----------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"nope": 1},
        [{"foo": "bar"}],
        ["not an object"],
        "plain string",
    ],
)
def test_detect_format_rejects_unrecognised_input(data):
    with pytest.raises(LoadError):
        detect_format(data)


def test_unknown_explicit_format_rejected():
    with pytest.raises(LoadError, match="unknown format"):
        parse_tools([{"name": "x", "input_schema": {}}], fmt="gemini")  # type: ignore[arg-type]


def test_tool_without_name_rejected():
    with pytest.raises(LoadError, match="no name"):
        parse_tools([{"input_schema": {}}])


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ToolSet(tools=[Tool(name="a"), Tool(name="a")])


def test_invalid_json_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(LoadError, match="not valid JSON"):
        load_tools(bad)


# --- model helpers ----------------------------------------------------------------


def test_parameter_helpers():
    tool = Tool(
        name="x",
        description="desc",
        parameters={"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a"]},
    )
    assert tool.parameter_names == ["a", "b"]
    assert tool.required_parameters == ["a"]
    stripped = tool.schema_only()
    assert stripped.description == ""
    assert stripped.parameters == tool.parameters
    assert tool.description == "desc"  # original untouched


def test_toolset_get_unknown_name():
    with pytest.raises(KeyError):
        ToolSet(tools=[]).get("missing")


def test_toolset_roundtrip(tmp_path):
    original = load_tools(FIXTURES / "mcp_tools.json")
    path = tmp_path / "toolset.json"
    original.save(path)
    assert ToolSet.load(path) == original


# --- CLI --------------------------------------------------------------------------


def test_cli_inspect():
    result = CliRunner().invoke(app, ["inspect", str(FIXTURES / "openai_tools.json")])
    assert result.exit_code == 0, result.output
    assert "search_docs" in result.output
    assert "openai" in result.output


def test_cli_inspect_missing_file(tmp_path):
    result = CliRunner().invoke(app, ["inspect", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "error" in result.output
