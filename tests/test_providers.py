import json

import pytest

from toolclash.loaders import load_tools, parse_tools
from toolclash.models import Tool, ToolSet
from toolclash.providers import (
    FakeProvider,
    get_provider,
    to_anthropic_tools,
    to_openai_tools,
)

from .conftest import FIXTURES


def test_exporters_roundtrip_through_loaders():
    """ToolSet -> provider format -> ToolSet must give back the same tools."""
    original = load_tools(FIXTURES / "mcp_tools.json")
    for exported in (to_openai_tools(original), to_anthropic_tools(original)):
        again = parse_tools(json.loads(json.dumps(exported)))
        assert [t.model_dump() for t in again.tools] == [t.model_dump() for t in original.tools]


def test_exporters_never_send_empty_schema():
    ts = ToolSet(tools=[Tool(name="ping")])
    assert to_openai_tools(ts)[0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert to_anthropic_tools(ts)[0]["input_schema"] == {"type": "object", "properties": {}}


def test_fake_picks_obvious_tool(toolset):
    fake = FakeProvider()
    assert fake.choose_tool("please send a slack message to #eng", toolset).tool == "send_slack_message"
    assert fake.choose_tool("search the docs for onboarding", toolset).tool == "search_docs"


def test_fake_returns_none_for_unrelated_request(toolset):
    assert FakeProvider().choose_tool("what is the capital of France?", toolset).tool is None


def test_fake_is_deterministic_for_a_seed(toolset):
    q = "I want to search docs, something like get doc"  # a tie between two tools
    a = [FakeProvider(seed=1).choose_tool(q, toolset).tool for _ in range(5)]
    b = [FakeProvider(seed=1).choose_tool(q, toolset).tool for _ in range(5)]
    assert a == b


def test_fake_flip_rate_adds_wrong_picks(toolset):
    shaky = FakeProvider(flip_rate=1.0, seed=3)
    picks = {shaky.choose_tool("send slack message", toolset).tool for _ in range(30)}
    assert len(picks) > 1


def test_fake_tracks_usage(toolset):
    fake = FakeProvider()
    fake.choose_tool("search docs", toolset)
    fake.complete("hello")
    assert fake.usage.calls == 2
    assert fake.usage.input_tokens > 0


def test_fake_complete_answers_direct_request():
    prompt = (
        "ignore this\n<request>\n"
        + json.dumps({"task": "direct", "n": 3, "tool": {"name": "search_docs", "parameters": {
            "type": "object", "properties": {"query": {}, "limit": {}}, "required": ["query"]}}})
        + "\n</request>\nreply"
    )
    lines = json.loads(FakeProvider().complete(prompt))
    assert len(lines) == 3
    assert all("search docs" in line for line in lines)
    assert all("query is" in line for line in lines)


def test_fake_complete_without_request_block_returns_empty_list():
    assert FakeProvider().complete("tell me a story") == "[]"


def test_get_provider_specs():
    assert isinstance(get_provider("fake"), FakeProvider)
    assert get_provider("fake:0.25").flip_rate == 0.25
    with pytest.raises(ValueError, match="needs a model"):
        get_provider("openai")
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("mistral:large")


def test_get_provider_gemini_needs_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_provider("gemini:some-model")


def test_get_provider_gemini_and_ollama_use_openai_compatible_client(monkeypatch):
    pytest.importorskip("openai")
    from toolclash.providers import GEMINI_BASE_URL, OpenAIProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    g = get_provider("gemini:some-model")
    assert isinstance(g, OpenAIProvider)
    assert g.name == "gemini:some-model"
    assert str(g.client.base_url).startswith(GEMINI_BASE_URL)

    monkeypatch.setenv("OLLAMA_HOST", "http://box:11434")
    o = get_provider("ollama:llama3")
    assert o.name == "ollama:llama3"
    assert str(o.client.base_url).startswith("http://box:11434/v1")
