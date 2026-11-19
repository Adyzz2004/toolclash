from pathlib import Path

import pytest

from toolclash.loaders import load_tools
from toolclash.providers import FakeProvider
from toolclash.questions import generate_questions

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def toolset():
    return load_tools(FIXTURES / "mcp_tools.json")


@pytest.fixture
def questions(toolset):
    return generate_questions(
        toolset, FakeProvider(), per_tool=3, pairs=2, per_pair=2, no_tool=3, heldout_fraction=0.3
    )
