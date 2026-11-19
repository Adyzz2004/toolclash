import pytest

from toolclash.providers import FakeProvider
from toolclash.questions import (
    GenerationError,
    QuestionSet,
    _ambiguous_prompt,
    _direct_prompt,
    assign_split,
    generate_questions,
    parse_string_list,
    similar_pairs,
)


def test_counts_and_kinds(questions):
    kinds = {k: sum(q.kind == k for q in questions.questions) for k in ("direct", "ambiguous", "no_tool")}
    assert kinds == {"direct": 4 * 3, "ambiguous": 2 * 2 * 2, "no_tool": 3}
    assert questions.generator == "fake"


def test_expected_tool_is_set_correctly(questions):
    for q in questions.questions:
        if q.kind == "no_tool":
            assert q.expected is None
        else:
            assert q.expected in questions.tool_names
        if q.kind == "ambiguous":
            assert q.distractor in questions.tool_names and q.distractor != q.expected


def test_ids_are_unique(questions):
    ids = [q.id for q in questions.questions]
    assert len(ids) == len(set(ids))


def test_heldout_split_is_roughly_the_requested_fraction(toolset):
    qs = generate_questions(toolset, FakeProvider(), per_tool=50, pairs=0, no_tool=0, heldout_fraction=0.3)
    held = sum(q.split == "heldout" for q in qs.questions) / len(qs.questions)
    assert 0.2 < held < 0.4


def test_split_is_stable():
    assert assign_split("search_docs-direct-0", 0.3) == assign_split("search_docs-direct-0", 0.3)
    assert assign_split("anything", 0.0) == "dev"
    assert assign_split("anything", 1.0) == "heldout"


def test_subset(questions):
    dev, held = questions.subset("dev"), questions.subset("heldout")
    assert len(dev) + len(held) == len(questions.questions)
    assert len(questions.subset("all")) == len(questions.questions)


def test_similar_pairs_puts_overlapping_tools_first(toolset):
    pairs = [(a.name, b.name) for a, b in similar_pairs(toolset, 1)]
    assert pairs == [("search_docs", "get_doc")]  # both have "doc(s)" in the name
    assert len(similar_pairs(toolset, 100)) == 6  # C(4, 2), never more than exist


def test_generator_never_sees_descriptions(toolset):
    tool = toolset.get("search_docs")
    assert tool.description  # the fixture has one
    assert tool.description not in _direct_prompt(tool, 3)
    assert tool.description not in _ambiguous_prompt(tool, toolset.get("get_doc"), 3)
    assert '"name": "search_docs"' in _direct_prompt(tool, 3)  # but it does see the schema


@pytest.mark.parametrize(
    "text,expected",
    [
        ('["a", "b"]', ["a", "b"]),
        ('Sure! Here you go:\n```json\n["a", "b"]\n```', ["a", "b"]),
        ('["a", 2, "", "  b "]', ["a", "b"]),
        ("no list here", []),
        ("[not valid", []),
    ],
)
def test_parse_string_list(text, expected):
    assert parse_string_list(text) == expected


def test_generation_error_on_bad_reply(toolset):
    class Mute:
        name = "mute"
        usage = FakeProvider().usage

        def complete(self, prompt):
            return "I refuse."

        def choose_tool(self, q, t):
            raise NotImplementedError

    with pytest.raises(GenerationError):
        generate_questions(toolset, Mute(), per_tool=1, pairs=0, no_tool=0)


def test_roundtrip(questions, tmp_path):
    path = tmp_path / "q.json"
    questions.save(path)
    assert QuestionSet.load(path) == questions
