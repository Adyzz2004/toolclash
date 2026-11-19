"""toolclash: measure which of your agent's tools the model confuses with each other."""

from .bench import BenchRun, run_bench
from .gate import GateResult, compare
from .loaders import LoadError, load_tools, parse_tools
from .models import Tool, ToolSet
from .providers import FakeProvider, Provider, get_provider
from .questions import QuestionSet, generate_questions
from .report import Summary, summarize

__all__ = [
    "BenchRun",
    "FakeProvider",
    "GateResult",
    "LoadError",
    "Provider",
    "QuestionSet",
    "Summary",
    "Tool",
    "ToolSet",
    "compare",
    "generate_questions",
    "get_provider",
    "load_tools",
    "parse_tools",
    "run_bench",
    "summarize",
]
__version__ = "0.1.0"
