"""Model providers: the only place in the project that talks to an LLM API.

Every provider does two jobs:

1. ``choose_tool(question, tools)`` — show the model the tool list and one user
   request, and return the name of the tool it decided to call (or ``None`` if
   it called nothing). This is what the bench measures.
2. ``complete(prompt)`` — plain text in, plain text out. Used by the question
   generator, which must run on a *different* model than the one being benched.

``FakeProvider`` needs no network or API key. It is used by the tests and by
``--provider fake`` so the whole pipeline can be run end to end for free.
"""

from __future__ import annotations

import ipaddress
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .models import Tool, ToolSet

SYSTEM_PROMPT = (
    "You are an assistant that can call tools. Read the user's request and call the "
    "single most appropriate tool with sensible arguments. If none of the tools fits "
    "the request, do not call any tool; reply with one short sentence instead."
)

# Providers reject an empty schema; this is the smallest valid "no arguments" schema.
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class Usage:
    """Token counts for one provider, summed over every call. Cost = tokens x price."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.calls += 1
        # Some OpenAI-compatible servers send odd values here; count only real numbers.
        self.input_tokens += input_tokens if isinstance(input_tokens, int) else 0
        self.output_tokens += output_tokens if isinstance(output_tokens, int) else 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class Choice:
    """What the model picked for one question. ``tool`` is None when it called nothing.

    ``error`` is set when the reply carries no usable decision: ``"truncated"`` (it ran
    out of tokens, e.g. while thinking) or ``"refused"``. Counting those as "called no
    tool" would quietly lower accuracy and hide over-triggering, so the report leaves
    them out of every score and says how many there were.
    """

    tool: str | None
    raw: Any = None  # the provider's own response object, kept for debugging
    error: str | None = None


class ProviderError(RuntimeError):
    """A model API call failed: missing key, unknown model, rate limit, network, ...

    The message is one readable line, meant to be shown to the user as is.
    """


def _describe_api_error(e: Exception, base_url: object) -> str:
    """One readable line from an OpenAI or Anthropic SDK exception: no HTML, no passwords."""
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    if isinstance(body, list) and body:  # Gemini wraps its error object in a list
        body = body[0]
    detail: Any = None
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            kind, message = err.get("type"), err.get("message")
            detail = f"{kind}: {message}" if kind and message else message or kind
        else:
            detail = err
    elif isinstance(body, str):
        detail = body
    if not detail and status is not None:
        detail = _response_text(e)  # e.g. a gateway's HTML or plain-text error page
    detail = _one_line(str(detail or getattr(e, "message", "") or e or type(e).__name__))
    if status is None:  # connection error or timeout: no HTTP response at all
        cause = _root_cause(e)  # the SDK says only "Connection error."; the cause says why
        if cause is not None and str(cause) and str(cause) not in detail:
            detail = f"{detail} {_one_line(str(cause))}"
        return _scrub(f"could not reach {_without_userinfo(base_url)} ({detail})")
    return _scrub(f"HTTP {status}: {detail}")


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_USERINFO_RE = re.compile(r"://[^/\s@]+@")


def _one_line(text: str, limit: int = 300) -> str:
    """Collapse an error text to one short line; an HTML page shrinks to its title."""
    title = _TITLE_RE.search(text)
    if title:
        text = title.group(1)
    elif text.lstrip().startswith("<"):
        text = _TAG_RE.sub(" ", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _response_text(e: Exception) -> str:
    try:
        return e.response.text  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, ValueError):  # no response, or its body was not read
        return ""


def _root_cause(e: BaseException) -> BaseException | None:
    cause, seen = e.__cause__ or e.__context__, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        deeper = cause.__cause__ or cause.__context__
        if deeper is None:
            return cause
        cause = deeper
    return cause


def _without_userinfo(url: object) -> str:
    """A URL fit for an error message: any user:password@ part is dropped."""
    parts = urlsplit(str(url))
    if "@" not in parts.netloc:
        return str(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _scrub(message: str) -> str:
    return _USERINFO_RE.sub("://", message)


_REPLY_ERRORS = (AttributeError, TypeError, KeyError, IndexError)  # reading a malformed reply


def _unexpected_reply(name: str, base_url: object, what: str) -> ProviderError:
    return ProviderError(f"{name}: unexpected reply from {_without_userinfo(base_url)} ({what})")


class Provider(Protocol):
    name: str
    usage: Usage

    def choose_tool(self, question: str, tools: ToolSet) -> Choice: ...

    def complete(self, prompt: str) -> str: ...


# --- ToolSet -> provider wire format (the reverse of loaders.py) --------------------


def to_openai_tools(tools: ToolSet) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or _EMPTY_SCHEMA,
            },
        }
        for t in tools.tools
    ]


def to_anthropic_tools(tools: ToolSet) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters or _EMPTY_SCHEMA,
        }
        for t in tools.tools
    ]


# --- real providers -------------------------------------------------------------------


class OpenAIProvider:
    """Chat Completions API with function tools. Reads OPENAI_API_KEY from the environment.

    Also serves Gemini and Ollama: both expose an OpenAI-compatible endpoint, so they
    are this same class pointed at a different ``base_url`` (see ``get_provider``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        label: str = "openai",
    ):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'toolclash[openai]' to use the openai provider") from e
        self.model = model
        self.temperature = temperature
        self.name = f"{label}:{model}"
        self.usage = Usage()
        self._api_error = openai.APIError
        self._sdk_error = openai.OpenAIError
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if url:
            try:
                _ = urlsplit(url).port  # raises ValueError for a malformed URL
            except ValueError:
                # The error text can quote part of a password, so it is not repeated here.
                raise ProviderError(
                    f"{self.name}: the base URL {'' if base_url else '(OPENAI_BASE_URL) '}is not a valid "
                    "URL; check the host and port, and percent-encode special characters in a login"
                ) from None
        try:
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        except openai.OpenAIError as e:  # raised right away when no API key is set
            raise ProviderError(f"{self.name}: {_scrub(_one_line(str(e)))}") from e

    def _create(self, **kwargs: Any) -> Any:
        # Some newer OpenAI models reject a temperature argument, so only send it when set.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(model=self.model, **kwargs)
        except self._api_error as e:
            raise ProviderError(f"{self.name}: {_describe_api_error(e, self.client.base_url)}") from e
        except self._sdk_error as e:
            raise ProviderError(f"{self.name}: {_scrub(_one_line(str(e)))}") from e
        except ValueError as e:  # e.g. a 200 reply whose body is not valid JSON
            self.usage.add(None, None)
            raise _unexpected_reply(self.name, self.client.base_url, "the reply could not be parsed") from e
        # Some OpenAI-compatible servers leave usage out; still count the call.
        usage = getattr(resp, "usage", None)
        self.usage.add(getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
        choices = getattr(resp, "choices", None)
        if not isinstance(choices, list) or not choices or getattr(choices[0], "message", None) is None:
            raise _unexpected_reply(self.name, self.client.base_url, "not a chat completion")
        return resp

    def choose_tool(self, question: str, tools: ToolSet) -> Choice:
        resp = self._create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=to_openai_tools(tools),
            tool_choice="auto",
        )
        try:
            choice = resp.choices[0]
            calls = choice.message.tool_calls or []
            name = calls[0].function.name if calls else None
            problem = None if calls else _openai_problem(choice)
        except _REPLY_ERRORS as e:
            raise _unexpected_reply(self.name, self.client.base_url, f"{type(e).__name__}: {e}") from e
        if calls and not isinstance(name, str):
            raise _unexpected_reply(self.name, self.client.base_url, "a tool call without a name")
        return Choice(tool=name, raw=resp, error=problem)

    def complete(self, prompt: str) -> str:
        resp = self._create(messages=[{"role": "user", "content": prompt}])
        try:
            choice = resp.choices[0]
            problem = _openai_problem(choice)
            content = choice.message.content
        except _REPLY_ERRORS as e:
            raise _unexpected_reply(self.name, self.client.base_url, f"{type(e).__name__}: {e}") from e
        if problem:
            raise ProviderError(
                f"{self.name}: the reply was {problem} (finish_reason {choice.finish_reason!r}) "
                "before it finished writing questions"
            )
        if content is not None and not isinstance(content, str):
            raise _unexpected_reply(self.name, self.client.base_url, "message content is not text")
        return content or ""


def _openai_problem(choice: Any) -> str | None:
    if choice.finish_reason == "length":
        return "truncated"
    if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
        return "refused"
    return None


# Messages API stop reasons that mean "no usable answer", as Choice.error values.
_ANTHROPIC_PROBLEMS = {"max_tokens": "truncated", "refusal": "refused"}


class AnthropicProvider:
    """Messages API with tools. Reads ANTHROPIC_API_KEY from the environment.

    Current Claude models think before they answer, and thinking counts against
    ``max_tokens``, so the default leaves plenty of room (tokens are billed only when
    used). No sampling parameters are sent: the current API version rejects
    temperature, top_p and top_k.
    """

    def __init__(self, model: str, *, max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install 'toolclash[anthropic]' to use the anthropic provider"
            ) from e
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        self.usage = Usage()
        self._api_error = anthropic.APIError
        # Raised when ANTHROPIC_CONFIG_DIR / ANTHROPIC_PROFILE point at no usable login,
        # either right here or when a request is built.
        self._credentials_error = getattr(anthropic, "CredentialsError", ())
        # Any other client-side SDK error, e.g. an expired `ant auth login` that fails to refresh.
        self._sdk_error = anthropic.AnthropicError
        try:
            self.client = anthropic.Anthropic()
        except self._credentials_error as e:
            raise ProviderError(self._no_credentials(e)) from e
        except self._sdk_error as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}: {_scrub(_one_line(str(e)))}") from e

    def _no_credentials(self, e: Exception | None = None) -> str:
        found = f"no usable Anthropic credentials ({_one_line(str(e))})" if e else "no Anthropic credentials found"
        return f"{self.name}: {found}; set ANTHROPIC_API_KEY and try again"

    def _create(self, **kwargs: Any) -> Any:
        try:
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, **kwargs)
        except self._api_error as e:
            raise ProviderError(f"{self.name}: {_describe_api_error(e, self.client.base_url)}") from e
        except self._credentials_error as e:
            raise ProviderError(self._no_credentials(e)) from e
        except self._sdk_error as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}: {_scrub(_one_line(str(e)))}") from e
        except TypeError as e:
            # With nothing configured at all, the SDK only notices when it builds the request.
            if "authentication" not in str(e):
                raise
            raise ProviderError(self._no_credentials()) from e
        except ValueError as e:  # the SDK's own checks before sending, e.g. "streaming is required"
            raise ProviderError(f"{self.name}: {_one_line(str(e))}") from e
        usage = getattr(resp, "usage", None)
        self.usage.add(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
        if not isinstance(getattr(resp, "content", None), list) or usage is None:
            raise _unexpected_reply(self.name, self.client.base_url, "not a Messages API response")
        return resp

    def choose_tool(self, question: str, tools: ToolSet) -> Choice:
        resp = self._create(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
            tools=to_anthropic_tools(tools),
        )
        try:
            names = [b.name for b in resp.content if b.type == "tool_use"]
            problem = _ANTHROPIC_PROBLEMS.get(resp.stop_reason)
        except _REPLY_ERRORS as e:
            raise _unexpected_reply(self.name, self.client.base_url, f"{type(e).__name__}: {e}") from e
        if names:
            if not isinstance(names[0], str):
                raise _unexpected_reply(self.name, self.client.base_url, "a tool call without a name")
            return Choice(tool=names[0], raw=resp)
        return Choice(tool=None, raw=resp, error=problem)

    def complete(self, prompt: str) -> str:
        resp = self._create(messages=[{"role": "user", "content": prompt}])
        try:
            problem = _ANTHROPIC_PROBLEMS.get(resp.stop_reason)
            texts = [b.text for b in resp.content if b.type == "text"]
        except _REPLY_ERRORS as e:
            raise _unexpected_reply(self.name, self.client.base_url, f"{type(e).__name__}: {e}") from e
        if problem:
            raise ProviderError(
                f"{self.name}: the reply was {problem} (stop_reason {resp.stop_reason!r}) "
                "before it finished writing questions"
            )
        if not all(isinstance(t, str) for t in texts):
            raise _unexpected_reply(self.name, self.client.base_url, "a text block without text")
        return "".join(texts)


# --- fake provider ----------------------------------------------------------------------


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    """Lower-case words, underscores split, plurals crudely stripped ("docs" -> "doc")."""
    words = _WORD_RE.findall(text.lower().replace("_", " "))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def _tool_words(tool: Tool) -> set[str]:
    return _words(tool.name) | {w for p in tool.parameter_names for w in _words(p)}


@dataclass
class FakeProvider:
    """A stand-in model that needs no API key.

    ``choose_tool`` picks the tool whose name and parameter words overlap most with
    the question. Ties are broken at random, which is the fake's version of
    "confusion". ``flip_rate`` adds extra random wrong picks so reports look
    realistic in demos.

    ``complete`` only understands the prompts that ``questions.py`` builds: it reads
    the JSON inside ``<request>...</request>`` and fills in templates.
    """

    flip_rate: float = 0.0
    seed: int = 0
    name: str = "fake"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_tool(self, question: str, tools: ToolSet) -> Choice:
        self.usage.add(len(question) // 4, 8)  # rough token estimate, so cost math has data
        q = _words(question)
        scores = {t.name: len(q & _tool_words(t)) for t in tools.tools}
        best = max(scores.values(), default=0)
        if best == 0:
            return Choice(tool=None)
        pick = self.rng.choice(sorted(n for n, s in scores.items() if s == best))
        if self.rng.random() < self.flip_rate:
            pick = self.rng.choice(tools.names())
        return Choice(tool=pick)

    def complete(self, prompt: str) -> str:
        self.usage.add(len(prompt) // 4, 64)
        match = re.search(r"<request>\s*(\{.*\})\s*</request>", prompt, re.DOTALL)
        if not match:
            return "[]"
        req = json.loads(match.group(1))
        task, n = req["task"], int(req["n"])
        if task == "direct":
            lines = _fake_direct(req["tool"], n)
        elif task == "ambiguous":
            lines = _fake_ambiguous(req["target"], req["distractor"], n)
        else:
            lines = _FAKE_NO_TOOL[:n]
        return json.dumps(lines)


def _verb(tool: dict[str, Any]) -> str:
    return tool["name"].replace("_", " ")


def _args(tool: dict[str, Any], i: int) -> str:
    props = list((tool.get("parameters") or {}).get("properties", {}))
    required = (tool.get("parameters") or {}).get("required") or props[:1]
    return ", ".join(f"{p} is value{i}" for p in required) or "no arguments"


_DIRECT_TEMPLATES = [
    "Please {verb}: {args}.",
    "I need to {verb}; {args}.",
    "Could you {verb}? {args}.",
    "Go ahead and {verb}, where {args}.",
    "Help me {verb} — {args}.",
    "{verb} for me, {args}.",
]


def _fake_direct(tool: dict[str, Any], n: int) -> list[str]:
    return [
        _DIRECT_TEMPLATES[i % len(_DIRECT_TEMPLATES)].format(verb=_verb(tool), args=_args(tool, i))
        for i in range(n)
    ]


def _fake_ambiguous(target: dict[str, Any], distractor: dict[str, Any], n: int) -> list[str]:
    return [
        f"I want to {_verb(target)} (something like {_verb(distractor)}), {_args(target, i)}."
        for i in range(n)
    ]


_FAKE_NO_TOOL = [
    "What is the capital of France?",
    "How are you doing today?",
    "Explain why the sky is blue.",
    "Give me a one-line summary of the French Revolution.",
    "What is 17 times 23?",
    "Tell me a fun fact about octopuses.",
    "Translate 'good morning' into Spanish.",
    "Who wrote Pride and Prejudice?",
]


# --- factory -----------------------------------------------------------------------------


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"  # what an unset OLLAMA_HOST means

PROVIDER_KINDS = ("fake", "openai", "anthropic", "gemini", "ollama")


def get_provider(spec: str) -> Provider:
    """Build a provider from a spec string.

    * ``fake`` or ``fake:0.2`` (random flip rate) — no network, no key.
    * ``openai:<model>`` — needs OPENAI_API_KEY.
    * ``anthropic:<model>`` — needs ANTHROPIC_API_KEY.
    * ``gemini:<model>`` — needs GEMINI_API_KEY; uses Google's OpenAI-compatible endpoint.
    * ``ollama:<model>`` — local Ollama at localhost:11434 (override with OLLAMA_HOST).
    """
    kind, _, rest = spec.partition(":")
    if kind == "fake":
        return FakeProvider(flip_rate=float(rest) if rest else 0.0)
    if kind not in PROVIDER_KINDS:
        raise ValueError(f"unknown provider {kind!r}; expected one of {', '.join(PROVIDER_KINDS)}")
    if not rest:
        raise ValueError(f"provider spec {spec!r} needs a model, e.g. '{kind}:<model-name>'")
    if kind == "openai":
        return OpenAIProvider(rest)
    if kind == "anthropic":
        return AnthropicProvider(rest)
    if kind == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        return OpenAIProvider(rest, base_url=GEMINI_BASE_URL, api_key=key, label="gemini")
    return OpenAIProvider(
        rest, base_url=ollama_base_url(os.environ.get("OLLAMA_HOST")), api_key="ollama", label="ollama"
    )


def ollama_base_url(value: str | None) -> str:
    """The OpenAI-compatible endpoint for an OLLAMA_HOST value, read the way Ollama reads it.

    Like Ollama's own client: quotes and spaces are ignored; without a scheme the
    scheme is http and the port defaults to 11434 (with http:// or https:// it is 80
    or 443); a bare IP, including IPv6, is fine; an empty value means 127.0.0.1:11434.
    A listen-on-everything address (0.0.0.0, ::) is turned into loopback, since it
    cannot be connected to on Windows.
    """
    value = (value or "").strip().strip("\"'")
    scheme, sep, hostport = value.partition("://")
    default_port = "11434"
    if not sep:
        scheme, hostport = "http", value
    elif scheme == "http":
        default_port = "80"
    elif scheme == "https":
        default_port = "443"
    hostport, _, path = hostport.partition("/")
    userinfo, at, hostport = hostport.rpartition("@")  # a proxy login: keep it, parse the rest
    login = f"{userinfo}@" if at else ""
    host, port = _split_host_port(hostport)
    if host is None:  # no port given
        host, port = hostport.strip("[]") or "127.0.0.1", default_port
    if not port.isdigit() or int(port) > 65535:
        port = default_port
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    if _is_ipv6(host):
        host = f"[{host}]"
    path = path.strip("/")
    base = f"{scheme}://{login}{host}:{port}" + (f"/{path}" if path else "")
    return base if base.endswith("/v1") else base + "/v1"


def _split_host_port(hostport: str) -> tuple[str | None, str]:
    """Go's net.SplitHostPort: "host:port" or "[v6]:port"; (None, "") when there is no port."""
    if hostport.startswith("["):
        host, sep, rest = hostport[1:].partition("]")
        if sep and rest.startswith(":"):
            return host, rest[1:]
        return None, ""
    if hostport.count(":") == 1:
        host, _, port = hostport.partition(":")
        return host, port
    return None, ""  # no colon, or a bare IPv6 address


def _is_ipv6(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).version == 6
    except ValueError:
        return False
