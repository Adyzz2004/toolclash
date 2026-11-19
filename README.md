# toolclash

Measure which of your agent's tools the model confuses with each other.

When an agent has many tools, the model picks the wrong one more often than
end-to-end tests reveal, because the agent often recovers and still gets the
final answer right. `toolclash` tests tool *selection* directly: it loads your
tool definitions, generates questions, asks the model which tool it would call
(several times per question), and reports a per-tool confusion matrix, a
stability score, and a "mutual confusion" score for every pair of tools. It
runs as a CI gate that fails the build when adding a tool makes an existing one
worse.

Works with MCP, OpenAI and Anthropic tool formats; benches on OpenAI,
Anthropic, Gemini or local Ollama models; ships a free `fake` provider so the
whole pipeline runs without an API key.

## Install

```bash
pip install -e ".[all,dev]"      # or .[openai] / .[anthropic] / plain .
```

## Pipeline

```bash
# 1. See what toolclash sees (format auto-detected: MCP, OpenAI or Anthropic)
toolclash inspect tools.json

# 2. Generate test questions. Use a DIFFERENT model from the one you bench.
#    The generator only sees each tool's name and argument schema, never its description.
toolclash generate tools.json -o questions.json --provider anthropic:<model>

# 3. Ask the benched model which tool it would call, 5 times per question
toolclash run tools.json questions.json -o run.json --provider openai:<model> -n 5

# 4. Confusion matrix, per-tool accuracy / stability / flip rate, most confused pairs
#    (pass the provider's per-million-token prices for a cost-per-run estimate)
toolclash report run.json --json summary.json --price-in 3 --price-out 15

# 5. CI gate: did adding a tool make an existing one worse?
toolclash run tools_v2.json questions.json -o run_v2.json --provider openai:<model>
toolclash compare run.json run_v2.json --max-accuracy-drop 0.10   # exit 1 on regression
```

No API key? Use `--provider fake` (or `fake:0.15` for a 15% random flip rate so
the report has something to show).

| Provider spec | Needs |
|---|---|
| `openai:<model>` | `OPENAI_API_KEY` |
| `anthropic:<model>` | `ANTHROPIC_API_KEY` |
| `gemini:<model>` | `GEMINI_API_KEY` (uses Google's OpenAI-compatible endpoint) |
| `ollama:<model>` | Ollama running locally; `OLLAMA_HOST` to point elsewhere |
| `fake` / `fake:<flip-rate>` | nothing |

## What the numbers mean

| Number | Meaning |
|---|---|
| **accuracy** | share of samples where the right tool was picked |
| **stability** | average agreement with the majority pick per question; 100% = every repeat gave the same answer |
| **flip rate** | share of a tool's questions where the repeats disagreed — "correct but shaky" |
| **mutual confusion (A, B)** | (A→B + B→A) / all picks for A and B — the pairs whose descriptions need work |
| **over-trigger rate** | share of no-tool questions where the model called a tool anyway |

## Anti-cheating rules built in

- Questions are generated from **name + schema only**; descriptions are hidden from the generator.
- `run` warns if the generator and the benched model are the same.
- Ambiguous questions target the **most similar tool pairs** ("route to A, easy to mistake for B").
- **No-tool** questions measure over-triggering.
- Every question is hashed into a `dev` or `heldout` split; tune on `dev`, report on `heldout`.

## Example output (fake provider, 4 tools, 15% flip rate)

```
overall accuracy 86%   over-trigger rate 0%

expected \ picked    search_docs  get_doc  send_slack_message  create_ticket  (none)
search_docs                   35        1                   1              3       ·
get_doc                        4       54                  11              1       ·
send_slack_message             2        1                  36              1       ·
create_ticket                  1        1                   ·             38       ·
(none)                         ·        ·                   ·              ·      25

FAIL adding send_slack_message made get_doc 33 points worse on accuracy (100% → 67%)
```

## Status

Core pipeline complete: loaders, generator, runner, report, CI gate, cost
estimate, five providers. Next: run on real public MCP servers and publish the
numbers; concurrency for large runs; accuracy-vs-tool-count chart.

## Development

```bash
pytest -q
ruff check .
```
