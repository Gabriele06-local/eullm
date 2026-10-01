# Reflex for MCP clients: `eullm-reflex-mcp`

An MCP server that gives an agent — Claude Code, Claude Desktop, Cursor, an
IDE assistant — Reflex, EuLLM's decision primitive: a small local model that
reads text and answers enumerated questions with probabilities read from the
model. Nothing is generated, nothing leaves your machine unless you point it
at an EuLLM elsewhere, and every decision is in EuLLM's audit trail.

| Tool | What it decides |
|---|---|
| `select_tools` | which tools of a catalog a request needs, or none of them |
| `rag_gate` | whether retrieved passages suffice to answer a question: answer, retrieve more, or abstain |
| `decide` | typed questions about a state — yes or no, one of N, a level on a scale — in the System One request shape of `POST /v1/systemone` |
| `model_info` | which decision model EuLLM has loaded, and how this server is configured |

[jev-style](https://github.com/lawrence3699/jev-style)'s own MCP server
already gives an agent the raw `decide`, `noul`, `choice`, `score` and
`model_info` against EuLLM (see
[docs/engine.md](../../docs/engine.md#jev-style-with-eullm-mcp-server-cli-python-client)).
This one adds what needs EuLLM's embeddings or the
[ReflexBench](../../bench/reflexbench/README.md) measurements — tool
selection in two stages, and the RAG gate with a calibrated threshold — and
keeps `decide` and `model_info`, so that one server is enough.

## Install

Python 3.10 or later. From a checkout of the repository:

```bash
python3 -m venv ~/.venvs/eullm-reflex-mcp
~/.venvs/eullm-reflex-mcp/bin/pip install ./adapters/reflex-mcp
~/.venvs/eullm-reflex-mcp/bin/eullm-reflex-mcp --version
```

Or let [uv](https://docs.astral.sh/uv/) build and run it, from the checkout
or straight from GitHub:

```bash
uvx --from ./adapters/reflex-mcp eullm-reflex-mcp --version
uvx --from "git+https://github.com/eullm/eullm#subdirectory=adapters/reflex-mcp" eullm-reflex-mcp --version
```

From GitHub, the first run fetches the whole repository with the engine's
llama.cpp submodule, about half a gigabyte; from a checkout you already
have, nothing but the dependencies. Those are the MCP Python SDK (`mcp`,
MIT) and `httpx2` (BSD-3-Clause).

## EuLLM, with a decision model and an embedding model

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
eullm pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \
            --embedding-model qwen3-embedding-0.6b-gguf-q8_0
```

The 2B is the model ReflexBench measured on a GPU. On a CPU the Jev-Style
0.8B (`hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M`, served as
`jev-style-0.8b-decision-v3-gguf-q4_k_m`) answers in a second or two; see
[what it costs](#what-it-costs). The embedding model is used by
`select_tools` alone; without one, it reads catalogs of up to 37 tools whole
and refuses larger ones.

## Claude Code

```bash
claude mcp add eullm-reflex --scope user \
  -e EULLM_EMBED_MODEL=qwen3-embedding-0.6b-gguf-q8_0 \
  -e 'EULLM_EMBED_QUERY_PREFIX=Instruct: Given a user request, retrieve the tools needed to handle it\nQuery:' \
  -- uvx --from "git+https://github.com/eullm/eullm#subdirectory=adapters/reflex-mcp" eullm-reflex-mcp
```

With the package installed in a virtual environment, the command after `--`
is the script itself: `~/.venvs/eullm-reflex-mcp/bin/eullm-reflex-mcp`.
`claude mcp list` shows whether it connects; in a session, `/mcp` lists its
four tools.

Over streamable HTTP instead, one server for several clients, with its
settings in the environment it is started in:

```bash
EULLM_EMBED_MODEL=qwen3-embedding-0.6b-gguf-q8_0 \
EULLM_EMBED_QUERY_PREFIX='Instruct: Given a user request, retrieve the tools needed to handle it\nQuery:' \
  eullm-reflex-mcp --transport streamable-http        # http://127.0.0.1:11436/mcp
claude mcp add --transport http eullm-reflex http://127.0.0.1:11436/mcp
```

## Claude Desktop, Cursor and other `mcpServers` clients

```json
{
  "mcpServers": {
    "eullm-reflex": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/eullm/eullm#subdirectory=adapters/reflex-mcp",
        "eullm-reflex-mcp"
      ],
      "env": {
        "EULLM_URL": "http://localhost:11434",
        "EULLM_EMBED_MODEL": "qwen3-embedding-0.6b-gguf-q8_0",
        "EULLM_EMBED_QUERY_PREFIX": "Instruct: Given a user request, retrieve the tools needed to handle it\nQuery:"
      }
    }
  }
}
```

Claude Desktop reads it from `claude_desktop_config.json` (Settings →
Developer → Edit Config), Cursor from `~/.cursor/mcp.json` or a project's
`.cursor/mcp.json`. A desktop application does not always see your shell's
`PATH`: if it cannot find `uvx`, give its full path (`which uvx`).

## Configuration

| Variable | Default | |
|---|---|---|
| `EULLM_URL` | `http://localhost:11434` | the EuLLM server |
| `EULLM_API_KEY` | none | sent as a bearer token, when EuLLM requires keys (`EULLM_API_KEYS`) |
| `EULLM_EMBED_MODEL` | none | the embedding model `select_tools` shortlists with |
| `EULLM_EMBED_QUERY_PREFIX` | none | put before the request when it is embedded; `\n` is a newline |
| `REFLEX_GATE_THRESHOLD` | none | `rag_gate`'s threshold on P(answer), [calibrated](#calibrating-the-gate) on your own cases |
| `EULLM_TIMEOUT` | 300 | seconds to wait for EuLLM's answer |

An empty variable counts as unset.

| Option | |
|---|---|
| `--transport stdio` | the default: the client starts the server and talks to it over its standard input and output |
| `--transport streamable-http` | an HTTP server on `127.0.0.1`, at `/mcp` |
| `--port` | its port, 11436 by default |
| `--allow-remote` | accept an `EULLM_URL` that is not on this machine |

**A local EuLLM unless you say otherwise.** The requests carry the agent's
text, so an `EULLM_URL` that is not `localhost` or a loopback address is
refused at start unless the server is started with `--allow-remote`. The
HTTP transport listens on `127.0.0.1` only: the server holds
`EULLM_API_KEY` and has no authentication of its own.

Qwen3-Embedding was trained with an instruction before each query and none
before the documents; the prefix above is the one ReflexBench measured.
Another embedding model wants its own instruction, or none.

## The tools

The examples are what an agent sends, and what the Jev-Style 0.8B answered
on a 4-core CPU.

### `select_tools(request, tools, shortlist=20, allow_none=true)`

`tools` is the catalog, `[{"name": ..., "description": ...}]`. When it has
more than `shortlist` tools, EuLLM's embeddings keep the `shortlist` closest
to the request. The decision model then reads the request with those tools'
descriptions and a "none" option, and gives each a probability; with "none"
they sum to 1.

```json
{"request": "Will it rain in Rome tomorrow? Do I need an umbrella?",
 "tools": [{"name": "get_weather", "description": "Current weather and the forecast for a city"},
           {"name": "send_email", "description": "Send an email message to a recipient"},
           "... six more ..."],
 "shortlist": 4}
```

```json
{"tools": [{"name": "get_weather", "probability": 0.9732},
           {"name": "search_flights", "probability": 0.0029},
           {"name": "stock_price", "probability": 0.0028},
           {"name": "book_meeting", "probability": 0.0026}],
 "none_probability": 0.0184, "none_wins": false,
 "method": "two-stage", "catalog_size": 8, "left_out": 4, "questions": 1,
 "decision_model": "Jev-Style-0.8B-Decision-v3-Q4_K_M",
 "embedding_model": "qwen3-embedding-0.6b-gguf-q8_0",
 "evaluated_tokens": 172, "ms": 4804.5, "note": null}
```

Why it works this way — ReflexBench, Jev-Style 2B on an RTX 5070 Ti
([results](../../docs/reflex-roadmap.md#first-results--rtx-5070-ti-jev-style-2b-500-requests-a-set)):

- **The embeddings choose the shortlist.** For one tool out of MetaTool's
  199, they ranked the right one first 75.4% of the time in 51 ms; the 2B
  reading the whole catalog, 73.4% in 439 ms, and 76 s a decision on a CPU.
  The 2B ranking the embeddings' 20 had the right tool among the first 10
  for 95.2% of the requests, reading 613 tokens.
- **The model judges a few options.** On BFCL's real requests, 2 to 37
  functions each, the 2B ranked the right one first 93–95% of the time
  against the embeddings' 91%. Without an embedding model `select_tools`
  reads catalogs of up to 37 tools whole, and refuses larger ones instead of
  reading 199 descriptions on every request.
- **"None" is a decision the embeddings cannot make** without a calibrated
  threshold. On BFCL the 2B said it for 69% of the requests no function fit,
  and wrongly for 2% of those one did; over MetaTool's 199 generic plugins it
  said it wrongly for 14–20%. Trust it on a few well-described tools.
- **Several tools.** For requests that need two tools of 199, the two stages
  had both among the first 3 for half of the requests and among the first 10
  for 81%: look further than the first.

Long descriptions are not cut. A question longer than the model's input
budget is refused whole by EuLLM — the 0.8B allows 2,048 tokens for a
question with its options, which it reads twice — and the tools are split
over twice as many questions until they fit. "None" is offered in each, and
weighed against the best tool in that tool's question. Twenty tools with
59-word descriptions went to the 0.8B as two questions: 3,081 tokens, 21 s
on a 4-core CPU. If the model cannot read the shortlist even one tool a
question, the embeddings' order comes back with `"method": "embeddings"` and
EuLLM's message in `note`.

Each tool is embedded once, as `name: description`, and kept (up to 2,048
vectors): an agent sends the same catalog with every request, and on a CPU
embedding it again would cost seconds every time.

### `rag_gate(question, passages, threshold=null)`

`answer` — the passages hold every fact the answer needs; `retrieve_more` —
some are there, at least one is missing; `abstain` — nothing there helps.
The question and the passages are put to the model exactly as
[`ragbench.py`](../../bench/reflexbench/README.md#the-rag-gate-ragbenchpy)
puts them, so that a threshold fitted there holds here; a test checks the
two have not drifted apart.

```json
{"question": "Who wrote the novel The Name of the Rose?",
 "passages": ["The Name of the Rose: The Name of the Rose is the 1980 debut novel by Italian author Umberto Eco. ...",
              "Bologna: Bologna is the capital of the Emilia-Romagna region in northern Italy."]}
```

```json
{"decision": "answer", "decided_by": "model_choice",
 "threshold": null, "threshold_source": null, "model_choice": "answer",
 "probabilities": {"answer": 0.9664, "retrieve_more": 0.0213, "abstain": 0.0123},
 "decision_model": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "evaluated_tokens": 235, "ms": 1378.3,
 "note": "decided by the model's own choice, with no calibrated threshold: ..."}
```

With a threshold — the `threshold` argument, else `REFLEX_GATE_THRESHOLD` —
the decision is `answer` when P(answer) is at or above it, and otherwise the
likelier of `retrieve_more` and `abstain`; `decided_by` says `threshold` and
`threshold_source` where it came from. Without one it is the model's own
choice, and the result says so. Measured on MuSiQue
([results](../../docs/reflex-roadmap.md#rag-gate-first-results--rtx-5070-ti-jev-style-2b)):

- calibrated, the 2B stopped about half again as many insufficient contexts
  as the embeddings' best similarity, at the same cost in good answers;
- left to its own choice it was far too cautious: it stopped 90% of the
  insufficient contexts, and 65% of those that did suffice;
- telling `retrieve_more` from `abstain` is the weak part: three-way, it was
  right 46% of the time.

### `decide(state, questions)`

The System One request shape of `POST /v1/systemone`, passed through, and
EuLLM's response unchanged (shortened below): see
[docs/engine.md](../../docs/engine.md#decisions-v1systemone-and-the-decision-slot)
for the question types, the limits and the response. The state is read once
for all the questions.

```json
{"state": "Help! My payouts have been failing for 3 days.",
 "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
               "team": {"type": "choice", "instructions": "Which team should handle it?",
                        "criteria": {"billing": "Payments and payouts", "tech": "Bugs"}}}}
```

```json
{"model": "Jev-Style-0.8B-Decision-v3-Q4_K_M",
 "answers": {"is_urgent": {"type": "noul", "noul": 0.5012, "eullm": {"...": "..."}},
             "team": {"type": "choice", "choice": "billing",
                      "probabilities": {"billing": 0.9232, "tech": 0.0768},
                      "confidence": 0.8464, "eullm": {"...": "..."}}},
 "usage": {"input_tokens": 122, "output_tokens": 0}, "timing": {"total_ms": 875.8},
 "eullm": {"mode": "shared_prefix", "temperature": 0.8801, "...": "..."}}
```

When EuLLM refuses a request, the agent reads EuLLM's own message as a tool
error:

```
Error executing tool decide: EuLLM's /v1/systemone answered 422 invalid_question:
a choice question needs 2 to 255 options, got 1 (question 'q')
```

### `model_info()`

```json
{"eullm_url": "http://127.0.0.1:11602", "eullm_version": "0.7.20",
 "decision_model": {"name": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "readout": "verdict",
                    "context_tokens": 8192, "head_max_tokens": 2048},
 "embedding_model": "qwen3-embedding-0.6b-gguf-q8_0",
 "embed_query_prefix": "Instruct: Given a user request, retrieve the tools needed to handle it\nQuery:",
 "gate_threshold": null, "max_tools_without_embeddings": 37,
 "adapter_version": "0.1.0", "note": null}
```

## Calibrating the gate

A threshold means something only for the domain it was fitted on: the
2B's MuSiQue threshold says nothing about your contracts or your tickets.
Fit one on cases of your own with
[`ragbench.py`](../../bench/reflexbench/README.md#the-rag-gate-ragbenchpy),
against the decision model the MCP server will use.

1. Write the cases, one JSON object per line: a question, the passages your
   retrieval returns for it, and what should happen. Cases of the same
   question share a `group`, and stay on the same side of the split. A
   question with its passages and the same question with the one that
   matters left out make the most useful pair.

   ```json
   {"id": "q1:a", "group": "q1", "question": "Entro quanto si ricorre al TAR?", "passages": ["Art. 29 c.p.a.: ...", "..."], "label": "answer"}
   {"id": "q1:x", "group": "q1", "question": "Entro quanto si ricorre al TAR?", "passages": ["Art. 30 c.p.a.: ...", "..."], "label": "abstain"}
   ```

   `label` is `answer`, `retrieve_more` or `abstain`. A few hundred questions:
   the MuSiQue threshold was fitted on 500.

2. Run the gate over them, with the same decision model and an audit
   directory of its own:

   ```bash
   EULLM_AUDIT_DIR=/tmp/ragbench-audit \
     eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m
   python3 bench/reflexbench/ragbench.py --sets '' --data my-cases.jsonl \
     --methods reflex-gate --out my-gate.json --details my-gate.jsonl
   ```

   The questions are split in two halves: the threshold is fitted on one
   (the best balanced accuracy there), and the table reports on the other
   what it catches and what it blocks: `fitted: caught` is the share of the
   insufficient cases it stops, `fitted: blocked` the share of the
   sufficient ones it stops anyway.

3. Read the threshold:

   ```bash
   python3 -c "import json; m = json.load(open('my-gate.json'))['results'][0]['metrics']; print(m['fitted_threshold'], m['fitted'])"
   ```

   and give it to the server as `REFLEX_GATE_THRESHOLD`: one more `-e
   REFLEX_GATE_THRESHOLD=…` in the `claude mcp add` command above (after
   `claude mcp remove eullm-reflex --scope user`, if it is already there), or
   one more variable in an `mcpServers` block's `env`. An agent may also pass
   it as `threshold` on a call. For another trade-off — at most one
   sufficient case in ten blocked, say — `my-gate.jsonl` has every case's
   P(answer) (`score`) and label to choose it from.

Fit it again when the decision model changes: the 0.8B and the 2B give
different probabilities to the same case.

## What it costs

On a GPU — ReflexBench, Jev-Style 2B on an RTX 5070 Ti: a RAG gate decision
over five passages, 49 ms at the median, about 770 tokens; two stages, the
embeddings and the 2B ranking 20 of 199 tools, 109 ms, 613 tokens; a few
tools read whole, 26–28 ms.

On a CPU — measured for this server, Jev-Style 0.8B and Qwen3-Embedding-0.6B
on a shared 4-core machine:

| Call | Time | Tokens the model read |
|---|---|---|
| `rag_gate`, one or two passages | 1.3–3.6 s | 201–235 |
| `select_tools`, 8 tools read whole | 1.7 s | 265 |
| `select_tools`, two stages keeping 4 of 8 tools, a new catalog | 2.3–4.8 s | 126–172 |
| the same, with the catalog already embedded | 1.1 s | 117 |
| `select_tools`, 20 tools of 59 words, in two questions | 21 s | 3,081 |
| `decide`, two questions | 0.9 s | 122 |

The machine was shared with other jobs, which is most of the spread.

A call the client cancels closes its request to EuLLM, which drops the
decision and records nothing; it notices at its next question, since a
question being decoded is not interrupted. The 20-tool call above,
cancelled after 3 s, kept the 0.8B busy for 26 s.

## Tests

```bash
pip install ./adapters/reflex-mcp
python3 -m unittest discover -s adapters/reflex-mcp/tests -v
```

Offline: every tool through the MCP SDK's in-memory client against a
stand-in EuLLM, the settings, the wording shared with ReflexBench, and the
command itself on stdio and on streamable HTTP. Against a running EuLLM,
end to end, through the command on stdio:

```bash
REFLEX_MCP_E2E_URL=http://localhost:11434 \
REFLEX_MCP_E2E_EMBED_MODEL=qwen3-embedding-0.6b-gguf-q8_0 \
  python3 -m unittest discover -s adapters/reflex-mcp/tests -p test_e2e.py -v
```
