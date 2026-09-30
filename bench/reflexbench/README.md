# ReflexBench — does a decision pick the tools a request needs?

MVP 0 of the [Reflex roadmap](../../docs/reflex-roadmap.md). Before Reflex
selects tools for anyone, this benchmark measures it on public labelled sets
against simpler ways of doing the same thing, and answers three questions:

1. **How many tools can be dropped without losing the right one?**
   recall@k, and the k that keeps every needed tool for 95% and 99% of the
   requests (k95, k99).
2. **How much of the tool-calling model's prompt does that save?** The share
   of the tool specs still sent when only the first k95 tools are kept.
3. **What does the decision cost?** Latency p50 and p95 as the client sees
   it, and for Reflex the tokens the decision model read.

Only the Python standard library is needed.

## Running it

```bash
# 1. A server with a Jev-Style decision model. Its audit trail goes to a
#    directory of its own: a run is thousands of decisions.
eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
EULLM_AUDIT_DIR=/tmp/reflexbench-audit \
  eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m --decision-ctx 25600

# 2. The benchmark, from another terminal
python3 bench/reflexbench/reflexbench.py --limit 200 \
  --out rb-2b.json --details rb-2b.jsonl
```

It prints a Markdown table and writes the full report as JSON (`--out`);
`--details` keeps every ranking, one JSON line each. The sets are downloaded
on first use to `~/.cache/reflexbench` (`$REFLEXBENCH_CACHE`), never into
the repository.

For the `embed` baseline, load an embedding model in the same server and
name it with `--embed-model`; without it the method is skipped.
Qwen3-Embedding-0.6B (Apache-2.0) is a strong small one, and wants an
instruction before the request, never before the tools:

```bash
eullm pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
# serve as above, adding --embedding-model qwen3-embedding-0.6b-gguf-q8_0
python3 bench/reflexbench/reflexbench.py --limit 200 \
  --embed-model qwen3-embedding-0.6b-gguf-q8_0 \
  --embed-query-prefix 'Instruct: Given a user request, retrieve the tools needed to handle it\nQuery:'
```

**On a CPU, lower `--limit`.** In layout A a MetaTool request reads the whole
catalog of 199 tools, about 7,000 tokens: some two minutes a decision for the
2B on a 4-core machine. `--limit 20` gives a first cost figure in under two
hours.

| Option | Default | |
|---|---|---|
| `--url` | `http://localhost:11434` | the EuLLM server |
| `--model` | the one loaded | decision model to ask for |
| `--embed-model` | none | embedding model for `embed` |
| `--embed-query-prefix` | none | text before each request for `embed`; `\n` is a newline |
| `--api-key` | `$EULLM_API_KEY` | when the server requires one |
| `--sets` | `metatool-single,metatool-multi,bfcl-live-multiple,bfcl-live-irrelevance` | public sets, below |
| `--data` | none | a set of your own, repeatable |
| `--methods` | `bm25,embed,reflex-a,reflex-b` | methods, below |
| `--limit` | 100 | requests per set, 0 for all |
| `--catalog-size` | 0 (all) | offer each request at most this many tools, its own among them |
| `--abstain` | `auto` | offer Reflex a "no tool" option: `on`, `off`, or `auto`, on sets with requests no tool fits |
| `--seed` | 1 | the order requests are drawn in |

## The sets

| Set | Source | Licence | What |
|---|---|---|---|
| `metatool-single` | [MetaTool](https://github.com/HowieHwong/MetaTool) | MIT | one tool of 199 needed |
| `metatool-multi` | MetaTool | MIT | two tools of 199 needed |
| `bfcl-multiple` | [BFCL v3](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Apache-2.0 | one function of 2–4 fits |
| `bfcl-live-multiple` | BFCL v3 | Apache-2.0 | one function of 2–37 fits, from real users |
| `bfcl-irrelevance` | BFCL v3 | Apache-2.0 | no function offered fits |
| `bfcl-live-irrelevance` | BFCL v3 | Apache-2.0 | no function offered fits, from real users |

MetaTool offers every request the same catalog, so it is the set that shows
what reusing the decision state is worth; BFCL offers each request a few
functions of its own. A tool's *spec* is what a tool-calling model receives
for it: the JSON schema for BFCL, the name and description for MetaTool,
which has no schemas.

## The methods

| Method | How it ranks the tools |
|---|---|
| `bm25` | keyword matching over names and descriptions, no model |
| `embed` | cosine similarity of EuLLM embeddings; the tools are embedded once, so a request costs one embedding |
| `reflex-a` | `/v1/systemone`: the request is the state, the tools with their descriptions are the options of a `choice` question |
| `reflex-b` | the catalog is the state, the request goes in the question, the options are the tool names alone. The same catalog on every request is read once and reused |

Reflex ranks by each option's verdict score. A question has at most 255
options, and the Jev-Style 0.8B reads a question with its options in 2,048
tokens; when the server refuses a question as too long, the options are
split over several questions of one request, and the size that fits carries
over to the next requests.

## Reading the report

| Column | |
|---|---|
| R@k | share of the requests with every tool they need among the first k |
| k95, k99 | the smallest k that keeps 95% and 99% of the requests whole |
| specs kept @k95 | share of the tool-spec characters still sent at k95 — the rest is prompt saved |
| abstain ok | on requests no tool fits, how often "no tool" beat every tool (Reflex only) |
| false abstain | on requests that need a tool, how often "no tool" won anyway |
| p50 ms, p95 ms | decision latency as the client sees it; each run starts with one untimed request, so a cold start is not counted |
| tokens/decision | tokens the decision model read (`eullm.evaluated_tokens`): with a reused state, the catalog is not among them |

The JSON report adds MRR, R@20, the k99 share, and an estimate of the spec
tokens at four characters a token; an exact count needs the tokenizer of
the model that will receive the tools.

## A set of your own

One JSON object per line; `needed` may be empty, for a request no tool fits:

```json
{"id": "q1", "request": "Invoice ACME for September", "needed": ["create_invoice"], "tools": [{"name": "create_invoice", "description": "Create an invoice for a customer"}, {"name": "send_email", "description": "Send an email"}]}
```

`spec`, optional in each tool, is what the tool-calling model would receive
(default: the tool's JSON as written). Requests with the same tools in the
same order share one catalog.

## Tests

```bash
python3 -m unittest discover -s bench/reflexbench
```

Offline: the metrics, BM25, the loaders on made-up files, and the Reflex
client against a stand-in server.
