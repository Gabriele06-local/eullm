# ReflexBench — does a decision pick the tools a request needs?

Two benchmarks share this directory: `reflexbench.py`, tool selection (MVP
0), and `ragbench.py`, whether retrieved passages suffice to answer
([below](#the-rag-gate-ragbenchpy), MVP 1). Beside them, `qualify.py` says
whether one decision model may replace another
([below](#the-qualification-test-qualifypy), MVP 4).

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

It says which decision model answers before counting its decisions — check
it is the one you meant — then prints a Markdown table and writes the full
report as JSON (`--out`);
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
| `--methods` | `bm25,embed,reflex-a,reflex-b,two-stage` | methods, below |
| `--shortlist` | 20 | tools the embeddings keep for Reflex in `two-stage` |
| `--limit` | 100 | requests per set, 0 for all |
| `--catalog-size` | 0 (all) | offer each request at most this many tools, its own among them |
| `--abstain` | `on` | offer Reflex a "no tool" option on every set: it should win on requests no tool fits and lose on the others; `off` for rankings alone |
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
| `two-stage` | the embeddings keep `--shortlist` tools, `reflex-a` ranks them, the others follow in the embeddings' order. Its cost is both stages |

Reflex ranks by each option's verdict score. A question has at most 255
options, and the Jev-Style 0.8B reads a question with its options in 2,048
tokens: each request starts with its tools in one question, and when the
server refuses it as too long they are split over several questions of the
same request. "No tool", offered in each of them, is weighed against the
best tool in that tool's own question. A request the model cannot read even
beside a single tool — with the 0.8B, a request of some 2,000 tokens inside
a layout-B question — is refused, as the server refuses it rather than
truncate it: it ranks nothing and counts as keeping every tool.

## Reading the report

| Column | |
|---|---|
| refused | requests the decision model could not take; they count as keeping every tool |
| R@k | share of the requests with every tool they need among the first k |
| k95, k99 | the smallest k that keeps 95% and 99% of the requests whole |
| specs kept @k95 | share of the tool-spec characters still sent at k95 — the rest is prompt saved |
| abstain ok | on requests no tool fits, how often "no tool" beat every tool (Reflex only: the others cannot say it without a threshold, and a threshold needs calibrating) |
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

## The RAG gate: `ragbench.py`

MVP 1 of the roadmap. A RAG system retrieves passages and hands them to a
large model, which answers whether or not the facts it needs are there. A
gate between the two decides: `answer`, `retrieve_more` (some facts are
there, one is missing), or `abstain` (nothing there helps).

```bash
EULLM_AUDIT_DIR=/tmp/ragbench-audit \
  eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m --decision-ctx 25600 \
              --embedding-model qwen3-embedding-0.6b-gguf-q8_0
python3 bench/reflexbench/ragbench.py --limit 1000 \
  --embed-model qwen3-embedding-0.6b-gguf-q8_0 \
  --embed-query-prefix 'Instruct: Given a question, retrieve passages that answer it\nQuery:' \
  --out rag-2b.json --details rag-2b.jsonl
```

**The set.** [MuSiQue](https://github.com/StonyBrookNLP/musique) (CC BY 4.0)
gives each question 20 Wikipedia paragraphs: the 2 to 4 it needs, marked,
and others retrieved for being close to it. Each question makes three cases
of `--passages` passages (5): every paragraph it needs filled up with others
(`answer`), all but one (`retrieve_more`), none (`abstain`). The three differ
in what they hold, not in how the question reads. Read from a copy of the
v1.0 release on Hugging Face at a fixed revision. `--data` takes a set of
your own, one JSON object per line:

```json
{"id": "q1:a", "group": "q1", "question": "Entro quanto si ricorre al TAR?", "passages": ["Art. 29 c.p.a.: ...", "..."], "label": "answer"}
```

`document`, optional, names the text a question was written from, when
several questions share one — `rg_openbook.py` writes the article, as
`codice_civile/2043`; MuSiQue's cases name the questions that rest on the
same supporting paragraph, since MuSiQue builds many questions on one
single-hop question.

**An Italian set.** `rg_openbook.py` writes one from Forge's open-book
pairs: for each question asked by topic about an article of Italian law,
the passages retrieval finds with that article among them (`answer`), and
what it returns once the article is left out (`abstain`) — the same two
contexts Forge trains the legal model on. It needs the pairs and the
legislation records Forge prepares, and writes the set where they are:

```bash
python3 bench/reflexbench/rg_openbook.py $WORK/eullm_runs/stage3/openbook-v04.jsonl \
  --norms $WORK/norms/legislazione_*.chunks.jsonl --out rag-legal-it.jsonl
python3 bench/reflexbench/ragbench.py --sets '' --data rag-legal-it.jsonl ...
```

The pairs are written by a large model on the cluster. Without them,
`--by-heading` asks by an article's rubrica — "Che cosa prevede la legge in
materia di risarcimento per fatto illecito?" — for the articles whose rubrica
is theirs alone and names a topic, from the legislation records only:
plainer questions, the same two contexts.

```bash
python3 bench/reflexbench/rg_openbook.py --by-heading --limit 1000 \
  --norms ~/work/corpus/legislazione_*.chunks.jsonl --out rag-legal-it.jsonl
```

**The methods.**

| Method | Score, and decision |
|---|---|
| `embed-max` | the best cosine similarity between the question and a passage, the signal a RAG system has after retrieval; no decision of its own |
| `reflex-gate` | `/v1/systemone`, the question and passages as the state, one `choice` among the three; score P(`answer`) |
| `reflex-yesno` | the same state, one `noul`: do the passages hold every fact the answer needs? Score P(yes), a yes above one half lets the model answer |

**The report.** The questions are split in two halves: a threshold on each
score is fitted on the dev half (the best balanced accuracy there) and every
method is scored on the test half, the same cases for all.

| Column | |
|---|---|
| AUROC | how well the score separates sufficient passages from the rest, before any threshold |
| AUROC within a question | the same among the three cases of one question, averaged: whether the score follows the passages, whatever the question's difficulty does to its level |
| caught | of the cases whose passages do not suffice, the share the gate stops: answers not made up from missing facts |
| blocked | of the cases whose passages suffice, the share it stops anyway: answers lost for nothing |
| own | at the method's own decision, with no labelled data |
| fitted | at the threshold fitted on the dev half, what calibrating on a domain's own cases buys |
| 3-way accuracy, macro-F1 | the decision among the three: Reflex's own choice, the embeddings' two fitted thresholds |
| ECE | how far Reflex's probability of `answer` is from how often it is right |

**A gate model of your own.** The sets are labelled, so they train one:
`eullm-forge decisions import-rag` writes their cases as decision traces —
the state and the questions built by `rg_methods.request`, the body the
Reflex methods post, so the model is trained on the prompt the gate sends —
holding out whole questions, and every question about one `document`, for
dev and test. The test side's cases go to `rag-test/<set>.jsonl` beside the
traces, in this format, for `--data`: the trained model and the Jev-Style
2B are compared here on cases the model was not trained on. See
[docs/forge.md](../../docs/forge.md#import-rag-the-rag-gates-labelled-cases).

## The qualification test: `qualify.py`

MVP 4 of the roadmap. A decision model replaces another because it passed
this test on the domain's own labelled decisions, not because a
configuration line names it. The candidate — a model trained with
`eullm-forge decisions` ([docs/forge.md](../../docs/forge.md#decision-models-trained-on-your-decisions)),
say — and, optionally, the model it would replace answer the same labelled
requests, each request in the three evaluation modes of `/v1/systemone`
(`eullm.mode`), and the test prints PASS or FAIL.

```bash
# Each model on a server of its own, each with an audit directory of its
# own: a run is thousands of decisions.
EULLM_AUDIT_DIR=/tmp/qualify-candidate \
  eullm serve --port 11500 --decision-model ~/models/decide-q8_0.gguf
EULLM_AUDIT_DIR=/tmp/qualify-current \
  eullm serve --port 11501 --decision-model jev-style-2b-decision-v3-gguf-q4_k_m

python3 bench/reflexbench/qualify.py \
  --candidate http://localhost:11500 --current http://localhost:11501 \
  --data ~/decisions/data/test.labelled.jsonl --out qualify.json
```

The exit status is 0 for PASS, 1 for FAIL, 2 when a server could not be
asked at all (none listening, no decision model loaded, a wrong key). One
server with both models loaded by name (`--candidate-model`,
`--current-model`) works too: every request of the candidate is asked
before the first of the current model, so the decision slot swaps once.

**The labelled requests.** One JSON object per line: a request as
`/v1/systemone` takes it, and the right answer to each question — an
option's name for a `choice`, a level's number (from 0) for a `score`,
true or false for a `noul`. `sources`, optional, says where each answer
came from:

```json
{"id": "t1", "state": "Payouts failing for 3 days", "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Bugs"}}, "is_urgent": {"type": "noul", "instructions": "Is it urgent?"}}, "answers": {"team": "billing", "is_urgent": true}, "sources": {"team": "feedback:user", "is_urgent": "rules"}}
```

A question with no answer is not asked; a line in
`bench/decision_calibration.py`'s format (`state`, `question`, `label`) is
a request of one question. `eullm-forge decisions build` writes the states
it held out in this format (`test.labelled.jsonl`), so a model is qualified
on states it was not trained on. `--traces DIR` reads a directory a server
wrote with `EULLM_DECISION_TRACES` instead: every decision with feedback is
a request, with the questions it asked and the answers the feedback gave.
`--sources feedback` keeps only the answers people (or rules) gave after
the fact, leaving out those a teacher model labelled — against a teacher's
labels, accuracy is agreement with the teacher.

**What it measures**, per question type and over all of them:

| Column | |
|---|---|
| answers, refused | labelled answers, and those the server would not give (a `choice` of 30 options for a model that reads 26; a state over its context): a refused answer counts as wrong |
| accuracy, 95% CI | right answers in `--serve-mode` (default `shared_prefix`, the engine's), with the Wilson interval |
| commonest | the accuracy of giving each question its commonest right answer every time: what a model has to beat to have learnt anything |
| ECE, NLL | expected calibration error of the top answer (15 equal-width bins, as `decision_calibration.py`), and −log p(right answer) |
| coverage | code readout: the model's probability on a valid answer code |
| max Δ between modes | the largest change in any answer's probability between `separate` and the other modes: the same request, batched differently |
| changed | the share of answers a mode changed |
| p50 ms, p95 ms | a request as the client sees it, in `--serve-mode`; every mode's are in the JSON report |

With `--current`, the report also counts the answers only one of the two got
right and gives McNemar's exact p-value: whether the difference in
accuracy is more than the luck of the set.

**The thresholds.** Every one is an option; the defaults, and why:

| Check | Default | Why |
|---|---|---|
| `--min-answers` | 50 | below 50 answers of a type, its accuracy is known to no better than ±14 points (95% Wilson interval at 50%): too little to rest a verdict on |
| better than the commonest answer | — | a model no more often right than one that gives every question its commonest answer has learnt nothing about the states |
| `--max-ece` | 0.10 | a decision's probability is what thresholds are set on ("above 0.9, automate"): on average it may sit at most 10 points from how often the model is right |
| `--max-mode-delta` | 0.05 | a decision must not depend on how it was batched: models trained for decisions stay under it (Jev-Style 0.8B Q4_K_M: 0.024 on a CPU, 0.039 on a GPU), Qwen3-0.6B Q4_K_M, which is not, moves by up to 0.53 ([docs/engine.md](../../docs/engine.md)) |
| `--max-mode-flips` | 0.01 | an answer that changes with the evaluation mode cannot be replayed: at most one answer in a hundred near a tie |
| `--min-coverage` | 0.90 | below it, much of the model's probability goes to something other than an answer code, and its probabilities describe a minority of what it would say |
| `--max-accuracy-drop` | 0.02 | against `--current`: a replacement may be faster or better calibrated, not less often right |
| `--max-latency-ratio` | 1.5 | against `--current`, p95 in `--serve-mode`: callers budget for the decision they have; half again as slow is another budget |
| `--max-p95-ms` | none | depends on the hardware and the caller: set it for the machine that will serve |

A failed check prints why it matters. A model served with a temperature
(`eullm.temperature`; `eullm-forge decisions train` fits one on its dev
split) is qualified with it: `--candidate-temperature`,
`--current-temperature`.

| Option | Default | |
|---|---|---|
| `--candidate`, `--current` | — | the servers; `--current` is optional |
| `--candidate-model`, `--current-model` | the one loaded | decision model to ask for |
| `--data` | none | a labelled set, repeatable |
| `--traces` | none | a traces directory, repeatable |
| `--sources` | all | keep the answers whose source starts with one of these, comma-separated |
| `--modes` | `separate,shared_prefix,batched` | the modes compared; noise needs two |
| `--serve-mode` | `shared_prefix` | the mode accuracy, calibration and latency are measured in |
| `--limit` | 0 (all) | requests per set |
| `--out`, `--details` | `qualify-<time>.json`, none | the report; every answer, one JSON line each |

Each request is asked once per mode and per server. On a CPU, where a
Jev-Style 0.8B takes about three seconds a request of three questions,
100 requests against two servers take about half an hour.

## Tests

```bash
python3 -m unittest discover -s bench/reflexbench
```

Offline: the metrics, BM25, the loaders on made-up files, the
qualification test's sets, metrics and thresholds, and every client
against a stand-in server.
