# Reflex — roadmap for EuLLM's decision primitive

**Status:** MVP 0 done; MVP 1, the RAG gate: Reflex stops half again as many
insufficient contexts as embeddings, once calibrated · 30 September 2026
**Built on:** `POST /v1/systemone`, shipped in v0.7.20

Operational document: every item has a tag —
**[✅ done]** already implemented · **[🔧 now]** in progress ·
**[🆕 next]** planned, not started. Update it as decisions change; the
reasoning behind each decision is written next to it so that it can be
revisited, not just obeyed.

---

## What Reflex is, and is not

Reflex is EuLLM's **decision primitive**: a small model answers typed
questions about a state — yes or no, one of N options, a level on a scale —
with probabilities read straight from the model. Nothing is generated, and
every decision goes to the audit trail. It is served today by
`/v1/systemone`, with the Jev-Style 0.8B and 2B models.

**EuLLM stays an inference platform.** Reflex is something orchestrators
call, not an orchestrator: no executor, no workflow engine, no agent loop of
our own. LangGraph, the OpenAI Agents SDK, n8n and custom code already
orchestrate well; what none of them has is a local, fast, audited decision
they can call, and a way to train it on the user's own decisions.

"Reflex" names the fast decision; the endpoint's name, System One, says the
same thing — the quick judgement, next to the slower reasoning of a large
model.

## Decisions already taken

- **A primitive, not a framework.** MCP arrives as an adapter that exposes
  Reflex to existing agents, never as an execution runtime of ours.
- **Code filters, the model judges.** Options that code can rule out
  deterministically are removed *before* the model sees them, not vetoed
  after. Measured in the Snake example: offered a way to the food that
  ended in a trap but pointed at the food, the Jev-Style 0.8B took it
  20 times out of 20, whatever the facts said.
- **A decision and its arguments are separate things.** A decision model
  scores enumerated options; it does not write a search query or fill in a
  `top_k`. Arguments come from code, from the downstream model, or from a
  separate constrained generation — the engine already has GBNF grammars —
  measured on its own, with its own latency.
- **An evaluator before any feature.** Every decision question ships with a
  labelled set and a comparison against simpler baselines. Phrasing alone
  moved the 2B from 11 to 16 right answers in 20 on the hardest Snake boards,
  and email triage from three false phishing alarms to 13 emails right out
  of 13.
- **No confidence thresholds without calibration** measured on the
  domain's own labelled data (expected calibration error). The Jev-Style
  temperature was fitted on its release's data, not on ours, and an
  instruction-tuned chat model moved its answers by up to 0.53 between
  evaluation modes.
- **Full traces only opt-in, and local.** The audit trail keeps a state as a
  SHA-256 on purpose. Capturing the text to train on is an explicit choice,
  stored locally, with personal data redacted.
- **What we claim.** Auditable micro-decisions, locally, in milliseconds on
  a GPU: each decision is a probability distribution over enumerated
  options, logged, and replayable bit for bit in `separate` mode. That helps
  document automated decisions in regulated work; it does not by itself make
  a system compliant, and we never say it does.

---

## MVP 0 — ReflexBench, the evaluator  [✅ done]

One harness to measure any decision against a labelled set and against the
simpler ways of making it. Its first job is tool selection (MVP 1), and it
must answer three questions before anything else is built:

1. **How many tools can be dropped without losing the right one?**
2. **How many tokens does that keep out of the large model's prompt?**
3. **What does the decision cost, on a GPU and on a CPU?**

The harness is [`bench/reflexbench/`](../bench/reflexbench/README.md):
standard library only, one command, offline unit tests.

- [✅ done] A normalized dataset format, one JSON object per line: the
  request, the tools on offer (name and description), the tools it needs
  (none, one or several).
- Loaders that download public sets at run time — nothing is committed to
  the repository:
  - [✅ done] [MetaTool](https://github.com/HowieHwong/MetaTool) (MIT):
    single and multi-tool selection over one catalog of 199 tools;
  - [✅ done] [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
    (Apache-2.0): one function of several fits, or none of them does;
  - [🆕 next] [ToolRet](https://github.com/mangopy/benchmarking-tool-retrieval)
    (Apache-2.0, ACL 2025): 7.6k tasks over a 43k-tool corpus, for scale.
- [🆕 next] An Italian set of our own, a few hundred requests over the tools
  a typical Italian company uses: invoices, calendar, CRM, stock, email.
- [✅ done] The methods compared on the same items:
  - **no filter**: every tool goes to the large model, the reference cost
    every "specs kept" figure is a share of;
  - **BM25** over names and descriptions, pure keyword matching;
  - **embeddings** from EuLLM itself (`/v1/embeddings`), cosine similarity;
    Qwen3-Embedding-0.6B (Apache-2.0) with its query instruction is the
    baseline to beat;
  - **Reflex, layout A**: the request is the state, the tool descriptions
    are the options — natural, but every request reads the whole catalog;
  - **Reflex, layout B**: the catalog is the state, the request goes in the
    question, the options are the tool names. The catalog is the same from
    one request to the next, and since v0.7.20 a repeated state is not read
    again, so a request costs its question and the names. Whether the model
    still understands a tool from its name alone is what the benchmark says;
  - [✅ done] **two stages**: embeddings keep 20 (`--shortlist`), Reflex,
    layout A, ranks them;
  - [🆕 next] **the large model** choosing on its own, the quality ceiling.
- Metrics:
  - [✅ done] recall@k — every tool the request needs is among the first k;
  - [✅ done] the k that keeps 95% and 99% of requests whole;
  - [✅ done] accuracy on "no tool needed", where the set has such requests,
    and how often "no tool" wins on requests that do need one;
  - [✅ done] the share of the tool specs still sent at that k, against all
    of them; [🆕 next] the same in tokens, counted with the tokenizer of the
    model that will receive them (four characters a token until then);
  - [✅ done] decision latency p50 and p95, and the tokens the decision
    model read; [🔧 now] measured on GPU and CPU, Jev-Style 0.8B and 2B.
- [✅ done] Benchmark runs write to their own `EULLM_AUDIT_DIR`: thousands
  of decisions do not belong in a production audit trail.

**Done when** a report, reproducible from one command, answers the three
questions above on MetaTool and BFCL, on an RTX 5070 Ti and on a CPU.

### First results — RTX 5070 Ti, Jev-Style 2B, 500 requests a set

30 September 2026, embeddings from Qwen3-Embedding-0.6B with its query
instruction. One tool needed out of MetaTool's 199:

| Method | Right tool first | In the first 10 | k95 | p50 | Tokens read |
|---|---|---|---|---|---|
| BM25 | 28.2% | 54.8% | 151 | 1 ms | — |
| Embeddings | 75.4% | 94.6% | 11 | 51 ms | — |
| Reflex, layout A | 73.4% | 93.2% | 17 | 439 ms | 7,151 |
| Reflex, layout B | 60.8% | 87.8% | 20 | 116 ms | 1,463 |
| Two stages: the 2B ranks the embeddings' 20 | 74.4% | 95.2% | 10 | 109 ms | 613 |

- **Alone, Reflex does not beat the embeddings** on one tool out of 199:
  slightly less recall for nine times the latency (A), or clearly less
  (B). By the kill criterion of MVP 1, Reflex does not replace embeddings
  for tool selection.
- **Nor does it in two stages.** The two miss different requests — where
  the embeddings do not rank the right tool first (123 of 500), Reflex A
  does 36 times — but neither says when it is the one to trust. The 2B
  ranking the embeddings' shortlist of 20 puts the right tool first 74.4%
  of the time; fusing the two rankings by reciprocal rank, 73.0%. An
  earlier estimate of 78.2% for the fusion counted every tie between the
  two rankings in the right tool's favour, and was wrong.
- **Where several tools are needed, it helps.** With two tools of 199
  (MetaTool multi), the two stages have both within the first 3 for 49.7%
  of the requests and within 10 for 80.7%, against 35.6% and 71.6% for the
  embeddings; Reflex B keeps 95% of the requests at k = 30, the embeddings
  at 40.
- **Few tools, BFCL live (2–37 a request):** Reflex ranks the right one
  first 93–95% of the time, the embeddings 91%, in 26–28 ms against
  109 ms — though the engine's embeddings pay for a context created on
  every request, a cost it could avoid.
- **"No tool":** on BFCL live, Reflex A says it for 69% of the requests no
  function fits and wrongly for 2.0% of those one does (B: 48% and 0.8%) —
  a decision the embeddings cannot make without a calibrated threshold. On
  MetaTool's 199 generic plugins it says it wrongly for 13.8% of the
  requests (two stages: 19.8%): a gate for a small, well-described set of
  tools, not for a large catalog.
- **Verdict for MVP 1:** the kill criterion is met for picking one tool out
  of a large catalog — the embeddings do it as well at a fraction of the
  cost. What Reflex adds is the judgement on few options: which of a
  handful, several at once, or none of them.

**The 0.8B on the same GPU** (500 requests a set, "no tool" offered on
every set) reads MetaTool's catalog in 9 questions, about 10,200 tokens, in
414 ms — no faster than the 2B — and ranks the right tool first 65.6% of the
time (2B: 73.4%). In two stages it ranks the embeddings' shortlist of 20
worse than they do: 70.4% first against 75.4%. Where it earns its place is
"no tool" on few tools: on BFCL live it says it for 66.6% of the requests no
function fits and for only 1.8% of those one does, a decision the
embeddings cannot make without a calibrated threshold. In layout B it barely
knows a tool by its name (20.8% first), and a request of 2,000 tokens does
not fit its question at all.

**On a CPU** — the same 2B, release binary v0.7.20 (x86-64-v3), 20
requests a set — a decision costs 76 s reading MetaTool's 199 tools in
layout A and 20 s in layout B, 1.5–1.8 s on BFCL's few, against 0.3–0.4 s
for an embedding: 55 to 170 times the GPU's latency for Reflex, 8 times for
the embeddings. The 2B is not a reflex on a CPU; for few options the 0.8B
is the candidate there, not yet measured.

## MVP 1 — tool selection, then a RAG sufficiency gate  [🆕 next]

- [✅ measured] Tool selection with the best layout ReflexBench finds.
- **Kill criterion:** if embeddings reach the same recall at the working k
  for a small fraction of the cost, Reflex is not the answer to tool
  selection. That is a result, not a failure: move to the next use case.
  **Met** for one tool out of a large catalog (see the MVP 0 results): the
  embeddings choose the shortlist. Reflex keeps the judgement on few
  options — which of a handful, several at once, or none of them.
- [🔧 now] A **RAG sufficiency gate** — answer, retrieve more, or abstain —
  with RAG Enterprise as its first consumer: a judgement on few options,
  the kind the benchmark found Reflex good at.
  - [✅ done] Its evaluator, [`ragbench.py`](../bench/reflexbench/README.md#the-rag-gate-ragbenchpy):
    MuSiQue (CC BY 4.0), three contexts a question that differ only in
    what they hold — every passage the answer needs, all but one, none of
    them; the embeddings' best similarity, the signal a RAG system already
    has, against Reflex's choice and a yes/no; thresholds fitted on a dev
    half, every method scored on the test half.
  - [✅ done] The first run on the GPU: see the results below.
  - [✅ done] The ceiling: Qwen3-8B, four times the size and not trained
    for decisions, asked the same questions through the code readout. It
    does no better — see the results below: MuSiQue's limit is the task,
    not the 2B's size.
  - [🔧 now] An Italian set: `rg_openbook.py` writes it from Forge's
    open-book pairs — each question asked by topic about an article of
    Italian law, with the articles retrieval finds, its own among them or
    left out, the two contexts Forge trains the legal model on. Written
    where the pairs are; while the cluster is down, `--by-heading` asks by
    each article's rubrica from the legislation records alone. Its run is
    next.

### RAG gate, first results — RTX 5070 Ti, Jev-Style 2B

30 September 2026. MuSiQue, 1,000 questions, 3,000 cases of five passages;
thresholds fitted on 500 questions, every number below on the other 500.
Of the cases whose passages do not suffice, the share each method stops
when it may stop at most about one sufficient case in ten:

| Method | Stopped: all | one passage missing | nothing relevant | Sufficient stopped | AUROC | Within a question | p50 |
|---|---|---|---|---|---|---|---|
| Embeddings, best similarity | 33.3% | 18.0% | 48.6% | 8.2% | 0.704 | 0.754 | 55 ms |
| Reflex, choice among three | 48.1% | 29.0% | 67.2% | 9.6% | 0.769 | 0.815 | 49 ms |
| Reflex, yes/no | 47.9% | 27.8% | 68.0% | 9.4% | 0.763 | 0.794 | 50 ms |

- **Reflex is the better gate:** at the same price in good answers lost it
  stops about half again as many insufficient contexts as the embeddings,
  in the same time on a GPU (about 770 tokens a decision).
- **Only with a threshold calibrated on the domain's own cases.** Left to
  its own decision the 2B is far too cautious: it stops 90% of the
  insufficient cases but 65% (choice) to 75% (yes/no) of the sufficient
  ones. The rule on thresholds holds: 500 labelled questions made it
  usable.
- **The three-way decision does not work yet:** 46% right, no better than
  the embeddings' two thresholds (47%). A context short of one hop is
  mostly taken for one that holds nothing, and that is the hard case for
  every method.
- **A larger model is not the answer.** Qwen3-8B, asked the same two
  questions: AUROC 0.765 (choice) and 0.772 (yes/no) against the 2B's
  0.769 and 0.763, the same balanced accuracy at a fitted threshold (0.70),
  in 134–139 ms against 49–50 ms. It follows the passages better within a
  question (0.83–0.86 against 0.79–0.82), and its own yes/no is less
  cautious — it stops 62% of the insufficient cases and 22% of the
  sufficient ones without any threshold — but calibrated, the 2B does as
  well at a third of the latency.
- MuSiQue's questions take two to four hops; most questions put to a
  company's documents take one. The Italian set is what says how the gate
  does on those.

## MVP 2 — adapters, not a runtime  [🆕 next]

- An MCP server exposing `decide` and `select_tools` over `/v1/systemone`,
  so that any MCP client — an IDE agent, a desktop assistant — can use
  Reflex without code of ours in its loop.
- Examples for LangGraph and n8n.
- A server-side policy: options filtered before the model sees them, and a
  deny list. Configured through `EULLM_*` environment variables, like every
  perimeter setting of the engine (see `engine/CLAUDE.md`), with remote
  models off unless enabled.

## MVP 3 — several chat models resident, then `model: "auto"`  [🆕 next]

- **The engine keeps one chat model loaded today**, next to the embedding
  and decision slots; asking for another model swaps it, which takes
  seconds. Choosing per request between two local chat models needs both
  resident: a second chat slot, VRAM sizing for both, a scheduler per model.
  Built behind a flag, with its own tests, validated on GPU before merge —
  the one part of this roadmap that changes the engine's memory management.
- **Large-VRAM testing on EuroHPC**, where two big models fit side by side:
  - **LUMI-G** — AMD MI250X, 64 GB per GCD, eight GCDs per node. The
    development allocation EHPC-DEV-2026D09-278 funds exactly this kind of
    work: memory against model size, execution across GPUs, NVIDIA against
    AMD. See [`lumi/lumi-g.md`](lumi/lumi-g.md).
  - **Leonardo** — NVIDIA A100 64 GB, four per node. The AI Factory
    allocation EHPC-AIF-2026PG01-1147 exists to train `legal-it-4b` and ends
    on 2 November 2026: engine tests there only if its budget leaves room.
    See [`leonardo-allocation-plan.md`](leonardo-allocation-plan.md).
- Then `model: "auto"` on the OpenAI and Ollama endpoints: Reflex picks the
  resident model that answers, with no change in the application. Measured
  in large-model calls avoided, latency, and quality against always using
  the large one.

## MVP 4 — decision models trained on your decisions  [🆕 next]

- Opt-in, local capture of traces: state, options, decision, outcome,
  correction.
- Forge trains a small decision model from them, with rules, a large model
  or people as the teacher.
- A qualification test before any decision model is swapped in:
  calibration, noise between evaluation modes, accuracy on the domain's set.
  "Interchangeable" is earned by passing it, not by a configuration line.

---

## Risks

- **Strategic.** Embeddings may match Reflex on tool selection at a fraction
  of the cost. ReflexBench exists to find that out early and cheaply.
- **Dependency.** The Jev-Style models are a third party's (Apache-2.0).
  MVP 4, decision models of our own from Forge, is the long-term answer.
- **CPU cost.** Reading a large tool catalog on every request may take tens
  of seconds on a CPU. Layout B and the two-stage variant exist to bring that
  down, and the benchmark measures it instead of guessing.
- **The engine.** Nothing before MVP 3 changes it: MVP 0 to 2 are additive
  and use the endpoint as it is. Without a decision model loaded, chat,
  completions and embeddings behave exactly as they do now.
