# Reflex — roadmap for EuLLM's decision primitive

**Status:** planning · 29 September 2026 · work starts on `feat/reflexbench`
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

## MVP 0 — ReflexBench, the evaluator  [🔧 now]

One harness to measure any decision against a labelled set and against the
simpler ways of making it. Its first job is tool selection (MVP 1), and it
must answer three questions before anything else is built:

1. **How many tools can be dropped without losing the right one?**
2. **How many tokens does that keep out of the large model's prompt?**
3. **What does the decision cost, on a GPU and on a CPU?**

- [🔧 now] A normalized dataset format, one JSON object per line: the
  request, the tools on offer (name and description), the tools it needs
  (none, one or several).
- [🔧 now] Loaders that download public sets at run time — nothing is
  committed to the repository:
  - [MetaTool](https://github.com/HowieHwong/MetaTool) (MIT): single and
    multi-tool selection, and whether a tool is needed at all;
  - [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
    (Apache-2.0): requests for which no offered function fits;
  - [ToolRet](https://github.com/mangopy/benchmarking-tool-retrieval)
    (Apache-2.0, ACL 2025): 7.6k tasks over a 43k-tool corpus, for scale.
- [🆕 next] An Italian set of our own, a few hundred requests over the tools
  a typical Italian company uses: invoices, calendar, CRM, stock, email.
- [🔧 now] The methods compared on the same items:
  - **no filter**: every tool goes to the large model, the reference cost;
  - **BM25** over names and descriptions, pure keyword matching;
  - **embeddings** from EuLLM itself (`/v1/embeddings`), cosine similarity;
  - **Reflex, layout A**: the request is the state, the tool descriptions
    are the options — natural, but every request reads the whole catalog;
  - **Reflex, layout B**: the catalog is the state, the request goes in the
    question, the options are the tool names. The catalog is the same from
    one request to the next, and since v0.7.20 a repeated state is not read
    again, so a request costs its question and the names. Whether the model
    still understands a tool from its name alone is what the benchmark says;
  - [🆕 next] **two stages**: embeddings keep 20, Reflex picks among them;
  - [🆕 next] **the large model** choosing on its own, the quality ceiling.
- [🔧 now] Metrics:
  - recall@k — every tool the request needs is among the first k;
  - the k that keeps 95% and 99% of requests whole;
  - accuracy on "no tool needed", where the set has such requests;
  - tokens of tool descriptions sent at that k, against all of them,
    counted with the tokenizer of the model that will receive them;
  - decision latency p50 and p95, GPU and CPU, Jev-Style 0.8B and 2B.
- [🔧 now] Benchmark runs write to their own `EULLM_AUDIT_DIR`: thousands of
  decisions do not belong in a production audit trail.

**Done when** a report, reproducible from one command, answers the three
questions above on MetaTool and BFCL, on an RTX 5070 Ti and on a CPU.

## MVP 1 — tool selection, then a RAG sufficiency gate  [🆕 next]

- Tool selection with the best layout ReflexBench finds, and the prompt
  size it saves measured on a real tool-calling model.
- **Kill criterion:** if embeddings reach the same recall at the working k
  for a small fraction of the cost, Reflex is not the answer to tool
  selection. That is a result, not a failure: move to the next use case.
- Next use case: a **RAG sufficiency gate** — answer, retrieve more, or
  abstain — with RAG Enterprise as its first consumer.

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
