# decision-langgraph — Reflex decides, LangGraph routes, a chat model writes

Two [LangGraph](https://github.com/langchain-ai/langgraph) applications in
which every routing decision is made by Reflex, EuLLM's decision primitive
(`POST /v1/systemone`), and a chat model served by the same EuLLM writes,
through its OpenAI-compatible `/v1/chat/completions`. LangGraph orchestrates;
the decisions are local, come back as probabilities, and each one is in
EuLLM's audit trail.

- `triage_graph.py` — support tickets: Reflex picks the team, the urgency and
  whether a person must handle the ticket; the team's node drafts the reply,
  and a ticket the model is unsure of waits for a person.
- `rag_graph.py` — questions over a few documents: Reflex judges whether the
  passages retrieved can answer the question, and the graph answers,
  retrieves more, or abstains without generating anything.

## Setup

EuLLM with a decision model, an embedding model and a chat model of your
choice. The Jev-Style 2B is the decision model trained for this; Qwen3-8B
(Apache-2.0) is a good chat model beside it on a 16 GB card, and any chat
model you have pulled works:

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
eullm pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
eullm pull qwen3-8b

eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \
            --embedding-model qwen3-embedding-0.6b-gguf-q8_0
```

The decision and embedding models load at startup and stay resident; the
chat model loads on the first request that names it, so the first draft
takes a few seconds longer. Then, from another terminal, in the repository:

```bash
python3 -m venv ~/.venvs/eullm-langgraph        # sudo apt install python3-venv, if missing
. ~/.venvs/eullm-langgraph/bin/activate
pip install -r examples/decision-langgraph/requirements.txt

python examples/decision-langgraph/triage_graph.py --chat-model qwen3-8b
python examples/decision-langgraph/rag_graph.py --chat-model qwen3-8b
```

`--url` points either script at another server, and `--api-key` (or
`EULLM_API_KEY`) sends a key when the server requires one: to EuLLM, never
the `OPENAI_API_KEY` of your environment. `--decision-model` names another
decision model, which the server then loads in place of its own.

The graphs talk to your EuLLM server and nothing else. LangGraph sends
traces to LangSmith only when tracing is switched on in the environment
(`LANGSMITH_TRACING=true`): leave it off and the tickets and questions stay
on your machine.

## Support triage: `triage_graph.py`

```
                              ┌─→ billing ───┐
                              ├─→ technical ─┤
START ─→ decide ─→ route() ─→ ├─→ account ───┼─→ END
                              ├─→ sales ─────┘
                              └─→ person ─→ interrupt() ─→ a team, or END
```

| Node | What it does |
|---|---|
| `decide` | One `/v1/systemone` request with the ticket (sender, subject, message) as the state and three questions about it, read once for all three: which team (`choice` among the teams), is the customer blocked or facing a deadline today (`noul`), must a person handle it (`noul`). The probabilities go into the graph's state, and the one about being blocked sets the priority. |
| `route()` | The conditional edge: plain code over those probabilities. A ticket a person must handle goes to `person` (`--person-threshold`, 0.5), and so does one whose likeliest team is under `--min-team` (0.6); any other goes to its team. |
| `billing`, `technical`, `account`, `sales` | The chat model drafts the reply as that team, told the ticket has priority when it has. |
| `person` | `interrupt()`: the graph stops and waits, keeping its state, with the reason and the probabilities. A person resumes it with a team, whose node drafts the reply, or keeps the ticket. |

`--ask` asks at the terminal, for every ticket that goes to a person, which
team takes it. Without it those tickets are listed as waiting. The graph
keeps a waiting ticket in memory; an application keeps it in a database
instead (`langgraph-checkpoint-sqlite` or `-postgres`), and resumes it with
`Command(resume="billing")` when the person answers, from whatever screen
they use.

```
  3  Locked out before a presentation    → account    high    (account 0.94, blocked 0.96, person 0.12)
     │ (the account team's draft)
  8  Change of email                     → person     normal  (billing 0.52, blocked 0.04, person 0.06)
     waiting for a person: unsure of the team: billing 0.52, account 0.46, technical 0.01
```

With the Jev-Style 2B, the eight sample tickets go where they should: five
to their teams, with the team at 0.90 or more; the formal notice that
mentions a lawyer, and the stranger in the account, to a person (0.77 and
0.93, no other ticket above 0.12); and the change of email, which needs both
billing and account (0.52 and 0.46), to a person who picks the team. The
two customers who are blocked get priority (0.96, 0.95), no one else above
0.35. The 0.8B routes them the same way, except the change of email, which
it gives to account (0.75).

How a question is put matters as much as the model:

- **"Is blocked or has a deadline today", not "needs an answer today".**
  Asked whether the customer needs an answer today, the 2B said yes, at 0.5
  or more, for seven of the eight tickets, a double charge (0.61) and a
  change of email (0.79) among them.
- **Say what a person would see, not its category.** Asked whether the
  ticket "reports a security breach or a data leak", the 0.8B put the one
  about a stranger's session in the account at 0.26 (the 2B at 0.66); asked
  whether it "says that someone else may have got into the account or seen
  its notes", at 0.64 (the 2B at 0.93).

Try a question on a few tickets you know the answer to before trusting it,
and set the thresholds from those. `--teams` takes a JSON object of team
name → what the team handles, `--tickets` a JSON list of
`{"from", "subject", "message"}`.

## RAG with a gate: `rag_graph.py`

```
START ──→ retrieve ──→ gate ──→ after_gate() ──→ write ──→ END
             ↑                       │
             └──── retrieve_more ────┤
                                     └──→ abstain ──→ END
```

| Node | What it does |
|---|---|
| `retrieve` | The first time, embeds the question with EuLLM (after Qwen3-Embedding's query instruction) and ranks the passages, which were embedded once at startup; each time, adds the next `--k` (3) passages. |
| `gate` | One `/v1/systemone` request: the question and the passages so far, numbered, as the state, and one `choice` among `answer`, `retrieve_more` (a fact is missing) and `abstain` (nothing there helps). |
| `after_gate()` | The conditional edge, plain code: the model's choice, or with `--answer-threshold` P(answer) against the threshold; back to `retrieve` at most `--max-more` (2) times. |
| `write` | The chat model answers from the passages alone, citing them. |
| `abstain` | Says why, and generates nothing. |

The documents are five short texts written for this example about a
made-up notes app, in `documents/`: 19 passages, one per paragraph, each
starting with its document's title. `--docs` takes a folder of your own
Markdown or text files.

```
We paid for the Team plan five weeks ago. Can we still get money back, and how?
  retrieve  Getting help 0.77, Plans and prices 0.72, Refunds 0.65
  gate      answer 0.31 · retrieve_more 0.51 · abstain 0.19  (model: retrieve_more)
  retrieve  Refunds 0.62, Signing in and two-factor authentication 0.55, Refunds 0.54
  gate      answer 0.64 · retrieve_more 0.28 · abstain 0.08  (model: answer)
  answer    ...
```

With the Jev-Style 2B the three sample questions take the three ways out:
the refund window is answered from the first passages (0.93); the question
about the Team plan, whose first passages say that it is paid yearly but not
what is refunded after 30 days, retrieves more (0.51) and is answered with
the passage that says it (0.64); the office in Lisbon, which no document
mentions, is abstained on (0.92). The 0.8B abstains on the second question
at once (0.51, against 0.44 for retrieving more): it takes a context short
of one fact for one that holds nothing, which is where the three-way
decision is weakest in the [RAG gate's
benchmark](../../bench/reflexbench/README.md#the-rag-gate-ragbenchpy).

### A threshold of your own

Left to its own choice, the 2B is a cautious gate: on MuSiQue it stopped 65%
of the contexts that did suffice. Calibrated on a domain's own cases it
stops about half again as many insufficient contexts as the embeddings'
similarity, for the same answers lost. The gate here asks its question word
for word as `bench/reflexbench/ragbench.py` does (a test checks it), so a
threshold ragbench fits on your own labelled cases applies as it is:

```bash
# One JSON object per line: a question, the passages your retrieval returns,
# and whether they suffice (answer), miss a fact (retrieve_more) or hold
# nothing (abstain). A few hundred questions, each with its contexts.
python3 bench/reflexbench/ragbench.py --sets '' --data my-cases.jsonl \
    --methods reflex-gate --out my-gate.json
threshold=$(jq '.results[] | select(.method == "reflex-gate") | .metrics.fitted_threshold' my-gate.json)

python examples/decision-langgraph/rag_graph.py --chat-model qwen3-8b \
    --answer-threshold "$threshold" --docs my-documents/
```

The graph then answers whenever P(answer) reaches the threshold, and below
it takes the likelier of retrieving more and abstaining. [ragbench's
README](../../bench/reflexbench/README.md#the-rag-gate-ragbenchpy) describes
the format of `my-cases.jsonl`, and how the threshold is fitted on one half
of the questions and scored on the other.

## Tests

```bash
pip install -r examples/decision-langgraph/requirements.txt
python -m unittest discover -s examples/decision-langgraph
```

Offline: both graphs run against a stand-in EuLLM, an HTTP server from the
standard library that answers `/v1/systemone`, `/v1/embeddings` and
`/v1/chat/completions` the way EuLLM does, with the decisions each test
sets. They check the routes, the person's interrupt and resume, the
retrieval loop and its limits, the threshold, what each request sends, and
that the gate's question is still ragbench's.

The numbers above were measured on a 4-core CPU, where a decision of the 2B
takes several seconds; on a GPU it takes tens of milliseconds, and the
probabilities can differ in their last digits.
