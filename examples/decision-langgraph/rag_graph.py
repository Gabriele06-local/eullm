#!/usr/bin/env python3
"""A RAG loop in LangGraph, with Reflex as the gate before the chat model.

The question is embedded by EuLLM and the passages of the documents closest
to it are retrieved. Before any of them reach the chat model, a decision
model judges them through `POST /v1/systemone`, with one `choice` question:

  * `answer`: together, the passages hold every fact the answer needs;
  * `retrieve_more`: they hold some of them, and at least one is missing;
  * `abstain`: none of them holds anything the answer needs.

The conditional edge after the gate, `after_gate()`, acts on it: write the
answer, retrieve the next passages and ask again — at most `--max-more`
times — or abstain. Only an answer reaches the chat model, which is told to
answer from the passages alone and cite them; abstaining costs no
generation at all.

The gate is asked exactly as `bench/reflexbench/ragbench.py` asks it, so
that a threshold ragbench fits on your own labelled cases applies here as
it is: with `--answer-threshold`, the graph answers whenever P(answer)
reaches it, whatever the model chose. Without one, the model's own choice
decides, and on MuSiQue the Jev-Style 2B left to its own choice stopped 65%
of the contexts that did suffice; see bench/reflexbench/README.md.

    eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
    eullm pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
    eullm pull qwen3-8b
    eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \\
                --embedding-model qwen3-embedding-0.6b-gguf-q8_0
    python examples/decision-langgraph/rag_graph.py --chat-model qwen3-8b
    python examples/decision-langgraph/rag_graph.py --chat-model qwen3-8b \\
        "How long do I have to ask for a refund on a yearly plan?"

`--docs` points at a folder of your own Markdown or text files: each
paragraph is a passage.
"""

import argparse
import math
import os
import pathlib
import re
import sys
import time
from typing import TypedDict

import openai
from langgraph.graph import END, START, StateGraph

from eullm_client import EuLLM, EuLLMError

DOCUMENTS = pathlib.Path(__file__).resolve().parent / "documents"

# The gate, word for word as bench/reflexbench/rg_methods.py asks it: a
# threshold calibrated there holds only for the same question.
GATE = "Can the question be answered from these passages alone?"
OPTIONS = {
    "answer": "Yes: together, the passages hold every fact the answer needs.",
    "retrieve_more": (
        "Partly: they hold some of the facts the answer needs, and at least one is missing."
    ),
    "abstain": "No: none of the passages holds anything the answer needs.",
}

# Qwen3-Embedding is trained with an instruction before the query and none
# before the documents.
QUERY_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer it\nQuery:"

ANSWER = (
    "Answer the question from the numbered passages alone, in at most three "
    "sentences, and cite the passages you use, like [2]. If they do not hold "
    "the answer, say so."
)

SAMPLES = [
    "How long do I have to ask for a refund on a yearly plan?",
    "We paid for the Team plan five weeks ago. Can we still get money back, and how?",
    "Does Acme Notes have an office in Lisbon?",
]


def load_documents(folder):
    """Every paragraph of the folder's .md and .txt files, as a passage that
    starts with its document's title, the way MuSiQue's passages do."""
    passages = []
    folder = pathlib.Path(folder)
    for path in sorted([*folder.glob("*.md"), *folder.glob("*.txt")]):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = path.stem
        if lines and lines[0].startswith("# "):
            title, lines = lines[0][2:].strip(), lines[1:]
        for block in re.split(r"\n\s*\n", "\n".join(lines)):
            text = " ".join(block.split())
            if text:
                passages.append({"source": path.name, "title": title, "text": f"{title}: {text}"})
    return passages


def unit(vector):
    """The vector scaled to length 1, so that a dot product is a cosine."""
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class Index:
    """The passages and their vectors, embedded once, as a RAG system's
    index holds them; a question costs one embedding."""

    def __init__(self, eullm, model, passages, query_instruction=QUERY_INSTRUCTION):
        self.eullm, self.model, self.passages = eullm, model, passages
        self.query_instruction = query_instruction
        texts = [p["text"] for p in passages]
        vectors = []
        for start in range(0, len(texts), 32):
            vectors += eullm.embed(texts[start : start + 32], model)
        self.vectors = [unit(v) for v in vectors]

    def rank(self, question):
        """Every passage, as (index, cosine similarity), closest first."""
        query = unit(self.eullm.embed([self.query_instruction + question], self.model)[0])
        scores = [sum(a * b for a, b in zip(query, v)) for v in self.vectors]
        return sorted(enumerate(scores), key=lambda s: -s[1])


def gate_state(question, passages):
    """The question and its passages, numbered: ragbench's decision state."""
    numbered = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, 1))
    return f"Question: {question}\n\nPassages:\n{numbered}"


class Rag(TypedDict, total=False):
    """The graph's state: the question, and what each node added to it."""

    question: str
    ranking: list  # every passage as (index, similarity), closest first
    passages: list  # the passages retrieved so far, in that order
    rounds: int  # retrievals made
    gate: dict  # the gate's last answer: probabilities, the model's choice, ms
    answer: str  # what the chat model wrote
    abstained: str  # why the graph did not answer


def gate_decision(state, max_more, answer_threshold):
    """What happens after the gate, and why: plain code over its answer."""
    gate = state["gate"]
    p = gate["probabilities"]
    if answer_threshold is None:
        decided = gate["choice"]
    elif p["answer"] >= answer_threshold:
        decided = "answer"
    else:
        # Below the threshold, the likelier of the other two.
        decided = "retrieve_more" if p["retrieve_more"] >= p["abstain"] else "abstain"
    if decided == "retrieve_more":
        if state["rounds"] > max_more:
            return "abstain", f"a fact is still missing after {state['rounds']} retrievals"
        if len(state["passages"]) >= len(state["ranking"]):
            return "abstain", "a fact is still missing, and every passage was retrieved"
    if decided == "abstain":
        return "abstain", "the passages hold nothing the answer needs"
    return decided, ""


def build_graph(eullm, chat, index, decision_model=None, k=3, max_more=2, answer_threshold=None):
    """retrieve → gate → write | retrieve | abstain."""

    def retrieve(state):
        ranking = state.get("ranking") or index.rank(state["question"])
        held = state.get("passages", [])
        new = [
            dict(index.passages[i], similarity=score)
            for i, score in ranking[len(held) : len(held) + k]
        ]
        return {"ranking": ranking, "passages": held + new, "rounds": state.get("rounds", 0) + 1}

    def gate(state):
        started = time.perf_counter()
        response = eullm.decide(
            gate_state(state["question"], [p["text"] for p in state["passages"]]),
            {"gate": {"type": "choice", "instructions": GATE, "criteria": OPTIONS}},
            decision_model,
        )
        answer = response["answers"]["gate"]
        return {
            "gate": {
                "probabilities": answer["probabilities"],
                "choice": answer["choice"],
                "ms": (time.perf_counter() - started) * 1000,
            }
        }

    def after_gate(state):
        return gate_decision(state, max_more, answer_threshold)[0]

    def write(state):
        numbered = "\n\n".join(f"[{n}] {p['text']}" for n, p in enumerate(state["passages"], 1))
        message = chat.invoke(
            [
                ("system", ANSWER),
                ("human", f"Passages:\n{numbered}\n\nQuestion: {state['question']}"),
            ]
        )
        return {"answer": message.content}

    def abstain(state):
        return {"abstained": gate_decision(state, max_more, answer_threshold)[1]}

    graph = StateGraph(Rag)
    graph.add_node("retrieve", retrieve)
    graph.add_node("gate", gate)
    graph.add_node("write", write)
    graph.add_node("abstain", abstain)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "gate")
    graph.add_conditional_edges(
        "gate", after_gate, {"answer": "write", "retrieve_more": "retrieve", "abstain": "abstain"}
    )
    graph.add_edge("write", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def run(app, question):
    """Run the graph on one question, printing every step as it happens."""
    print(f"\n{question}", flush=True)
    state = {}
    for update in app.stream({"question": question}, stream_mode="updates"):
        for node, change in update.items():
            held = len(state.get("passages", []))
            state.update(change)
            if node == "retrieve":
                new = state["passages"][held:]
                titles = ", ".join(f"{p['title']} {p['similarity']:.2f}" for p in new)
                print(f"  retrieve  {titles}", flush=True)
            elif node == "gate":
                gate = change["gate"]
                p = gate["probabilities"]
                print(
                    f"  gate      answer {p['answer']:.2f} · retrieve_more "
                    f"{p['retrieve_more']:.2f} · abstain {p['abstain']:.2f}  "
                    f"(model: {gate['choice']})  {gate['ms']:.0f} ms",
                    flush=True,
                )
            elif node == "write":
                lines = change["answer"].strip().splitlines() or [""]
                print(f"  answer    {lines[0]}")
                for line in lines[1:]:
                    print(f"            {line}")
            elif node == "abstain":
                print(f"  abstain   {change['abstained']}", flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("questions", nargs="*", help="questions (default: three samples)")
    parser.add_argument("--url", default="http://localhost:11434", help="EuLLM server URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--decision-model", default=None, help="decision model (default: the one loaded)"
    )
    parser.add_argument(
        "--embed-model",
        default="qwen3-embedding-0.6b-gguf-q8_0",
        help="embedding model (default: qwen3-embedding-0.6b-gguf-q8_0)",
    )
    parser.add_argument(
        "--query-instruction",
        default=QUERY_INSTRUCTION,
        help="text put before the question when it is embedded, \\n a newline "
        "(Qwen3-Embedding's by default; '' for a model trained without one)",
    )
    parser.add_argument(
        "--chat-model", required=True, help="the chat model that answers, e.g. qwen3-8b"
    )
    parser.add_argument(
        "--docs", default=str(DOCUMENTS), help="folder of .md or .txt files to answer from"
    )
    parser.add_argument("--k", type=int, default=3, help="passages retrieved at a time (default 3)")
    parser.add_argument(
        "--max-more",
        type=int,
        default=2,
        help="most times the gate may send the graph back to retrieve more (default 2)",
    )
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=None,
        help="answer whenever P(answer) reaches this, a threshold calibrated on your own "
        "cases with bench/reflexbench/ragbench.py (default: the model's own choice)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    args = parser.parse_args(argv)
    if args.k < 1 or args.max_more < 0:
        parser.error("--k must be at least 1, and --max-more at least 0")

    passages = load_documents(args.docs)
    if not passages:
        parser.error(f"no .md or .txt file with text in {args.docs}")
    instruction = args.query_instruction.replace("\\n", "\n")
    eullm = EuLLM(args.url, args.api_key, args.timeout)
    try:
        started = time.perf_counter()
        index = Index(eullm, args.embed_model, passages, instruction)
        print(
            f"{len(passages)} passages embedded in {time.perf_counter() - started:.1f} s",
            file=sys.stderr,
        )
        app = build_graph(
            eullm,
            eullm.chat_model(args.chat_model),
            index,
            args.decision_model,
            args.k,
            args.max_more,
            args.answer_threshold,
        )
        for question in args.questions or SAMPLES:
            run(app, question)
    except EuLLMError as e:
        raise SystemExit(str(e)) from None
    except openai.APIError as e:
        raise SystemExit(f"{args.url} (chat model {args.chat_model}): {e}") from None


if __name__ == "__main__":
    main()
