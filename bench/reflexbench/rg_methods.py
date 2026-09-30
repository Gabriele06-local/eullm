"""The ways the RAG gate is decided.

Every method gives each case a `Decision`: a score that grows with how
likely the passages suffice, and, when the method makes one on its own,
its choice among `answer`, `retrieve_more` and `abstain`.

  * `embed-max`: the highest cosine similarity between the question and a
    passage — the signal a RAG system already has after retrieval. It
    decides nothing by itself: a threshold has to be fitted on labelled
    cases;
  * `reflex-gate`: `/v1/systemone`, the question and the passages as the
    state, one `choice` among the three;
  * `reflex-yesno`: the same state, one `noul`: do the passages hold every
    fact the answer needs?
"""

import time

from rb_methods import Embeddings, post, unit

GATE = "Can the question be answered from these passages alone?"
OPTIONS = {
    "answer": "Yes: together, the passages hold every fact the answer needs.",
    "retrieve_more": (
        "Partly: they hold some of the facts the answer needs, and at least one is missing."
    ),
    "abstain": "No: none of the passages holds anything the answer needs.",
}
YESNO = "Together, do these passages hold every fact needed to answer the question?"


class Decision:
    def __init__(self, score, choice=None, probabilities=None, ms=0.0, server=None):
        self.score = score  # higher: likelier that the passages suffice
        self.choice = choice  # the method's own decision, if it makes one
        self.probabilities = probabilities  # what the choice was read from
        self.ms = ms  # wall time of the decision, as the client sees it
        self.server = server or {}  # tokens and timings reported by EuLLM


def state(case):
    """The question and its passages, numbered, as the decision state."""
    passages = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(case.passages, 1))
    return f"Question: {case.question}\n\nPassages:\n{passages}"


class EmbedMax:
    """The passages are embedded once, as an index would hold them; a case
    costs the embedding of its question."""

    name = "embed-max"

    def __init__(self, url, model, api_key, timeout, query_prefix=""):
        self.embeddings = Embeddings(url, model, api_key, timeout, query_prefix)
        self.seen = {}

    def decide(self, case):
        missing = [p for p in dict.fromkeys(case.passages) if p not in self.seen]
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            for text, vector in zip(batch, self.embeddings.embed(batch)):
                self.seen[text] = unit(vector)
        started = time.perf_counter()
        prefix = self.embeddings.query_prefix
        query = unit(self.embeddings.embed([prefix + case.question])[0])
        ms = (time.perf_counter() - started) * 1000
        best = max(sum(a * b for a, b in zip(query, self.seen[p])) for p in case.passages)
        return Decision(best, ms=ms)


class Reflex:
    """One question about the case, asked of `/v1/systemone`."""

    def __init__(self, url, model, api_key, timeout):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.model, self.api_key, self.timeout = model, api_key, timeout

    def ask(self, case, question):
        payload = {"state": state(case), "questions": {"q": question}}
        if self.model:
            payload["model"] = self.model
        started = time.perf_counter()
        body = post(self.url, payload, self.api_key, self.timeout)
        ms = (time.perf_counter() - started) * 1000
        info = body.get("eullm", {})
        server = {
            "evaluated_tokens": info.get("evaluated_tokens"),
            "prompt_tokens": info.get("prompt_tokens"),
            "request_ms": info.get("request_ms"),
            "model": body.get("model"),
        }
        return body["answers"]["q"], ms, server


class ReflexGate(Reflex):
    name = "reflex-gate"

    def decide(self, case):
        question = {"type": "choice", "instructions": GATE, "criteria": OPTIONS}
        answer, ms, server = self.ask(case, question)
        probabilities = answer["probabilities"]
        return Decision(probabilities["answer"], answer["choice"], probabilities, ms, server)


class ReflexYesNo(Reflex):
    name = "reflex-yesno"

    def decide(self, case):
        answer, ms, server = self.ask(case, {"type": "noul", "instructions": YESNO})
        p = answer["noul"]
        # A yes/no says whether to answer; it cannot tell a missing fact from
        # an irrelevant context, so its own decision is two-way.
        choice = "answer" if p >= 0.5 else "not answer"
        return Decision(p, choice, {"yes": p}, ms, server)
