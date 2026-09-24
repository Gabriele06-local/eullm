"""Instruction/answer pairs generated from the legal corpus, for stage 3.

Why this exists
---------------
Stage 2 produces a *continuation* model: distillation teaches Qwen3-4B-Base
to write Italian legal text the way the teacher does, and a model trained
only on that continues text rather than answering it. Stage 3 used to train
on some fifteen identity pairs ("Chi sei?"), which can teach a name but not
how to be an assistant. Turning the student into something that answers a
question needs a few thousand examples of questions being answered, in the
domain.

They are generated, not collected. Open Italian instruction sets are either
non-commercial or produced by models whose terms restrict training on their
output. `Qwen/Qwen3-30B-A3B-Instruct-2507` is Apache 2.0 with no such
clause, and it is the same architecture as the distillation teacher, so it
loads on Leonardo the way the teacher already does.

Grounding
---------
Every pair is written FROM a passage of the training corpus, handed to the
generator together with the task. The generator is asked to use only what
the passage says, so the answers carry the corpus's law rather than
whatever the generator half-remembers — the difference between teaching the
student the Codice civile and teaching it a 30B model's paraphrase of it.

The passages come from ``train.jsonl`` and never from ``val.jsonl`` or the
Consiglio di Stato held-out corpus: both are evaluation sets, and a pair
generated from them would put the answers to the exam into the training
data.

Personal data
-------------
The corpus is already pseudonymised (see ``anonymize.py``). The passage is
run through the regex layer again before it reaches the generator, and any
generated pair that still contains an anonymiser placeholder, or something
the regex layer would redact, is rejected rather than repaired. An
assistant that answers with ``[PERSONA_1]`` is broken, and one that answers
with a codice fiscale is worse.

What lives here and what does not
---------------------------------
This module is pure Python — prompts, parsing and the quality filters — so
it is testable without a GPU. The model itself is driven by
``forge/scripts/generate_instructions.py``. Generated data never goes into
git, in this repository or any other: it is derived from rulings.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field

from .anonymize import AnonymiserConfig, anonymize_text

# The three things an assistant for Italian law is asked to do most, and the
# three the student cannot do yet:
#
#   qa           a question asked WITHOUT the text, answered from knowledge.
#                The passage is shown to the generator only; the student sees
#                the question alone, so this is what teaches it to know.
#   riassunto    summarise a text the user pastes in.
#   spiegazione  explain a text the user pastes in, to someone who is not a
#                lawyer.
#
# `qa` is weighted highest because it is the one that needs the corpus: the
# two context tasks teach a skill, `qa` teaches the law.
TASK_WEIGHTS: dict[str, float] = {"qa": 0.6, "riassunto": 0.2, "spiegazione": 0.2}

SYSTEM_PROMPT = (
    "Sei un giurista italiano esperto che prepara materiale didattico. "
    "Scrivi in italiano corretto e preciso. Usa esclusivamente le "
    "informazioni contenute nel testo fornito: non aggiungere norme, date, "
    "numeri o fatti che il testo non contiene. Non nominare mai persone "
    "fisiche e non riportare segnaposto tra parentesi quadre come "
    "[PERSONA_1]: se servono, usa ruoli generici (il ricorrente, "
    "l'imputato, il datore di lavoro). Rispondi SOLO con un oggetto JSON "
    "valido, senza testo prima o dopo."
)

_TASK_PROMPTS = {
    "qa": (
        "Ecco un testo giuridico:\n\n<testo>\n{passage}\n</testo>\n\n"
        "Scrivi UNA domanda che un cittadino o un professionista potrebbe "
        "porre a un assistente legale, e la risposta corretta. La domanda "
        "deve riguardare la REGOLA o il PRINCIPIO giuridico che il testo "
        "applica, non la vicenda specifica: niente parti, date, città, "
        "numeri di ricorso o esiti di questa causa. Deve avere senso per "
        "chiunque, senza conoscere questa causa e SENZA il testo: non dire "
        "\"secondo il testo\", \"nel passaggio\" o simili. La risposta "
        "deve essere completa, basata solo su quanto il testo afferma, e "
        "citare l'articolo o la norma quando il testo li indica; se cita una "
        "sentenza, senza il nome delle parti. Lunghezza della risposta: da 3 "
        "a 10 frasi.\n\n"
        'Formato: {{"domanda": "...", "risposta": "..."}}'
    ),
    "riassunto": (
        "Ecco un testo giuridico:\n\n<testo>\n{passage}\n</testo>\n\n"
        "Scrivi un riassunto fedele del testo, in 4-8 frasi, che ne "
        "conservi i punti giuridici essenziali (norme richiamate, principio "
        "affermato, esito).\n\n"
        'Formato: {{"risposta": "..."}}'
    ),
    "spiegazione": (
        "Ecco un testo giuridico:\n\n<testo>\n{passage}\n</testo>\n\n"
        "Spiega il testo a una persona senza formazione giuridica: cosa "
        "stabilisce, a chi si applica, che conseguenze ha. Linguaggio "
        "semplice ma esatto, 4-8 frasi, senza inventare nulla.\n\n"
        'Formato: {{"risposta": "..."}}'
    ),
}

# What the USER says in the context tasks. Several phrasings, chosen per
# passage, so the student learns the task and not one sentence.
_CONTEXT_INSTRUCTIONS = {
    "riassunto": [
        "Riassumi il seguente testo:",
        "Puoi farmi un riassunto di questo testo giuridico?",
        "Mi serve una sintesi dei punti principali di questo testo:",
        "Riassumi in poche frasi:",
    ],
    "spiegazione": [
        "Spiegami in parole semplici cosa dice questo testo:",
        "Non sono un avvocato: cosa significa questo testo?",
        "Puoi spiegarmi questa norma in modo comprensibile?",
        "Cosa vuol dire, in pratica, il seguente testo?",
    ],
}

# Anonymiser output: [PERSONA_1], [CODICE_FISCALE], [RICORRENTE_2], …
# Case-insensitive on purpose: the generator varies capitalisation
# ([persona_1]), and a missed placeholder lands verbatim in training.
RE_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_]*(?:_\d+)?\]", re.IGNORECASE)
# A closed-book answer that points at a text the user never gave. Not
# "documento" (il documento informatico, di identità…) and not "testo unico",
# which is the name of half of Italian administrative law.
RE_TEXT_REFERENCE = re.compile(
    r"\b(?:secondo|nel|dal|del|il|questo|quel)\s+(?:testo|passaggio|brano)\b(?!\s+unic)",
    re.IGNORECASE,
)
# A closed-book question about ONE case — "la sentenza della Corte d'Appello di
# Bari del 15 gennaio 2024 è stata…" — has an answer the student cannot know,
# only invent. Training on it teaches exactly that: stating the outcome of a
# case with confidence. The pilot of 24 September produced one in three
# samples. A full date or a case number in the question is the signature —
# except that a date is also how Italian law names a STATUTE ("legge 7 agosto
# 1990, n. 241"), and those questions are exactly the ones wanted, so a date
# right after the name of a legislative act does not count.
_MONTHS = ("gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
           "settembre|ottobre|novembre|dicembre")
RE_FULL_DATE = re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\b", re.IGNORECASE)
RE_CASE_NUMBER = re.compile(
    r"\b(?:ricorso|sentenza|ordinanza|r\.g\.)\s*n\.?\s*\d", re.IGNORECASE
)
RE_ACT_BEFORE_DATE = re.compile(
    r"(?:legge|l\.|decreto(?:[- ]legge|\s+legislativo)?|d\.\s*lgs\.?|d\.\s*l\.|"
    r"d\.\s*p\.\s*r\.|dpr|regolamento|direttiva|codice)\s*(?:\(\w+\)\s*)?$",
    re.IGNORECASE,
)


def case_specific(question: str) -> str:
    """The first case-specific marker in a question, or "" if there is none."""
    m = RE_CASE_NUMBER.search(question)
    if m:
        return m.group()
    for m in RE_FULL_DATE.finditer(question):
        if not RE_ACT_BEFORE_DATE.search(question[max(0, m.start() - 40):m.start()]):
            return m.group()
    return ""
# Italian case citations carry the defendant's surname between the date and
# the Rv. number: "Sez. 6, n. 25273 del 23/05/2018, Zidane, Rv. 273392". The
# anonymiser's regex layer does not see a Title-Case surname, and the pilot
# let one through. The name is dropped and the citation kept: the citation is
# how a lawyer finds the ruling, the surname is only personal data.
RE_CITATION_NAME = re.compile(
    r"(\b\d{4}|\d{1,2}/\d{1,2}/\d{2,4}),\s*"
    r"[A-ZÀ-Ý][\w'’.-]*(?:\s+[A-ZÀ-Ý][\w'’.-]*){0,3},\s*(Rv\.)"
)
# Very common Italian words. A generated answer with almost none of them is
# not Italian, whatever else it is.
_IT_STOPWORDS = frozenset(
    "il lo la i gli le un una di a da in con su per tra fra e che non è del "
    "della dei delle al alla ai alle nel nella nei nelle si come anche più "
    "sono essere ha hanno può deve".split()
)


@dataclass
class GenConfig:
    """Limits the generated pairs are held to.

    Attributes:
        min_passage_chars: shorter passages carry too little to ask about.
        max_passage_chars: longer ones are cut to a window at a paragraph
            boundary, so the prompt fits and the answer stays grounded in
            something the generator actually read closely.
        min_answer_chars / max_answer_chars: bounds on the answer.
        min_question_chars / max_question_chars: bounds on a `qa` question.
        min_italian_ratio: share of the answer's words that must be common
            Italian function words (Italian prose runs around 0.35-0.45).
    """

    min_passage_chars: int = 400
    max_passage_chars: int = 5000
    min_answer_chars: int = 150
    max_answer_chars: int = 3000
    min_question_chars: int = 15
    max_question_chars: int = 600
    min_italian_ratio: float = 0.18
    anonymiser: AnonymiserConfig = field(
        default_factory=lambda: AnonymiserConfig(use_ner=False)
    )


@dataclass
class Job:
    """One pair to generate: a passage, a task and a stable key for resuming."""

    key: str
    task: str
    passage: str
    source: str
    instruction_prefix: str = ""


class Rejected(ValueError):
    """A generation that must not become a training example; ``reason`` says why."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _whole_sentences(text: str) -> str:
    """Trim a leading and trailing sentence fragment.

    ``train.jsonl`` was cut into chunks for distillation, so a record can
    begin "oggetto dello scorporo catastale. 2.2 Con il quarto motivo…". The
    generator copes, but in the context tasks the passage IS the user's
    message, and a user pastes a text that begins at the beginning. Only a
    fragment is trimmed: a text starting with a capital or a digit is left
    alone, and nothing is cut if no sentence boundary is near.
    """
    if text and not (text[0].isupper() or text[0].isdigit()):
        m = re.search(r"[.;:!?]\s+(?=[A-ZÀ-Ý0-9])", text[:600])
        if m:
            text = text[m.end():]
    if text and text[-1] not in ".;:!?)»\"":
        cut = max(text.rfind(". "), text.rfind(".\n"))
        if cut > len(text) - 600:
            text = text[:cut + 1]
    return text.strip()


def passage_window(text: str, max_chars: int, rng: random.Random) -> str:
    """A window of at most ``max_chars`` that starts and ends at paragraph breaks.

    Rulings are long, and what matters in them — the reasoning — is rarely in
    the first few thousand characters, which are the parties and the history
    of the case. A seeded random window over paragraph starts samples the
    whole document instead of its header, reproducibly.
    """
    text = _whole_sentences(text.strip())
    if len(text) <= max_chars:
        return text
    starts = [0] + [m.end() for m in re.finditer(r"\n\s*\n", text)]
    starts = [s for s in starts if s < len(text) - max_chars // 2] or [0]
    start = rng.choice(starts)
    window = text[start:start + max_chars]
    cut = window.rfind("\n\n")
    if cut > max_chars // 2:
        window = window[:cut]
    return window.strip()


def make_jobs(
    records: list[dict],
    limit: int,
    seed: int = 0,
    cfg: GenConfig | None = None,
    weights: dict[str, float] | None = None,
) -> list[Job]:
    """Choose passages and tasks, deterministically for a given seed.

    Records are shuffled with ``seed`` before selecting, so stopping at any
    point leaves a random sample of the corpus, not its first N records —
    which are whatever source happened to be concatenated first.

    The key hashes the passage and the task, so rerunning with the same
    arguments regenerates the same jobs and the caller can skip those
    already done.
    """
    cfg = cfg or GenConfig()
    weights = weights or TASK_WEIGHTS
    tasks, probs = zip(*weights.items())
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)

    jobs: list[Job] = []
    for i in order:
        if len(jobs) >= limit:
            break
        rec = records[i]
        text = rec.get("text") or ""
        if len(text.strip()) < cfg.min_passage_chars:
            continue
        passage = passage_window(text, cfg.max_passage_chars, rng)
        if len(passage) < cfg.min_passage_chars:
            continue
        passage, _ = anonymize_text(passage, config=cfg.anonymiser)
        task = rng.choices(tasks, weights=probs)[0]
        key = hashlib.sha1(f"{task}\x00{passage}".encode()).hexdigest()[:20]
        prefix = rng.choice(_CONTEXT_INSTRUCTIONS[task]) if task in _CONTEXT_INSTRUCTIONS else ""
        source = rec.get("source") or rec.get("kind") or rec.get("source_id") or ""
        jobs.append(Job(key, task, passage, str(source), prefix))
    return jobs


def build_messages(job: Job) -> list[dict[str, str]]:
    """The conversation sent to the generator for one job."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _TASK_PROMPTS[job.task].format(passage=job.passage)},
    ]


def _extract_json(text: str) -> dict:
    """The JSON object in a generation, tolerating code fences and stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise Rejected("no_json")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise Rejected("bad_json", str(exc)) from None
    if not isinstance(obj, dict):
        raise Rejected("bad_json", "not an object")
    return obj


def italian_ratio(text: str) -> float:
    """Share of words that are common Italian function words."""
    words = re.findall(r"[a-zàèéìòù]+", text.lower())
    if not words:
        return 0.0
    return sum(w in _IT_STOPWORDS for w in words) / len(words)


def strip_citation_names(text: str) -> str:
    """Drop the party surname from Cassazione-style citations, keep the rest."""
    return RE_CITATION_NAME.sub(r"\1, \2", text)


def _check_clean(text: str, what: str, cfg: GenConfig) -> None:
    if RE_PLACEHOLDER.search(text):
        raise Rejected("placeholder", f"{what}: {RE_PLACEHOLDER.search(text).group()}")
    _, stats = anonymize_text(text, config=cfg.anonymiser)
    if stats.total():
        raise Rejected("personal_data", f"{what}: {stats.to_dict()}")


def parse_generation(raw: str, job: Job, cfg: GenConfig | None = None) -> dict:
    """Turn one raw generation into a training pair, or raise `Rejected`.

    Returns:
        ``{"instruction", "output", "task", "source", "key"}`` — the first
        two are what `eullm_forge.identity` trains on, the rest is
        provenance so any pair can be traced back and a bad batch removed.
    """
    cfg = cfg or GenConfig()
    if "<think>" in raw:
        raise Rejected("thinking")
    obj = _extract_json(raw)

    answer = strip_citation_names(str(obj.get("risposta", "")).strip())
    if not answer:
        raise Rejected("empty_answer")
    if not cfg.min_answer_chars <= len(answer) <= cfg.max_answer_chars:
        raise Rejected("answer_length", str(len(answer)))
    if italian_ratio(answer) < cfg.min_italian_ratio:
        raise Rejected("not_italian", f"{italian_ratio(answer):.2f}")
    _check_clean(answer, "answer", cfg)

    if job.task == "qa":
        question = str(obj.get("domanda", "")).strip()
        if not cfg.min_question_chars <= len(question) <= cfg.max_question_chars:
            raise Rejected("question_length", str(len(question)))
        if RE_TEXT_REFERENCE.search(question) or RE_TEXT_REFERENCE.search(answer):
            raise Rejected("refers_to_text")
        marker = case_specific(question)
        if marker:
            raise Rejected("case_specific", marker)
        _check_clean(question, "question", cfg)
        instruction = question
    else:
        instruction = f"{job.instruction_prefix}\n\n{job.passage}"

    return {
        "instruction": instruction,
        "output": answer,
        "task": job.task,
        "source": job.source,
        "key": job.key,
    }
