"""Stage-3 pairs for the task the product does: answer with the norm in hand.

The held-out exam of 2026-09-27 settled what a 4B model can and cannot do.
Asked from memory, every model tried — ours and Qwen's own instruct model —
got about 3% of 233 questions right; with the retrieved articles in the
prompt, 46-60%. So legal-it answers from the text it is given. What our
models did worst was exactly that: restating an article they had just read
(v0.2 44/86, Qwen3-4B-Instruct 67/86), and a question about an article that
does not exist got an invented answer 18 times out of 18.

The existing stage-3 pairs (`instruct_gen`) are mostly closed-book questions
and summaries. These are the other kind, in the exact prompt the evaluation
and the engine use (`open_book_prompt`, same retrieval):

* ``grounded`` — the teacher reads ONE article and writes a realistic
  question it answers and the answer, from that text only, citing it. Half
  the questions name the article, half ask by topic, as people do. The
  training prompt then carries what retrieval returns for that question,
  with the article put in if retrieval missed it, so the student learns to
  find the relevant text among others and not just to copy the first block.
* ``missing`` — a question about an article number the collection does not
  hold. The prompt says so (`missing_article_note`) and the answer says it
  cannot describe an article that is not there. No teacher needed.

Articles in the held-out exam are excluded by (code, number), read from the
exam file and never printed, so training on these pairs cannot leak into the
number the exam reports.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..eval.norm_exam import CODE_LABELS, articles_from_records
from ..eval.retrieval import NormIndex, open_book_prompt, record_articles
from .instruct_gen import GenConfig, Rejected, _check_clean, _extract_json, italian_ratio

TEACHER_SYSTEM = (
    "Sei un giurista italiano esperto e scrupoloso. Prepari esempi per addestrare "
    "un assistente che risponde a domande di diritto leggendo i testi di legge. "
    "Scrivi in italiano corretto e professionale. Non inventare nulla che non sia "
    "nel testo che ti viene dato."
)

TEACHER_TASK = """Ecco il testo di un articolo ({where}).

--- TESTO ---
{text}
--- FINE ---

Scrivi UNA domanda realistica, come la porrebbe un avvocato o un cittadino, a cui
questo articolo risponde, e la risposta.
- La domanda {naming}
- La risposta si basa SOLO sul testo sopra, è precisa (termini, soggetti,
  condizioni come scritti nel testo), cita l'articolo (per esempio "Secondo
  l'art. {number} {of}, ...") ed è lunga da 2 a 6 frasi.
- Non aggiungere nulla che il testo non dica.

Rispondi solo con un oggetto JSON: {{"domanda": "...", "risposta": "..."}}"""

NAMING = {
    True: "cita il numero dell'articolo e la legge o il codice ({of}).",
    False: "NON cita il numero dell'articolo: chiede dell'argomento, e nomina il "
           "codice o la legge ({of}).",
}

# More than one pair per article, each asking about something else: the
# teacher is greedy, so the same prompt would give the same pair. Variant 0 is
# the original prompt (and key), so a run with one pair per article is the
# run of 2026-09-27 exactly.
FOCUS = [
    "",
    "Questa volta la domanda riguarda un termine, una condizione o un'eccezione "
    "prevista dall'articolo (se l'articolo non ne prevede, un altro aspetto preciso).",
    "Questa volta la domanda descrive un caso concreto, come lo racconterebbe un "
    "cittadino o un'impresa, a cui l'articolo si applica.",
    "Questa volta la domanda riguarda chi: il soggetto obbligato, competente o "
    "tutelato secondo l'articolo, e con quali effetti.",
]

MISSING_QUESTIONS = [
    "Che cosa prevede l'art. {n} {of}?",
    "Mi spieghi cosa stabilisce l'art. {n} {of}?",
    "Quali sono i termini previsti dall'art. {n} {of}?",
    "L'art. {n} {of} si applica anche ai contratti tra privati?",
]
MISSING_ANSWER = (
    "Nei testi normativi a mia disposizione non compare l'art. {n} {of}, quindi non "
    "posso dirti che cosa prevede: potrebbe non esistere, oppure il riferimento potrebbe "
    "essere errato. Ti consiglio di verificare il numero dell'articolo e la fonte; se mi "
    "indichi l'argomento, posso cercare le norme pertinenti."
)


@dataclass
class OpenBookJob:
    """One pair to make: a grounded question about an article, or a missing one."""

    key: str
    kind: str                  # "grounded" | "missing"
    code: str
    number: str
    text: str = ""
    named: bool = True
    meta: dict = field(default_factory=dict)


def exam_exclusions(items) -> set[tuple[str, str]]:
    """(code, article) of every item of the held-out exam, to be left out."""
    out = set()
    for it in items:
        md = it.metadata if hasattr(it, "metadata") else it.get("metadata", {})
        if md.get("code") and md.get("articolo"):
            out.add((md["code"], str(md["articolo"])))
    return out


def make_openbook_jobs(index: NormIndex, limit: int, *, seed: int = 0,
                       exclude: set[tuple[str, str]] = frozenset(),
                       missing_share: float = 0.15, max_chars: int = 3000,
                       per_article: int = 1, shard: tuple[int, int] = (0, 1)
                       ) -> list[OpenBookJob]:
    """Draw ``limit`` jobs, ``missing_share`` of them about absent articles.

    ``per_article`` asks up to that many different questions of each drawn
    article (see `FOCUS`); ``shard=(k, n)`` keeps every n-th job from the k-th,
    so n generation jobs can share one draw without writing the same pair.
    """
    if not 1 <= per_article <= len(FOCUS):
        raise ValueError(f"per_article must be 1..{len(FOCUS)}")
    rng = random.Random(seed)
    arts = [a for (code, num), a in sorted(articles_from_records(index.records).items())
            if code in CODE_LABELS and (code, num) not in exclude
            and len(a.text) >= 150 and "abrogat" not in a.text[:300].lower()]
    by_code: dict[str, list] = {}
    for a in arts:
        by_code.setdefault(a.code, []).append(a)
    n_missing = int(round(limit * missing_share))
    n_articles = -(-(limit - n_missing) // per_article)
    grounded = rng.sample(arts, min(n_articles, len(arts)))
    jobs = []
    for a in grounded:
        for v in range(per_article):
            jobs.append(OpenBookJob(
                key=f"ob-g-{a.code}-{a.number}" + (f"-v{v}" if v else ""), kind="grounded",
                code=a.code, number=a.number, text=a.text[:max_chars],
                named=rng.random() < 0.5, meta={"focus": v} if v else {}))
    codes = sorted(by_code)
    for i in range(n_missing):
        code = codes[i % len(codes)]
        last = max(int(re.match(r"\d+", a.number).group()) for a in by_code[code])
        while True:
            fake = str(last + rng.randint(30, 2000))
            if (code, fake) not in exclude:
                break
        jobs.append(OpenBookJob(key=f"ob-m-{code}-{fake}-{i}", kind="missing", code=code,
                                number=fake, meta={"template": i % len(MISSING_QUESTIONS)}))
    rng.shuffle(jobs)
    k, n = shard
    return [j for i, j in enumerate(jobs) if i % n == k]


def build_messages(job: OpenBookJob) -> list[dict[str, str]]:
    """The teacher prompt for a grounded job."""
    of, _ = CODE_LABELS[job.code]
    where = f"art. {job.number} {of}"
    task = TEACHER_TASK.format(where=where, text=job.text, number=job.number, of=of,
                               naming=NAMING[job.named].format(of=of))
    focus = FOCUS[job.meta.get("focus", 0)]
    if focus:
        task += "\n\n" + focus
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": task},
    ]


def _context(index: NormIndex, question: str, job: OpenBookJob, k: int,
             rng: random.Random) -> tuple[list[dict], str]:
    """What retrieval returns for the question, with the article ensured present."""
    found = index.search(question, k)
    note = index.missing_article_note(question)
    if job.kind == "grounded":
        mine = [r for r in index.records if r.get("code") == job.code
                and job.number in record_articles(r)]
        if mine and not any(r in found for r in mine):
            found = found[:k - 1]
            found.insert(rng.randrange(len(found) + 1), mine[0])
    return found, note


def missing_pair(job: OpenBookJob, index: NormIndex, k: int = 3) -> dict:
    """A pair about an article the collection does not hold; no teacher needed."""
    of, _ = CODE_LABELS[job.code]
    q = MISSING_QUESTIONS[job.meta.get("template", 0)].format(n=job.number, of=of)
    found, note = _context(index, q, job, k, random.Random(job.key))
    if not note:
        raise Rejected("article_exists", f"{job.code} art. {job.number}")
    return {"instruction": open_book_prompt(q, found, note=note),
            "output": MISSING_ANSWER.format(n=job.number, of=of),
            "task": "openbook_missing", "source": job.code, "key": job.key}


def parse_openbook(raw: str, job: OpenBookJob, index: NormIndex, k: int = 3,
                   cfg: GenConfig | None = None) -> dict:
    """Turn a teacher generation into a grounded pair, or raise `Rejected`."""
    cfg = cfg or GenConfig()
    if "<think>" in raw:
        raise Rejected("thinking")
    obj = _extract_json(raw)
    q = " ".join(str(obj.get("domanda", "")).split())
    a = " ".join(str(obj.get("risposta", "")).split())
    if not 15 <= len(q) <= 400:
        raise Rejected("question_length")
    if not 80 <= len(a) <= 1500:
        raise Rejected("answer_length")
    if italian_ratio(a) < cfg.min_italian_ratio:
        raise Rejected("not_italian")
    names_it = re.search(rf"\bart(?:icolo|\.)?\s*{re.escape(job.number.split('-')[0])}\b",
                         q, re.IGNORECASE) is not None
    if job.named and not names_it:
        raise Rejected("question_not_naming")
    if not job.named and names_it:
        raise Rejected("question_names_article")
    if not re.search(rf"\b{re.escape(job.number.split('-')[0])}\b", a):
        raise Rejected("answer_not_citing")
    _check_clean(q, "question", cfg)
    _check_clean(a, "answer", cfg)
    found, note = _context(index, q, job, k, random.Random(job.key))
    return {"instruction": open_book_prompt(q, found, note=note), "output": a,
            "task": "openbook_grounded", "source": job.code, "key": job.key,
            "named": job.named}
