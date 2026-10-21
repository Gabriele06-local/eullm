"""Rewards a program can check, for open-book legal questions.

Three kinds of question have an answer that needs no judge:

* ``termine`` / ``termine_argomento``: the article states one deadline, and
  the answer must state it. Checked with the same keyword the exam uses
  (`norm_exam._deadline_keyword`), digits or words.
* ``inesistente``: the article asked about does not exist, and the prompt
  says so; the answer must say it does not exist instead of describing one.
* ``assente``: a question by topic whose article was taken out of the
  retrieved texts; the answer must say the texts do not contain it.

And three built on a false premise (`eullm_forge.datasets.premise_gen`):

* ``premessa_termine``: the question states a wrong deadline; the answer
  must say the premise is wrong and state the right one.
* ``premessa_inesistente``: the question takes for granted an article that
  is not there, with no note saying so; checked as ``inesistente``.
* ``premessa_abrogato``: the question treats a repealed article as in force;
  the answer must say it was repealed.
* ``premessa_vera``: the question states the right deadline; the answer must
  state it and must not call the premise wrong.

Each check is written against the ways a policy learns to game it, because
under RL it will:

* **Listing every number.** An answer naming five deadlines contains the
  right one. A deadline answer earns nothing if it names more than
  `MAX_DEADLINES` different deadlines.
* **Always abstaining.** "Non trovo la disposizione" is right for two of the
  four kinds and wrong for the other two, so it is only worth what it is
  worth on the abstention kinds. A deadline answer that refuses earns
  nothing even if a number slipped in.
* **Abstaining and then answering anyway.** An abstention that goes on to
  state a deadline is not an abstention: it earns nothing.
"""

from __future__ import annotations

import re

from ..eval.metrics import (
    ANY_DEADLINE,
    DEADLINE_UNITS,
    NUMBER_WORDS,
    keyword_coverage,
    number_of,
)

DEADLINE_TYPES = frozenset({"termine", "termine_argomento"})
ABSTAIN_TYPES = frozenset({"inesistente", "assente", "premessa_inesistente"})
PREMISE_TYPES = frozenset({"premessa_termine", "premessa_inesistente", "premessa_abrogato",
                           "premessa_vera"})
# Kinds no program can check: what an article provides is graded by a judge
# model (`eullm_forge.rl.judge_reward`), and `answer_reward` leaves them to it.
# "pratica": the practical questions of eval.practical, as GRPO prompts
# (make_practical_prompts.py), have no verifiable answer either.
JUDGED_TYPES = frozenset({"contenuto", "pratica"})

#: How many different deadlines a right answer may name: the one asked, and
#: one more for the article that also mentions, say, a notice period.
MAX_DEADLINES = 2

# Ways of saying "it is not there", for a missing article or for texts that
# do not hold the answer. Matched on the lowercased answer; accents kept,
# since the models write them.
_ABSTAIN = re.compile(
    r"non esiste|inesistent|non (?:è|e') (?:previst|presente|contenut|riportat)"
    r"|non (?:sono|risulta(?:no)?) (?:presenti|riportat|contenut)"
    # "Non ho trovato" is "non trovo" in the past: RAG Enterprise's abstention
    # and legal-it-8b's ("Nei dati disponibili non ho trovato la norma che
    # risponde alla domanda"). Missing here, 62 such answers to absent
    # articles counted as wrong in the no-judge column of 2026-10-11.
    r"|non trovo|non ho trovato|non (?:è|e') possibile (?:individuare|trovare|rispondere)"
    # The verb in either number, and the handful of ways of saying it that
    # models reach for. Only "non contengono" was here, so "non contiene" --
    # the same sentence, one text instead of many -- scored nothing, and
    # "non risulta alcuna disposizione", which is how the answer is usually
    # put, with it.
    r"|non (?:contiene|contengono|include|comprende|compare|riporta|riportano)"
    # ...with the missing thing as its object: "la norma non include i
    # contratti a termine" describes an article, and an inesistente answer that
    # invents one and says that is not abstaining.
    r"\s+(?:\w+\s+){0,2}?(?:l['’]\s*)?(?:(?:articol[oi]|norm[ae]|disposizion[ei]"
    r"|testi|testo|riferiment[oi]|fonte)\b|art\.)"
    r"|non (?:risulta|risultano) (?:alcun[ao]?|nessun[ao]?)\s+(?:articol[oi]|norm[ae]"
    r"|disposizion[ei]|riferiment[oi]|testo)\b|non risulta nulla nei testi"
    r"|non (?:si )?trova(?:no)? (?:nei|tra i) testi"
    # How the false-premise answers (premise_gen) say it, and models with them:
    # the article "non c'è", and "non mi risulta che esista" -- scored nothing
    # until 2026-10-20, so the generator's own right answers were worth 0.
    r"|non c['’]è\s+(?:l['’]\s*)?(?:articol[oi]|art\.|norm[ae]|disposizion[ei])"
    r"|non (?:mi )?risulta che esist"
    r"|non (?:posso|sono in grado di) (?:rispondere|indicare)",
)


# The subset that can only be a refusal. "Non è previsto" also opens right
# deadline answers ("non è prevista alcuna proroga: il termine è di 60
# giorni"), so it may not cost a deadline answer its reward.
#
# "Non esiste" opens the same sentence, and is the more common way to write
# it, so it is here only where what it denies is the article or the norm --
# the thing a deadline question presupposes. "Non esiste alcuna proroga" and
# "non esiste alcun termine perentorio" deny something else, and an answer
# that opens with one of them states the right deadline. Paying the two
# sentences differently for a synonym is what this avoids: the same answer
# scored 1.0 with "non è prevista" and 0.0 with "non esiste", so GRPO trained
# the phrasing out of the policy. What the set is for still holds -- denying
# the article and answering anyway earns nothing, in either order.
_NOT_THE_ARTICLE = (
    r"articol[oi]|art\.?|norma|disposizion[ei]|prescrizion[ei]|regola|testo"
    r"|riferiment[oi]|passaggio|fonte"
)
_REFUSAL = re.compile(
    r"inesistent|non trovo|non contengono"
    r"|non (?:è|e') possibile (?:individuare|trovare|rispondere)"
    r"|non (?:posso|sono in grado di) (?:rispondere|indicare)"
    # "non esiste" the article, or the article "non esiste" -- the two orders
    # Italian uses. A period may be crossed, so "L'art. 10 non esiste" is
    # matched and "l'articolo 10 è stato abrogato. Non esiste alcuna proroga"
    # is still read as the refusal it looks like.
    rf"|non esiste\s+(?:alcun[ao]?\s+|nessun[ao]?\s+)?(?:l['’]\s*)?(?:{_NOT_THE_ARTICLE})"
    rf"|\b(?:{_NOT_THE_ARTICLE})\s*(?:n\.\s*)?\d*[\w-]*[^.?!;\n\d]{{0,60}}?\bnon esiste\b",
    re.IGNORECASE,
)


# Any "<number> <unit>" in an answer (metrics.ANY_DEADLINE). The exam's own
# pattern wants the statute's phrasing ("entro", "decorsi"); answers say "il
# termine è di 60 giorni", and a count that misses those would let a list of
# numbers through.


def mentioned_deadlines(answer: str) -> set[tuple[int, str]]:
    """The distinct deadlines an answer names, digits or words."""
    found = set()
    for num, unit in ANY_DEADLINE.findall(answer):
        n = number_of(num)
        if n:
            found.add((n, DEADLINE_UNITS[unit.lower()]))
    return found


def _keyword_deadline(keyword: str) -> tuple[int, str] | None:
    """The deadline a keyword from the exam stands for, or None.

    The keyword comes from `norm_exam._deadline_keyword`, which builds it from
    the article's own number and unit ("20 giorni|venti giorni"), so it is
    read back with the same pattern used on an answer. None means the
    keyword is not a deadline in this shape, and the caller falls back to
    the coverage test rather than assuming a number.
    """
    for num, unit in ANY_DEADLINE.findall(keyword):
        n = number_of(num)
        if n:
            return n, DEADLINE_UNITS[unit.lower()]
    return None


def abstains(answer: str) -> bool:
    """Whether the answer says the article or the answer is not there."""
    return bool(_ABSTAIN.search(answer.lower()))


def refuses(answer: str) -> bool:
    """Whether the answer declines to answer, in words that mean nothing else."""
    return bool(_REFUSAL.search(answer.lower()))


# Ways of saying the question's premise is wrong. Required of a corrected
# deadline: "Il termine è di 60 giorni" after "mi confermi che è di 30?" gives
# the right number and still lets the wrong one stand.
#
# The first version knew only "errata", "in realtà", "no" and "non di 30
# giorni", and the samples of 2026-10-20 showed right corrections it scored
# 0: "il termine di 20 giorni non è previsto dal codice civile", "non compare
# alcun riferimento a un termine di 60 giorni", "60 giorni, non 30".
_UNIT = r"(?:giorni|giorno|mesi|mese|anni|anno|ore)"
# Without "un", "uno", "una": ", non una semplice facoltà" denies nothing.
_NUM = r"(?:\d+|" + "|".join(sorted(set(NUMBER_WORDS) - {"un", "uno", "una"}, key=len,
                                     reverse=True)) + r")"
_CORRECTS = re.compile(
    r"\berrat[ao]\b|\binesatt[ao]\b|non (?:è|e') (?:corrett|esatt|cosi|così)|in realtà"
    rf"|^\W*no\b|\bnon (?:di|è di|sono)\s+{_NUM}\s+{_UNIT}\b"
    # "60 giorni, non 30": the wrong number denied after the right one
    rf"|,\s*non\s+(?:di\s+|a\s+)?{_NUM}\b"
    # "il termine di 20 giorni non è previsto", "non prevede un termine di 20 giorni"
    r"|non (?:è|e') (?:previst|stabilit|fissat|indicat)"
    rf"|non (?:prevede|stabilisce|fissa|indica)\s+(?:un |il |alcun )?(?:termine )?(?:di )?"
    rf"{_NUM}\s+{_UNIT}\b"
    r"|non compare alcun riferimento|non (?:vi è|c['’]è) alcun riferimento",
    re.IGNORECASE)


def corrects(answer: str) -> bool:
    """Whether the answer says the premise of the question is wrong."""
    return bool(_CORRECTS.search(answer.strip()))


def score_answer(answer: str, tipo: str, keywords: list[str] | None = None) -> float:
    """1.0 for a verifiably right answer, 0.0 otherwise.

    Raises ValueError for a kind it cannot check: a question that silently
    scored 0 whatever the model said would only teach it noise.
    """
    if tipo == "premessa_termine":
        if not corrects(answer):
            return 0.0
        return score_answer(answer, "termine", keywords)
    if tipo == "premessa_vera":
        if denies_premise(answer):
            return 0.0
        return score_answer(answer, "termine", keywords)
    if tipo == "premessa_abrogato":
        return 1.0 if "abrogat" in answer.lower() and not mentioned_deadlines(answer) else 0.0
    if tipo in DEADLINE_TYPES:
        if not keywords:
            raise ValueError(f"a {tipo} question needs its deadline keyword")
        if refuses(answer) or len(mentioned_deadlines(answer)) > MAX_DEADLINES:
            return 0.0
        # Compared as a number, not as a substring. The keyword is a
        # normalised substring test, so "20 giorni" is inside "120 giorni"
        # and "sessanta giorni" inside "centosessanta giorni": every wrong
        # deadline in the corpus that ends in the right one scored 1.0. The
        # number the answer names is already parsed by the check above, so
        # the question is whether it is the right number.
        wanted = [_keyword_deadline(kw) for kw in keywords]
        if all(w is not None for w in wanted):
            named = mentioned_deadlines(answer)
            return 1.0 if all(w in named for w in wanted) else 0.0
        # A keyword that is not a deadline in this shape: fall back to the
        # coverage test rather than invent a number to check it against.
        return 1.0 if keyword_coverage(answer, keywords) == 1.0 else 0.0
    if tipo in ABSTAIN_TYPES:
        return 1.0 if abstains(answer) and not mentioned_deadlines(answer) else 0.0
    raise ValueError(f"no verifiable reward for questions of type {tipo!r}")


def _text(completion) -> str:
    """A completion as TRL hands it: a string, or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") for m in completion if isinstance(m, dict))


def answer_reward(completions, tipo, keywords, **_) -> list[float | None]:
    """TRL reward function: one score per completion.

    TRL passes the dataset's other columns as keyword lists aligned with the
    completions, so ``tipo`` and ``keywords`` come from the prompts file
    written by make_grpo_prompts.py. A judged kind gets None, which TRL reads
    as "this function does not score this completion": the judge reward
    does.
    """
    return [None if t in JUDGED_TYPES else score_answer(_text(c), t, k)
            for c, t, k in zip(completions, tipo, keywords)]
