# legal-it — what was tried, what it gave, and why the course changed

> Status: living record · Last updated: 2026-10-18

The record of every legal-it training line, from the first distillation on
Cassazione rulings to the GRPO rounds of October. The model cards say how
the *published* models were made; this page says what else was tried and
why it was set aside, so a direction is not dropped and forgotten, or
retried without knowing how it went. Numbers come from the exams on
Leonardo; each section names its source.

## The exams

| Exam | Size | What it measures |
|---|---|---|
| Held-out statute exam | 429 questions | the published models' numbers (model cards); used once per release candidate |
| Development statute exam (`devbig`) | 472 questions | content, deadlines, nonexistent articles, asked open book with hybrid retrieval (BM25 + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B); graded by Qwen3-30B-A3B-Instruct-2507 under rubric v3, and the deadline and absent-article questions (190) also by a program |
| The same, without the note (`-hybnn`) | 472 | the prompt no longer says "nella raccolta normativa non è presente art. N": RAG products built on the models do not write that note, so this is the product condition (`legal_eval.py --no-note`, 2026-10-16) |
| Practical exam | 145 dev / 70 held-out | a practical question needing a group of linked articles, written by Qwen3.6-27B (`eullm_forge.eval.practical`) |
| Case-law exam | 1,200 questions | answering from Consiglio di Stato rulings and citing the right one (`sbatch_cds_exam.slurm`) |

All numbers below are Q4_K_M GGUFs through llama-server unless marked bf16.

## 1. September: distillation on the Cassazione corpus

**What.** `docs/legal-it-4b-strategy.md` and ADR-001. A corpus of 1.13 M
chunks, about 700 M tokens: Cassazione civil and criminal judgments
2021-2026 from ItalGiure (`docs/corpus-acquisition.md`), pseudonymised by
`anonymize.py` (rounds 1-6), deduplicated, plus the codes and the
Constitution. Teacher Qwen3-30B-A3B-Base, student Qwen3-4B-Base, logit
distillation on Leonardo (`legal_it`, `legal_it_v12`; an 8B run in
`legal_it_8b`). Then stage 3 on the distilled student (domain pairs plus
identity): `legal-it-4b` v0.2.

**What it gave.** The student learned the language of the corpus: on 40
held-out chunks of 512 tokens its perplexity at step 12,600 was 3.19
against 6.65 for Qwen3-4B-Base (`legal_it_v12/exports/perplexity.csv`).

**Why the course changed (2026-09-26/27, commits 749a0ee and 689ec50).**
Measured on questions, not on perplexity:

- v0.2 matched Qwen3-4B-Instruct on the keyword score, but read by a person
  four answers in ten were wrong: the TAR deadline given as 30 days (it is
  60), the ricorso straordinario as 30 (it is 120), art. 27 Cost. described
  as art. 25, the AI Act disclosure duty backwards.
- On the held-out exam a 4B model answered about 3% of 233 questions from
  memory, whoever trained it, and 46-60% with the retrieved norms in the
  prompt.
- At exactly that, answering from the text in the prompt, the distilled
  model did worst: restating an article just read, 44 of 86 against 67 for
  Qwen3-4B-Instruct; a question about a nonexistent article got an
  invented answer 18 times out of 18.

So legal-it became a model that answers **from the texts retrieval gives
it**, trained from an instruct model on open-book pairs (section 2). The
distillation did not fail at what it measured; it measured the wrong
thing for the product. Two more reasons weighed on publishing a model
trained on rulings: the anonymiser had once reported a clean run and still
left 59 codici fiscali in the training text (strategy, risks table), and a
published model that memorised a ruling cannot be recalled.

**What is still there.** The checkpoints and exports on `$WORK`
(`legal_it/exports`, `legal_it_v12/exports`, `legal_it_8b`): keep them
until section 6 is decided.

**What was never measured.** Whether the corpus helps on case law or on
practical questions, which did not exist as exams then; and the distilled
student with the open-book recipe on top of it. Every comparison above is
on statute questions, where knowing the rulings cannot help.

## 2. October: the open-book recipe, and the published models

From an instruct model: open-book SFT on 18,972 pairs (85% written by
Qwen3-30B-A3B-Instruct-2507 from one article each, 15% about nonexistent
articles), then GRPO with programmatic rewards (deadlines, abstention).
Statutes only, no rulings. Published 2026-10-04:

| | Base | Held-out (429) | Base model, same retrieval |
|---|---|---|---|
| `legal-it-8b` | Ministral-3-8B-Instruct-2512 (Mistral AI) | 89.0% | 80.9% |
| `legal-it-4b` | Qwen3-4B-Instruct-2507 | 85.1% | 77.6% |

Ministral-3-14B with the same recipe scored 436 on `devbig`, the same as
the 8B (436 bf16, 437 Q4): on this exam a larger model of the same family
added nothing.

**What it cost** (A100 GPU-hours, from `sacct` for 2026-09-25 to 10-05):

| | Job | Wall time | GPU-hours |
|---|---|---|---|
| `legal-it-8b`, SFT | `eullm-s3-ministral-v04` (59092257) | 4 h 09 m on 1 A100 | ~4.2 |
| `legal-it-8b`, GRPO | `eullm-grpo-ministral-v04` (59125822) | 1 h 33 m on 3 A100 | ~4.6 |
| `legal-it-8b`, merge and GGUF | serial partition | 16 m | CPU only |
| Ministral-3-14B, SFT + GRPO | | | 5.8 + 6.4 |
| Shared: open-book pairs written by the teacher (`gen-openbook`) | | | 40.2 |
| Shared: exams (`exam`) and their grading (`judge`) | | | 24.3 + 63.4 |
| `gen-instruct` | | | 22.1 |

The model itself is cheap: **about 9 GPU-hours** of training for the
published 8B. The data and the measuring, shared by every model of the
line, cost far more: the teacher writing the pairs took 40 GPU-hours, the
exams and their grading 88. `gen-instruct`
is the September line's (section 1): it wrote the `instruct-pairs` files
stage 3 used on the distilled student (commit 7040d91). The published 8B's
SFT loaded `openbook-v04.jsonl` alone, 18,972 pairs plus 180 identity
examples (job 59092257's log: `--no-format-tokens --lr 1e-4 --epochs 1
--max-length 4096 --batch-size 1 --grad-accum 16`).

## 3. October: the 27B teacher and the Qwen3.5 students

**What.** Privileged-context distillation for the case-law skill (commit
ba6decf, `opd_train.py`, from 2026-10-06): the student answers with the
passages the case-law index retrieves, the teacher reads the same question
with the source ruling in front, and the student is moved towards the
teacher's distributions at each answer token (reverse KL). For the Qwen3.5
students the teacher was **Qwen3.8-27B** (`opd-q38t-q35-*`). Ministral
was distilled from itself: no available teacher shared its tokenizer.
Then GRPO rounds on top, with the judge as reward (`submit_grpo_round.sh`):
`jp` (practical prompts), `jp2` (a second round on `jp`), `jpa` (with
abstention prompts), `jpn` (the same with the note removed from 70% of the
prompts), `jp3n` and `jpn2` (one more round each).

**Development statute exam (472), correct, with / without the note:**

| Model | With | Without | Program-graded (190), with / without |
|---|---|---|---|
| **legal-it-8b** (published) | 437 | **440** | 183 / 183 |
| **legal-it-4b** (published) | 422 | **393** | 176 / **143** |
| OPD 9B, the start | 403 | 412 | 170 / 173 |
| jp-9b | 437 | 436 | 178 / 181 |
| jp2-9b | **447** | not yet | 177 |
| jp3n-9b | 439 | 437 | 175 / 175 |
| jpn-9b | 434 | 431 | 176 / 176 |
| OPD 4B, the start | 377 | — | 166 |
| jpn-4b | 390 | **427** | 176 / 176 |
| jpn2-4b | 410 | **426** | 177 / 175 |

**Paired (McNemar), without the note:**

- against legal-it-4b: jpn-4b +34 (59:25, p < 0.001), jpn2-4b +33 (56:23,
  p < 0.001), significant strictly and leniently. Almost all of it on
  nonexistent articles (36:2 and 35:0); content even (15:12, 14:11);
  deadlines slightly worse (4:8, 4:9). **The published 4B depends on the
  note** (393 without, 143 of 190 program-graded): without it, it answers
  about articles that are not there. The new 4B does not.
- against legal-it-8b: jp −4 (p 0.65), jp3n −3 (p 0.76), jpn −9 (p 0.24),
  and leniently the 8B is ahead (p 0.002-0.023). **No Qwen3.5-9B beats the
  Mistral 8B.**

**Practical exam (145), correct:** jp2-9b 82, jp3n-9b 81, jpn2-4b 81,
jpa-9b 75, jpn-4b 75, jp-9b 73, jpn-9b 67, OPD 4B 67, OPD 9B 62. The
published models' run is queued.

**Case-law exam (1,200), source ruling cited:** 0.348-0.362 for every
round, with the source ruling retrieved in 0.382 of the questions: the
models cite it about nine times in ten when it is there. **Case law is
held back by retrieval, not by the models.**

**What it says.** The 27B start (403) was far below where the published
models end (437); GRPO did most of the work on the Qwen3.5 line (+30 to
+44). A third round on `jp2` (`jp3n`) did not improve it:
its groups were often all graded the same, nothing left to learn from.

**Why Qwen3.5 may be underused (open).** The strategy noted it in
September (`legal-it-4b-strategy.md`, alternatives table): Qwen3.5's
hybrid architecture makes logit distillation noisier and its tooling
younger. Besides that: it is a reasoning model and every run here keeps
reasoning off; it never had the open-book SFT the published models had;
and `devbig` is near its ceiling (93%). Each is a measurement, not a
belief: reasoning on (needs a `legal_eval.py` option), the same SFT on
Qwen3.5, bf16 against Q4.

## 4. The release candidate

`jpn2-4b` (or `jpn3-4b`, in training) against `legal-it-4b`, without the
note, on the held-out exam, once. If it holds, it replaces `legal-it-4b`:
same content, and it says when an article is not there.

## 5. Teachers, summary

| Teacher | Used for | Published? |
|---|---|---|
| Qwen3-30B-A3B-Base | logit distillation on the Cassazione corpus (September) | no (v0.2) |
| Qwen3-30B-A3B-Instruct-2507 | writes the open-book SFT pairs; the exam judge | yes, as data writer |
| Qwen3.8-27B | privileged-context OPD of Qwen3.5-4B/9B (case-law skill) | no |
| Qwen3.6-27B | writes the practical exam and the practical GRPO prompts | as data writer |

## 6. The Cassazione, again

The corpus is not wasted; what it was used for was the wrong half of the
job. What it can still give, in order of likely return:

1. **Retrieval over rulings.** The case-law exam is capped by retrieval
   (0.382): an index, embedder and reranker that find the right ruling
   move every model at once, and train nothing on personal data.
2. **Open-book case-law training.** As the Consiglio di Stato cards do
   (`export_clean_cards.py`): questions on rulings, the rulings in the
   prompt, answers that cite them, with made-up references and only
   principles of law, then SFT and GRPO on `legal-it-8b`. It teaches the
   skill the product uses, reading rulings, not remembering them.
3. **Knowledge: the distilled corpus model under the open-book recipe.**
   The experiment never run in section 1: continued training on the
   corpus, then the same SFT and GRPO, measured on the practical and
   case-law exams as well as the statutes. Not on top of a published
   model without the SFT and GRPO after it: v0.2 is what continued
   training alone does to the reading skill.

Any model trained on ruling text goes through the memorisation check
(roadmap phase 3) before it is published.

## 7. An external test of legal-it-8b (2026-10-18, from reddit.com)

A legal-tech team tested the published `legal-it-8b` Q4_K_M and sent a
report, after a post about the model on reddit.com. Criminal law and
procedure only; 11 questions in the model card's format (3 retrieved texts,
the missing-article note, temperature 0, 400 tokens), and 5 inside their own
pipeline with 50,000-80,000-token prompts, far from the training format.
Small sample: signals, not rates.

**What held.** It finds the right text among similar ones, also in very
long prompts, and restates its principle faithfully; it corrects a numeric
premise when the right figure is in the text (the penalty for fraud, the
deadline for an appeal); the nonexistent-article answer with the note was
"exemplary"; clean Italian; it fits a 16 GB T4 with 86,000 tokens.

**What failed, and what answers it:**

| Failure | Example | Answer | Where |
|---|---|---|---|
| A false premise accepted with the right text in the prompt | "after the Cartabia reform art. 624-bis c.p. is prosecuted on complaint: by when?" (it is not; the reform concerns art. 624) | false-premise questions written from the statute text, rewarded only when the answer corrects the premise | next GRPO round |
| A wrong ruling number let through | "Sezioni Unite n. 18474/2025" for a principle another ruling in front of it states | `caselaw_wrong_cite` pairs | Cassazione SFT (`make_caselaw_sft.py`) |
| No abstention when the texts do not answer | the expected ruling was not retrieved; the answer was built on another one | absent-context pairs (statutes: `openbook-v05`; case law: `caselaw_absent`); the jpn rounds without the note | Cassazione SFT; 4B candidate |
| Citations that cannot be checked | none of 11 answers used [1]-[3]; the wrong passage, or the operative part instead of the reasoning | every case-law citation ends with its passage number `[n]` | Cassazione SFT |
| Content from memory | art. 360 c.p.c. in a criminal-procedure answer; criteria not in the texts | a reward for citing only articles and rulings present in the prompt (`check_cited_articles.py` as a reward) | next GRPO round |
| Reasoning | art. 444 c.p.p. criterion wrong; art. 4 l. 110/1975 and art. 699 c.p. confused; facts of a case read as legal requirements; contradictions | multi-article practical questions; a stronger teacher (open: reasoning on, Mistral teacher) | open |
| Multi-part questions, the reasoning part dropped | the boundary between two offences, how to frame the appeal | practical questions with a rubric per part | next GRPO round |
| Fines in lire | Normattiva keeps the original lire amounts (art. 640 c.p.) and the model reports them as current | flag lire amounts at indexing time, and train answers to say they are original amounts to be converted | corpus and product |
| A hidden system prompt | with llama.cpp `--jinja` and no system message, Ministral's chat template inserts its default system prompt (~525 tokens, "Le Chat") | check what the training template did, then ship a template without it, or with a short one of our own, trained and served alike | packaging |

Their tests were all criminal; our case-law training is civil only, since
criminal rulings never leave the cluster (section 6). Abstract principles of
criminal law hold no personal data, but whether their cards may be trained
on is a decision still to be taken.

## 8. 2026-10-20: teachers measured, false premises, and what a long SFT undoes

**The candidate teachers on the traps.** The 36 LexIA trap questions
(section 7; the set itself stays private and is never trained on), asked
without the note, as the product asks, graded by the judge:

| Model | Right | Premise corrected (of 18) | Abstained (of 9) | Answered (of 9) |
|---|---|---|---|---|
| Ministral-3-8B, untrained | 6 | 2 | 0 | 4 |
| Ministral-3-8B-Reasoning, reasoning on | 4 | 0 | 1 | 3 |
| Qwen3.6-27B, reasoning on | 10 | 5 | 0 | 5 |
| Qwen3.6-27B | 16 | 6 | 4 | 6 |
| legal-it-8b (published) | 14 | 5 | 5 | 4 |
| legal-it-8b, our system prompt | 16 | 6 | 6 | 4 |

No teacher measured corrects more than 6 false premises of 18, so
distilling one cannot teach it; free reasoning made both reasoning models
worse. On the practical exam the published 8B ties the best students once
"partial" counts as right (jp2-9b 19:18, p=1.0): its answers are half as
long (529 characters against 1,100-1,400), and the judge sides with the
longer answer in two disagreements out of three.

**Our system prompt costs nothing.** The published 8B asked with
`system-prompt.txt` instead of Ministral's "Le Chat" default: devbig 437 and
437 with the note (9:9), 437 and 440 without (0:3, p=0.25), practical 6:5,
LexIA 15 and 16 of 36. The GGUF can be repackaged with it; training uses it
from now on (`stage3_sft.py --default-system-file`).

**False premises, built without a teacher** (`make_premise_pairs.py`): a
wrong deadline, a nonexistent article, a repealed one taken for in force,
from the statute text, with no note, answered in checked steps with the
passage quoted. 1,060 questions; 530 for SFT, 530 for GRPO. The published
8B, 8 samples each (`sample_answers.py`, graded by program):

| Premise | Questions | Pass rate | Always wrong | Mixed | Always right |
|---|---|---|---|---|---|
| nonexistent article | 289 | 0.95 | 8 | 10 | 271 |
| repealed article | 158 | 0.31 | 48 | 101 | 9 |
| wrong deadline | 83 | 0.40 | 27 | 47 | 9 |

Two lessons from reading the answers: the first reward missed right
corrections ("il termine di 20 giorni non è previsto", "60 giorni, non 30")
and scored the deadline kind 0.17 instead of 0.40, so a reward is checked on
real answers before it trains anything; and some "wrong" answers deny an
article that is in the prompt ("non compare l'art. 645 c.p.p."), the
over-abstention the earlier GRPO rounds on nonexistent articles taught.
From the same samples: 158 questions with mixed answers for GRPO (the
rest carry no gradient), 158 DPO pairs and 249 correction exercises, the
model's own right and wrong answers (`make_correction_pairs.py`).

**Citations by program.** `cite_openbook_pairs.py` added the passage number
and a literal quotation to 16,116 of 16,127 grounded statute pairs
(`openbook-v06`), the two things the LexIA report found missing.

**What a long SFT on the published model undid.** The Cassazione SFT
(section 6, item 2: 75,352 case-law pairs and 20,572 statute pairs, 6,007
steps at lr 1e-4) started from the published `legal-it-8b`:

| Exam | Published | After the Cassazione SFT |
|---|---|---|
| devbig with the note | 437 | 415 (10:32, p=0.001) |
| devbig without the note | 440 | 413 (3:30, p<0.001; lenient p=0.002) |
| practical (correct + partial) | 60 + 21 | 57 + 9 (lenient 9:24, p=0.014) |
| LexIA traps without the note | 14 | 4 |

Every LexIA answer was the same sentence, word for word: the fixed answer
of the absent-article pairs ("Tra i testi normativi riportati non trovo la
disposizione…"). Fixed-wording refusals were about 14% of that SFT (1,600
statute pairs and some 11,000 `caselaw_absent`); without the note the model
took the refusal for the safe answer, by-topic deadline questions included
(0:9), since those are the shape of the absent pairs. Two rules follow:

1. a long SFT on a model GRPO has shaped undoes the GRPO; what is added on
   top is short, and GRPO follows it;
2. fixed-wording refusals stay a small share of any SFT mix (the round below
   keeps them near 6%), and a model is examined without the note before it
   is built on.

The round queued that evening, from the published model: a short SFT
(about 7,500 rows: 4,000 cited statute pairs, 2,000 case-law pairs, the
false premises in two formats, the correction exercises) in two arms that
differ only in the premise answers' format, three checked steps or one
paragraph; then GRPO on the mixed-answer questions; every model examined
on devbig with and without the note, the practical exam and the LexIA
traps.

## 9. 2026-10-21/22: what moved the traps, what saturated, and why

**Mistral Small 4 as a teacher.** Measured on JUPITER on the same 36 traps,
answers graded on Leonardo by the same judge (the answer files stay private):
10 of 36 at 400 tokens, 11 at 1,000, 12 with reasoning on, against 16 for
Qwen3.6-27B. It cites articles that are not among the texts, and the judge
marks most of those citations wrong (4 of 6 on the traps, 12 of 17 on the
practical exam with reasoning). Not a better teacher for the traps; its
practical-exam comparison waits for judgements cancelled by mistake on
10-20 and queued again.

**The rounds after r2b, on the traps without the note** (no system prompt,
so the reference is the published model's 14):

| Model | Right | Corrected (of 18) | Abstained (of 9) | Answered (of 9) |
|---|---|---|---|---|
| legal-it-8b (published) | 14 | 5 | 5 | 4 |
| r2b-steps + GRPO | 15 | 7 | 6 | 2 |
| r2d, + true-premise twins | 14 | 6 | 6 | 2 |
| r2e, + practical GRPO | 15 | 7 | 5 | 3 |
| r2f, + penal premises | 15 | 6 | 7 | 2 |
| jpn4 / jpn6 / jpn7 (Qwen3.5-4B) | 6 / 7 / 7 | 1-2 | 4-5 | 0-2 |

No round reaches the promotion rule (20 of 36); the gains are one question,
inside the noise of 36, and every 8B round answers one or two plain
questions fewer than the published model: the over-suspicion is still
there. The 4B line corrects almost no premise.

**Saturation.** The GRPO rounds on generated premises stopped by themselves
(20 steps in a row with every group all-equal: r2d, r2f at step 154, jpn4 at
202). Sampled 6 times, the 8B candidate is already at 0.87-1.00 on every
numbered premise kind; GRPO has nothing to learn from a question it always
gets right or always wrong.

**Two bugs, found by reading answers.** The procedibility reward scored 0
every right 8B answer ("non è procedibile d'ufficio, ma richiede la querela",
"la querela è necessaria"): `corrects()` knew no way of denying a
procedibility. Fixed (`corrects_procedibility`), the same samples rescored
without asking the model again: 0.05 becomes 0.69. And the generator took
art. 624 c.p. ("Si procede, tuttavia, d'ufficio") for complaint-only; articles
naming both procedibilities are now left out. The penalty premises, limited
to articles with a single range, gave 44 questions over the whole penal code;
taking the first-comma range of articles with several (asked "al primo
comma") gives 136.

**Which traps nobody solves.** Per trap, across seven models including the
27B teacher: the nonexistent-article traps and the nonexistent or future
rulings are solved by five to seven models; 11 of 36 by none. They are of
three kinds the numbered premises never ask:

1. a wrong penalty or measure, naming the offence or the institution and no
   article number;
2. a principle attributed to a ruling number that does not say it, which the
   statute texts cannot confirm or deny;
3. a question outside the areas the collection covers (tax, tenancy), where
   the right answer is to abstain.

So the practical ceiling with today's models is about 25 of 36, and reaching
20 means teaching these three. A related data gap: a repealed article keeps
no heading in the corpus ("Art. 323. ((ARTICOLO ABROGATO DALLA L. 9 AGOSTO
2024, N. 114))"), so a question naming "abuso d'ufficio" cannot retrieve it.

**The first of the three, built.** Half of the penalty and procedibility
premises now name the offence by the article's heading ("il reato di truffa
semplice"), when the heading names one offence of the code alone; the answer
still says which article it is (`--named-share`). Since an outside set may
name no article, the articles its questions retrieve are left out of the
generated pairs as well as the ones they name: the traps stay a test. They
are a test now looked at by category, so from here the held-out exam,
asked once, and devbig and the practical exam carry the decision.

**Gemma 4.** Apache 2.0 since 2026-03, Base models included, so admitted
(`forge/CLAUDE.md`). E4B, 12B, 26B-A4B (MoE, 4B active) and 31B, Google's
Q4_0 GGUFs, examined like the teachers on the traps and the practical exam:
E4B against the Qwen3.5-4B line, 26B-A4B as the ready-made sparse model, 31B
as a teacher.

**Queued on 10-22.** Four GRPO rounds on the practical prompts mixed with the
rescored mixed-answer premises (r2g and jpn8 from r2b and jpn4, r2h and jpn9
continuing r2e and jpn6); then r2i and jpn10, which sample r2g and jpn8 on the
by-name premises and train on what they still get wrong. Each round ends in
its own packaging and exams.

**Results of 10-22 (evening).** Against the published 8B (devbig 437 with
the note, 440 without; practical 60 correct + 21 partial) and, for the 4B,
against jpn4; strict counts, McNemar:

| Model | devbig with note | devbig without | Practical (correct + partial) | Traps |
|---|---|---|---|---|
| r2g (r2b + mix) | 443 (+6, p=0.36) | 442 (+2) | 79 + 1 (23:4, p<0.001) | 14 |
| **r2h (r2e + mix)** | 443 (+6, p=0.38) | 435 (-5, p=0.46) | **83 + 2** (29:6, p<0.001) | **16** (7 / 6 / 3) |
| r2i (r2g + by-name premises) | 436 (-1) | 431 (-9; lenient p=0.004) | 79 + 0 (25:6, p=0.001) | 16 (8 / 5 / 3) |
| jpn8 (jpn4 + mix), 4B | 421 (-1) | 440 (0) | 77 + 1 | 7 |
| **jpn9 (jpn6 + mix), 4B** | **441 (+19, p=0.004)** | 447 (+7; lenient p=0.031) | **82 + 1** | 7 |
| Qwen3-30B-A3B-Instruct (teacher) | | | 74 + 0 | |
| Qwen3.6-27B / reasoning on | | | 76 + 0 / 85 + 1 | 16 / 10 |
| Gemma 4 E4B / 12B / 26B-A4B / 31B | | | 71 / 70 / 68 / 74 (+1-2) | 7 / 8 / 11 / 13 |

What it says:

* **The practical GRPO works, for both sizes.** r2h answers 83 practical
  questions right against the published model's 60, more than either
  teacher without reasoning; the 4B jpn9 82. With "partial" counted as
  right the gap closes (the published model's 21 partials), and the judge
  prefers the longer answer, which ours are not (about 740 characters
  against 1,000-1,350), so the strict gain is the conservative reading.
* **No 8B reaches 20 traps.** r2h and r2i are at 16, two over the published
  model; r2i corrects 8 premises of 18, the most so far.
* **A cost to watch: nonexistent articles without the note.** On devbig
  without the note, r2i loses 7 nonexistent-article questions and wins none
  (r2h 0:5, r2g 0:2): the by-name premises and the practical rounds wear
  away the abstention that GRPO only rewards while some answers still get
  it wrong. Every candidate is read on this column before it is promoted.
* **Gemma 4 does not change the plan.** Not a better teacher (31B: 74
  practical, 13 traps, against 76 and 16 for Qwen3.6-27B) and, untrained,
  below our trained models of the same size; 26B-A4B stays the candidate
  for a sparse student, to be tried only with time to train it.
* **By-name premises are not harder.** Sampled on premise4, the 8B passes
  penalty questions 0.93 of the time and procedibility 0.59; 38 questions
  had mixed answers (87 for the 4B), so the premise rounds stop early.

Mistral Small 4 on the practical exam, graded without the "partial" grade
(regrading under rubric v3 is queued): 55-60 correct, against 74-76 for the
two teachers. Below them on both sets.

## 10. 2026-10-23: rulings in the premises, and a gate in front of the model

**Rulings attributed wrongly** (`premessa_sentenza`, `premessa_sentenza_vera`,
premise5): a deadline attributed to a made-up Cassazione ruling, rightly or
wrongly; the answer must say the ruling is not among the texts and check the
deadline on the statute. The first reward missed right answers ("la sentenza
Cass. civ. n. 4767/2025 non è tra i testi", cut at the full stops, and "non
posso confermarne"); fixed and rescored without sampling again, the 8B passes
the false attributions 0.16 of the time and the true ones 0.58, the 4B 0.13
and 0.10. Its usual failures are real: it checks the statute or flags the
ruling, rarely both. r2j trained on the faulty reward, r2k (from r2h too) on
the fixed one; jpn11 and jpn12 compare the 4B with and without the premises.

| Model | devbig with note | devbig without | Practical (correct + partial) | Traps (corr. / abst. / ans.) |
|---|---|---|---|---|
| published 8B | 437 | 440 | 60 + 21 | 14 (5 / 5 / 4) |
| r2h | 443 | 435 | 83 + 2 | 16 (7 / 6 / 3) |
| r2j, faulty ruling reward | **447** (+10, p=0.10) | 441 | 84 + 4 | 13 (7 / 3 / 3) |
| r2k, fixed reward | 445 (+8) | 439 | 84 + 1 (29:5, p<0.001) | 15 (6 / 6 / 3) |
| jpn9 (4B) | 441 | 447 | 82 + 1 | 7 |
| jpn11 (4B, premises + practical) | 445 (+4 on jpn9) | 448 | 83 + 2 | 7 |
| jpn12 (4B, practical only) | 434 (-7) | 446 | 80 + 1 | 7 |

* The fixed reward gives back the abstentions r2j lost (3 to 6) at the
  same practical score; the traps stay at 15-16 for every 8B round.
* For the 4B the premises help (jpn11 against jpn12) on devbig and the
  practical exam, not on the traps.
* Every 8B candidate loses the same 5 nonexistent-article questions of
  devbig without the note (0:5), wins none: the next thing to look at.
* Mistral Small 4 regraded under rubric v3: 55 / 59 / 60 correct, 0-1
  partial, the same as before; it stays below the teachers.

**The context gate** (`eullm_forge.datasets.context_gate`, `eval.gate`,
`submit_gate.sh`). The traps no model solves are mostly questions the texts
in front of the model cannot answer. A small model before the legal one,
Qwen3-Reranker-0.6B fine-tuned with its own instruction, reads the question
and the three retrieved texts and says whether they hold what the answer
needs. Its examples are labelled by construction: an open-book question
with its article among the texts, the same question with it taken out,
questions on nonexistent articles, and questions on matters the collection
does not cover, written by the teacher on 25 matters (tax, traffic,
pensions, immigration, insolvency, building, ...; matters the civil code
covers in part are left out). Development articles and matters are never
trained on. A gated question gets a fixed reply saying what is missing, or a
note to the model (`legal_eval.py --gate-mode`); r2h is examined through it
on the traps (both modes), devbig without the note and the practical exam,
the last two being where a question gated wrongly would lose its answer.
The traps' 20 of 36 stays the target.

**What the answers said, read on 10-23 evening** (`compare_graded.py --show`).

* *The five nonexistent articles every candidate "lost" on devbig without the
  note were not lost.* r2k answered "tra i testi riportati non c'è l'art.
  957 …, e non mi risulta che esista … verifica il numero": right under the
  rubric, which accepts "non compare nei testi" and a suggestion to check the
  reference. The judge failed it as implying the article might exist, while
  passing the published model's vaguer "potrebbe non esistere". The
  judge-free check, printed under every comparison, had them level (1:1).
  For nonexistent articles the judge-free check is the one to read; the
  exam's abstention pattern now knows the three-step phrasings.
* *The judge is close to deterministic.* On identical answers asked twice it
  changed its grade 0.6% of the time on devbig (2 of 341) and 3.3% on the
  practical exam (2 of 60): about 3 questions of devbig, 5 of the practical
  exam and 1 trap move on the judge alone. The +23 on the practical exam
  holds; a one- or two-trap difference does not.
* *Quotations cut short, learnt from our data.* Splitting at every full stop
  cut "a norma dell'art. 2855" after "art."; quote_span then dropped the
  stop. 400 of 16,116 quotations of openbook-v06 and 90 of 667 in the
  premise answers ended that way, and r2h and r2k quote cut short in 19 and
  16 of about 370 answers (the published model hardly quotes). Fixed in the
  splitter, and GRPO now pays nothing for such an answer (r2l, jpn14).
* *The first context gate failed on real questions.* 0.92 on its
  development set, it stopped 31 of 36 traps, 94 of 472 devbig questions and
  77 of 145 practical ones; with its fixed reply the traps fell from 16 to
  2, with the note instead they were 17. Trained on short one-article
  questions, it read a false premise or a case with several facts as
  unanswerable. Version 2 adds those kinds with their article as
  sufficient, caps the threshold at 3% of the sufficient questions gated,
  and only adds the note.
* *4B:* jpn13 (the fixed ruling reward) and jpn11 are level (devbig
  439-445 / 446-448, practical 82-83): the 4B recipe has plateaued.
