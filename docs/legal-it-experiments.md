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
is most likely the September line's (section 1): it wrote the
`instruct-pairs` files stage 3 used on the distilled student (commit
7040d91), and the published models' SFT is the 18,972 open-book pairs
alone. To be confirmed from the SFT job's log, which names the pair files
it loaded.

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
