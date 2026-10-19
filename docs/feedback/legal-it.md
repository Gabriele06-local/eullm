# legal-it — feedback register

What people said about the legal-it models in public posts and comments, and
on which site (main domain only). No links, no thread or community names, no
usernames.

Each legal question the model gets wrong is recorded with the question, what
the model answered, and what the right answer was, so it can feed the next
training round.

## legal-it-8b — launch post (reddit.com)

### Comments

<!-- One entry per comment: date, topic, what was said, our reply (summary). -->

**2026-10-18 — scale and "sovereignty"** (reddit.com, one thread)
- Said: was a supercomputer needed for an 8B? With that compute, aim higher,
  or there will never be a "sovereign LLM". A second commenter defended the
  choice (the infrastructure staying in Europe is part of the message); a
  third asked what sovereignty means for a model trained on public data.
- Reply: the aim is not a general sovereign LLM (Mistral builds those) but
  small domain models that run on modest hardware; Leonardo built the model,
  it is not where it runs; next step a larger Mistral teacher distilled into
  small students. On sovereignty: what matters is where the model runs, on
  the lawyer's own machine, with open weights under Apache 2.0.

**2026-10-18 — training data: statutes or also judgments?** (reddit.com)
- Said: is the training data only the law, or also court decisions?
- Reply: statutes only (Normattiva), no court decisions and no personal data;
  Italy is a civil law system; case law is not covered and was not evaluated.

**2026-10-18 — "I could fine-tune this on a single 5090"** (reddit.com)
- Said: an 8B can be fine-tuned on one consumer GPU, so the cluster is a
  "weird flex".
- Reply: the recipe explained (open-book SFT with LoRA, GRPO with
  programmatic rewards, merge, Q4_K_M) with the score of each step (80.9 →
  84.6 → 89.0%); not tried on a 32 GB card, so no claim that it fits; the
  cluster also generated the data and graded the exams, and the next step
  (distillation from a larger teacher) needs teacher and student in memory.

**2026-10-18 — cost and amount of data** (reddit.com)
- Said: how much did it cost, how much data?
- Reply: 18,972 SFT examples from public legislation, then 250 GRPO steps;
  asked whether they mean the cost of training (a EuroHPC research
  allocation, no cloud bill, no GPU-hour figure published) or of running it.

**2026-10-18 — comparison with larger models** (reddit.com)
- Said: how does it compare out of the box with larger models (a 27B Qwen,
  larger Mistral models)?
- Reply: Ministral-3-14B with the same recipe scored the same as the 8B on
  the 472-question development exam (436 vs 437); the 8B ties or beats
  Qwen3.5-9B; numbers for 27B-class models not published yet.

**2026-10-18 — private contact** (reddit.com, private message)
- A team building a RAG platform for criminal law asked for a call and
  offered to test the model with their own set of trap questions and to have
  their lawyers review the answers.

### Model errors

<!--
Template:

#### <short title>
- **Where:** reddit.com, <date>
- **Setting:** with retrieved texts in the prompt / without (closed book) / unknown
- **Question:** ...
- **Model answer:** ...
- **Correct answer:** ... (source: article / law)
- **Error type:** invented article / accepted false premise / wrong term / wrong article cited / other
-->

#### Notice period of a short-term lease, with half the texts left out
- **Where:** reddit.com, 2026-10-19
- **Setting:** with texts in the prompt, but by the user's account half of
  the relevant ones were left out
- **Question:** rental contract questions; one on the notice period of a
  short-term lease (exact wording not given)
- **Model answer:** cited the right Civil Code articles, gave a wrong notice
  period (figure not given)
- **Correct answer:** not given; details asked. Residential leases are
  largely governed by l. 431/1998 and l. 392/1978, which are not in the
  collection the model was trained with.
- **Error type:** wrong term; no abstention on insufficient texts

The entries below come from one external test (reddit.com, private
contact, 2026-10-18): a team building a criminal-law RAG platform ran
`legal-it-8b-Q4_K_M.gguf` in two settings and sent a written report. In the
"card format" setting the prompt followed the model card (one user turn, the
first 3 texts of their own search over Cassazione rulings, maxims and
statutes, cut to 3,000 characters, the missing-article note, temperature 0,
400 tokens): 11 questions, graded by a model of another family and then
read by them. In the "pipeline" setting the model wrote answers inside their
production flow: 50,000-80,000 tokens of sources, strict citation rules,
automatic verification and correction rounds (5 questions). Most questions
are about case law, which the model was never trained on.

#### False premise accepted with the right text in the prompt
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format; art. 624-bis c.p. was the first of the 3 texts
- **Question:** after the Cartabia reform, theft in a dwelling (art. 624-bis
  c.p.) is prosecuted on complaint: within what time must the complaint be
  filed?
- **Model answer:** under art. 85 d.lgs. 150/2022 the victim may file the
  complaint by 30 March 2023, three months from the reform's entry into force.
- **Correct answer:** art. 624-bis stays prosecuted ex officio; the complaint
  introduced by the reform concerns simple theft (art. 624). Same answer in
  both runs.
- **Error type:** accepted false premise

#### Wrong reasoning on a correct figure (plea bargain)
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format
- **Question:** since fraud is punished with two to eight years, is a plea
  bargain at one year and four months possible?
- **Model answer:** the penalty for simple fraud is six months to three
  years, so one year and four months is within the maximum, as long as no
  aggravating circumstances raise the penalty. In another run: possible only
  with the aggravating circumstance of art. 61 n. 5.
- **Correct answer:** the penalty is right (art. 640 c.p.), the test is not:
  under art. 444 c.p.p. what counts is that the agreed penalty, after the
  circumstances and the reduction of up to a third, does not exceed five
  years; aggravating circumstances do not exclude it. The false premise is
  not named as such.
- **Error type:** other (wrong legal test); false premise not flagged

#### Neighbouring offences confused (pepper spray)
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format
- **Question:** can carrying a pepper spray meant for attack be punished
  under art. 4 l. 110/1975, and where is the line with art. 699 c.p.?
- **Model answer:** the spray is a weapon only if the circumstances show an
  unlawful purpose, and then the offence is art. 699 c.p.
- **Correct answer:** per the expected maxim (Rv. 284404-01) the unlawful
  purpose falls under art. 4, para. 2, l. 110/1975; art. 699 c.p. requires
  the offensiveness features of the 2011 ministerial decree. The model moves
  the requirements between the two rules (the other way round in another run)
  and does not explain the line, which was the question.
- **Error type:** other (offences confused)

#### Provision from another code
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format
- **Question:** is the rejection of the civil party's application abnormal?
- **Model answer:** right principle, concluding "non sussiste l'abnormità
  prevista dall'art. 360, comma 1, n. 4, c.p.p.".
- **Correct answer:** art. 360, para. 1, n. 4 is in the code of civil
  procedure and has nothing to do with abnormality in criminal proceedings.
- **Error type:** invented article (wrong code)

#### Answers from memory, and the case instead of the rule
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format
- **Question:** particular tenuousness of the offence in a continuing offence.
- **Model answer:** a list of evaluation criteria not found in the texts
  given; with another passage in the prompt (one excluding tenuousness on its
  facts) it answered the opposite, excluding it for a continuing offence.
- **Correct answer:** not stated in the report beyond the above; the model
  follows the facts of the passage rather than the general rule.
- **Error type:** other (content not from the texts; case taken as rule)

#### No abstention when the texts are not enough
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format; the expected decision was not among the 3 texts
- **Question:** conditional abbreviated trial, judgment by a judge other than
  the one who heard the witness.
- **Model answer:** built on a ruling about something else, bringing in trial
  rules (the witness list of art. 468 c.p.p.) that do not apply to the
  abbreviated procedure, with contradictory reasoning.
- **Correct answer:** say the texts do not answer the question.
- **Error type:** other (no abstention on insufficient texts)

#### Two opposite statements in one answer (Sezioni Unite)
- **Where:** reddit.com (private contact), 2026-10-18
- **Setting:** card format, with the text of SU n. 13783/2025 in the prompt
- **Question:** attributes to SU n. 18474/2025 confiscation by equivalent of
  the whole profit, jointly.
- **Model answer:** first gives as the SU principle the majority view the
  ruling describes (the whole to each participant), then concludes that the
  SU limit it to what each obtained.
- **Correct answer:** the SU limit it to what each obtained, and n.
  18474/2025 is about something else, which the answer does not say.
- **Error type:** accepted false premise (wrong ruling number), contradiction

#### Patterns reported over the whole test
- Wrong source for the right principle (pipeline): a ruling that decided
  the opposite; the passage with only the operative part instead of the
  reasoning; a neighbouring passage of the same ruling on another point.
  Maxims are never cited.
- Indicia of the concrete case written as legal requirements (pipeline,
  murder in a robbery).
- Multi-part questions: the part that needs reasoning is dropped (answers
  of 71-310 tokens).
- Citations: none of the 11 answers uses the [1], [2], [3] numbering; in the
  pipeline 4 first drafts out of 5 cite nothing, and quoted words are
  paraphrased.
- Does not correct itself when the drafts come back with precise remarks:
  repeats the same citation; after the rounds 2 answers in 5 end empty and 2
  reduced to one sentence.
- Fines in lire: Normattiva keeps the original amounts in lire in several
  criminal provisions (art. 640 c.p., art. 4 l. 110/1975) and the model
  gives them as the current penalty.
- Pipeline result: 1 answer in 5 passed their verification (their
  production model: 5 in 5), although the first draft held the right
  principle in 5 cases out of 5.

#### What worked, from the same report
- The missing-article note: on art. 416-quater c.p. it said the article is
  not in the texts and may not exist — the only one of 11 answers that also
  passed their source verification.
- Corrected numeric premises when the right figure was in the text: the
  penalty for fraud, and 15 days (not 30) to appeal a judgment with
  reasons given at once.
- Finds the right source among similar passages, also in prompts of
  50,000-80,000 tokens, and restates its principle faithfully.
- Clean legal Italian, no English words.
- Q4_K_M ran on a 16 GB T4 with 86,000 tokens of context (11.3 GB used).

#### Technical point to check
- With llama.cpp and `--jinja`, a request without a system message gets the
  GGUF chat template's default Ministral system prompt ("You are
  Ministral-3-8B-Instruct-2512… Le Chat…", about 525 tokens). The model card
  says to send no system prompt, so every user following it receives that
  one. To check whether training rendered the same default.

### Requests and suggestions

- Numbers against larger models (27B class, larger Mistral models), as
  competitors on the same exams (2026-10-18).
- The cost of running the model in an office or firm (2026-10-18).
- Case law in training: in criminal law the statute alone says little, what
  matters is how the Cassazione reads it (private contact, 2026-10-18).

