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

### Requests and suggestions

- Numbers against larger models (27B class, larger Mistral models), as
  competitors on the same exams (2026-10-18).
- The cost of running the model in an office or firm (2026-10-18).
- Case law in training: in criminal law the statute alone says little, what
  matters is how the Cassazione reads it (private contact, 2026-10-18).

