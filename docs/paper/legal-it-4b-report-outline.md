# Verticalizing an LLM on a fixed EuroHPC budget — report outline

> Status: outline · Started 2026-09-08 · Target: Zenodo (DOI → ORCID),
> plus a shorter engineering post on the ZeRO-3/MoE result.

This is the skeleton of the write-up for the `eullm/legal-it-4b` run, and —
more urgently — the list of measurements that **only exist while the run is
happening**. Throughput, per-phase node-hours, loss curves and memory peaks
cannot be reconstructed afterwards from a finished checkpoint. Capture them
as they appear; the prose can wait.

## What the contribution is, and what it is not

"We fine-tuned a model on Italian legal text" is not a contribution; there
are many such papers. What is under-documented, and what this report is
actually about, is the part usually left out:

1. **The budget determines the architecture.** A 1,250 node-hour allocation
   with a 24 h walltime cap and no network on compute nodes is why the
   teacher is a MoE rather than the dense model the plan started with. Papers
   report the configuration they ended with; the reasoning that forced it is
   the reusable part.
2. **A MoE teacher under ZeRO-3 has a memory profile nobody writes down.**
   Reproducible failure, exact numbers, verified fix (§4).
3. **GDPR-aware corpus construction from a public case-law archive**, stated
   honestly as pseudonymisation rather than anonymisation, including a
   redaction bug that reported success while leaking (§3).
4. **The deliverable is not gated on the run finishing.** A distillation
   run writes a shippable model every `save_steps`, not one at the end: each
   checkpoint is a complete adapter that merges, converts and quantizes like
   the final output, and is only less trained. Treating the model as the
   *terminus* of a four-week chain rather than as its *continuous output* is
   what makes a fixed-term allocation frightening — and it is a habit, not a
   constraint. Packaging on a cadence moves end-to-end validation from week
   four to day three, lets quality be measured while there is still budget to
   act on the answer, and means an expired allocation leaves a model behind
   rather than a directory of optimizer state.
5. **The failures, with numbers.** Model ids that did not exist, a launcher
   pointing at a stale checkpoint path, a pre-flight that verified the wrong
   tokenizer, an OOM at the first forward. Each cost measurable time, and
   each is the kind of thing the next team hits.

## Structure

1. **Introduction** — sovereign / EU-compliant LLMs, why verticalization
   rather than a general model, why Italian case law as the first domain.
2. **The corpus** — italgiure SentenzeWeb, size, split, chunking. The
   pseudonymisation pipeline: what is redacted, what is deliberately kept
   (public officials, R.G. numbers, company names) and the legal reasoning
   for the distinction. Why the output remains personal data under GDPR
   Art. 4(5).
3. **A redaction bug that reported success** — the `\b`-anchored codice
   fiscale pattern, the four OCR shapes it silently missed, 59 codes reaching
   the training text after a "clean" run over 5.7 M redactions, and the
   gate that now blocks a launch on a dirty corpus. Generalisable lesson: a
   clean report from a redaction tool is evidence about its patterns, not
   about the corpus.
4. **Teacher and student selection under a compute ceiling** — the model
   survey (what Qwen actually publishes as Base, why Qwen3.5/3.8, Mistral
   and Gemma were excluded), and the MoE decision: 3.3 B active of 30.5 B
   total on a forward-only path.
5. **The ZeRO-3 memory result** — the core engineering finding. ZeRO-3 keeps
   parameters partitioned at rest but materialises a whole layer on every
   GPU; a Qwen3-30B-A3B layer is ~625 M parameters against ~25 M for a dense
   layer of the same hidden size, so dense-tuned limits fill 63.4 GiB of an
   A100 64 GB twenty seconds into the first forward. With the derivation and
   the corrected configuration.
6. **Operating inside a batch allocation** — 24 h walltime chaining,
   resume-from-checkpoint, no network on compute nodes (prefetch discipline),
   and observability: why the job now reports on itself, and what the four
   heartbeat numbers distinguish.
7. **Results** — the table below.
8. **Reproducibility** — configs, scripts and commit hashes; what a reader
   needs to repeat this on their own allocation.
9. **Limitations** — single domain, single language pair, one student size,
   no human evaluation of legal correctness, corpus not publishable.

## Measurements to capture DURING the run

Fill these in as they occur. An empty cell after the run is a number lost.

### Phase 0 — corpus

| Quantity | Value |
|---|---|
| Source slices (years, sections) | Cassazione snciv + snpen 2021-2026 + codici + Costituzione |
| Chunks, train / val | 1,127,316 / 11,387 (99/1, seed 42) |
| Tokens (approx) | ~700 M |
| On-disk size, train / val | 2900 MiB / 29 MiB |
| Pseudonymisation counts by category | (from `metadata.anonymization`, aggregated) |
| Codici fiscali found by the round-6 sweep | 59 (57 train / 2 val) |

### Phase 1 — continued pre-training

| Quantity | Value |
|---|---|
| Trainable params / total / percent | 106,954,752 / 30,639,077,376 / 0.3491 % |
| Time to first training step (cold cache) | |
| Time to first training step (warm cache) | |
| Seconds per optimizer step | |
| Peak VRAM per GPU (steady state) | |
| Peak host RSS per rank | |
| Steps for one epoch | |
| Wall-clock and node-hours for one epoch | |
| Loss at step 0 / 1k / 10k / final | |
| Val perplexity at each eval | |
| Number of chained jobs actually needed | |

### Phase 2 — distillation

*In progress. Figures below are at step 18,360 of 70,457 (26.1 %), after
three 24 h links — jobs 57353618-20, 2026-09-13 to 2026-09-15.*

| Quantity | Value |
|---|---|
| Total optimizer steps for one epoch | 70,457 (1,127,316 chunks ÷ effective batch 16) |
| KL term at step 20 / 820 / 1,620 / 18,260 | 2.6989 / 1.0209 / 0.7881 / 0.5105 |
| Effective alpha schedule realised | none — constant 0.700, recovered from the logged components and matching `kl_alpha: 0.7` |
| Seconds per step | 10.8 s per optimizer step (1.48 micro-batches/s at accumulation 16) |
| Peak VRAM (teacher + student) | 36,452 MiB on cuda:0 (student + teacher shard), 24,512 / 16,274 / 16,274 MiB on the rest |
| Node-hours | 61 for 26.1 % of one epoch; ~240 projected for the full epoch |

The throughput figure has two independent sources that agree: the in-run
counter (1.48 micro-batches/s → 333 optimizer steps/hour) and the checkpoint
directory timestamps (1,000 steps every 2 h 59 m, four consecutive intervals
within three minutes of each other).

**The training loss flattened at around step 14,000 and has not moved since.**

| step | loss | KL | CE | lr |
|---:|---:|---:|---:|---:|
| 20 | 2.4462 | 2.6989 | 1.8562 | 1.00e-06 |
| 1,620 | 0.9549 | 0.7881 | 1.3439 | 5.00e-05 |
| 10,360 | 0.7321 | 0.5467 | 1.1643 | 4.78e-05 |
| 14,260 | 0.7008 | 0.5170 | 1.1297 | 4.56e-05 |
| 18,260 | **0.7028** | 0.5105 | 1.1515 | 4.28e-05 |

Steps 2,420 → 10,360 bought 0.166 of loss; the next 8,000 bought 0.029; the
last 4,000 — twelve hours of a Booster node — bought nothing, oscillating
between 0.696 and 0.708 with no trend, and CE rose slightly. This is worth
more than the usual plateau observation because `num_train_epochs: 1` with no
repetition means **every batch is unseen data**: the training loss here is a
running measurement on held-out text, not on memorised text.

It is not, however, evidence that the run is finished, and the distinction
matters for anyone sizing a chain from this report. The schedule is
`get_cosine_schedule_with_warmup` over all 70,457 steps, and at 4.28e-05
against a 5.00e-05 peak the learning rate is still at 86 % — the anneal has
barely started, and plateaus that break during the anneal are the normal case.

What follows from that is a scheduling result rather than a training one:
**the worst place to stop a cosine run is partway down it.** The seven links
booked for this phase reach roughly step 47,700, which is 68 % of the
schedule and a learning rate still a quarter of peak. That buys a model that
was never consolidated, for the same node-hours. Either the chain runs to the
end of the schedule or the schedule is recomputed for the chain; stopping
where the booking happens to end is the one option that wastes the compute it
spends.

### Phase 3 — quantization and export

Measured 20 September on `legal-it-4b-step8400` and `legal-it-4b-step28000`.

| Quantity | Value |
|---|---|
| F16 GGUF size | 7,672.62 MiB (16.00 BPW) |
| Q4_K_M GGUF size | **2,375.91 MiB (4.95 BPW)**, 2.4 GB on disk |
| Compression, F16 → Q4_K_M | 3.23× |
| Quantization wall-clock | 154–168 s, 4 CPU cores, no GPU |
| Generation, CPU only (4 cores, serial node) | 9.3–10.5 tok/s |
| Prompt processing, same | 28–48 tok/s |
| Perplexity before / after quantization | **not measured** — see below |
| Tokens/s on the target consumer GPU | **not measured** — no such hardware in this allocation |

Two gaps stated rather than left blank. The F16-vs-Q4 comparison needs a
perplexity run on the 7.7 GB intermediate, which is cheap but was not done
before the F16 files were deleted as build artefacts; it costs one re-export
to recover. And nothing in this allocation is a consumer GPU, so the headline
claim — *runs on a laptop* — is supported here only by the file size and by
CPU-only throughput on a serial node. It needs one measurement on real target
hardware before it appears in a report as a number.

### The first quality result

Perplexity of each model on Italian legal text from `val.jsonl`, same corpus,
same chunk count, same context, same quantization on every side.

**Read the numbers with the caveat below, which is not small.** `val.jsonl` is
a 1 % split at seed 42 and no record in it enters a gradient — but
`format_pretraining.py` shuffles and splits **chunks, not documents**. Each
record is a ~2,048-token fragment of a ruling, so the chunks of one ruling are
scattered across train and val: chunk 3 trained on, chunk 4 held out. The
student was therefore scored on passages whose immediate neighbours it had
read — same parties, same cited articles, same recurring formulas — while the
base had seen none of it.

That inflates the gap, by an amount this measurement cannot bound. What it
does not plausibly do is reverse the ordering: the base is worse than both
students on both samples, by a margin far outside the confidence intervals.
Treat the direction as established and the **magnitude as an upper bound**.

It has since been remeasured against text with no overlap at all — see
**The clean number**, below, which is the one to quote.

Two independent samples of the split, ~73 kB each, because a single 28-document
slice cannot tell a result from its sample:

| | corpus A (file order) | corpus B (seed 7) |
|---|---|---|
| Qwen3-4B-Base, untouched | 6.6500 ± 0.157 | 6.9796 ± 0.174 |
| legal-it-4b **step-8400** (split arm) | 3.2787 (**−50.7 %**) | 3.6434 (**−47.8 %**) |
| legal-it-4b **step-28000** (control arm) | 3.0320 ± 0.066 (**−54.4 %**) | 3.3085 (**−52.6 %**) |

Both orderings survive the change of sample: every distilled checkpoint
roughly halves the base model's perplexity, and step-28000 beats step-8400 by
7.5 % on A and 9.2 % on B. The defensible statement is **−48 to −54 % across
two chunk-level samples, as an upper bound** — not a single figure, and not a
document-level held-out result.

### The clean number

Consiglio di Stato, 2025. A court the training never touched, in a different
jurisdiction, from an index the project had not used before — clean by
construction rather than by an argument about splits. 300 judgments fetched
from the institutional portal against the CC BY 4.0 OpenGA metadata; the
evaluation corpus is 11 of them, 230,134 bytes, 120 chunks at ctx 512.

| | PPL | vs base |
|---|---|---|
| Qwen3-4B-Base, untouched | 6.4607 ± 0.089 | — |
| legal-it-4b **step-28000** | **4.9589 ± 0.064** | **−23.25 %** |

**This is the figure to quote**, and the one the abstract should carry. A 4 B
student, quantized to Q4_K_M at 4.95 bits per weight and 2.4 GB on disk,
reads administrative-law judgments 23 % less perplexedly than the model it
was distilled from — on a corpus neither of them had seen. The confidence
intervals are ±0.09 and ±0.06 against a gap of 1.5, so the effect is not in
question.

**Half of the in-domain gap does not survive the move**, and saying so is the
point of having both numbers:

| | Cassazione (chunk-level split) | Consiglio di Stato (clean) |
|---|---|---|
| step-28000 vs base | −54.4 % / −52.6 % | **−23.25 %** |

Two causes act at once and **this measurement does not separate them**: the
split contamination described above, and a genuine domain shift from ordinary
to administrative jurisdiction, where the institutes, the procedural frame and
the recurring formulas all differ. Attributing the drop to either alone would
be choosing the more convenient story. What the pair does establish is a
floor: whatever part of the in-domain figure was contamination, **at least
23 % is transfer to legal Italian the model had never read**.

Absolute perplexities are not comparable across the two corpora and should
not be tabulated as if they were. Only the ratios within a corpus mean
anything.

**The transfer curve**, measured on 2026-09-24 on the same corpus, the same
120 chunks and the same base. The GGUFs had been packaged earlier; by the
time they were scored their checkpoints were gone, because the trainer keeps
only the last three — the reason a checkpoint that is not packaged on a
cadence is a model that never existed.

| step | PPL | vs base | share of epoch | gain over the previous 7,000 steps |
|---|---|---|---|---|
| 7,000 | 5.2834 | −18.22 % | 9.9 % | 18.22 points |
| 14,000 | 5.0875 | −21.25 % | 19.9 % | 3.03 |
| 21,000 | 4.9951 | −22.68 % | 29.8 % | 1.43 |
| 28,000 | 4.9589 | −23.25 % | 39.7 % | 0.57 |

**Each block of 7,000 steps returns less than half the one before it**:
3.03, then 1.43, then 0.57 points. That is flatter than logarithmic — a
log-linear curve would return the same gain per doubling, and the second
doubling here (14,000 → 28,000, +2.00) already returned a third less than
the first (7,000 → 14,000, +3.03). Four-fifths of the gain measured at 28,000
was present at 7,000, a tenth of the way through the epoch.

Extrapolated, the remaining 60 % of the epoch is worth between 0.4 and 2.6
points depending on the model fitted, and the data sit with the lower end.

**What this does not settle** is the schedule. The learning rate follows a
cosine over the full epoch and is still at two thirds of its peak at 28,000;
the anneal in the last part of a cosine schedule is characteristically where
a flattened curve bends down again, and part of what reads here as a plateau
may be the high learning rate rather than the model. One point settles it:
the checkpoint at ~37,000, which exists and is next to be packaged. If
28,000 → 37,000 adds under half a point, the flattening is real.

**Design B against the control arm, per step**, on the same corpus and base:

| step | split arm (BF16 teacher) | control arm (8-bit teacher) | difference |
|---|---|---|---|
| 8,400 | −19.25 % | ≈ −19.0 % | 0.2 points |
| 12,600 | −20.97 % | ≈ −20.8 % | 0.2 points |

The control values at those steps are log-interpolated between its measured
points, not measured. The difference — 0.2 points, about 0.013 in
perplexity — is a fifth of the measurement's own confidence interval of
±0.06, so **the BF16 teacher does not teach measurably better per step**.
That is what P6 predicted: the 8-bit teacher matched the BF16 one at a KL of
0.00616 nats with 99.955 % top-5 agreement, and two teachers whose
distributions are that close produce students that learn alike. Design B's
case rests on throughput and memory headroom — the room it leaves for a
larger student — not on the quality of what it teaches.

**Four things this does not say**, each of which a reader will ask:

* **The split is chunk-level, not document-level.** Stated above and repeated
  here because it is the one a reviewer finds by opening
  `format_pretraining.py`. `split_indices` shuffles the record list and takes
  1 %; the records are chunks. Grouping by `sentence_id` before splitting is a
  four-line change, shipped, and is the fix for `medical-de` and `finance-fr`;
  it cannot retroactively clean this run's split, because with chunks assigned
  independently the chance that any multi-chunk ruling landed wholly in val is
  effectively nil. That is why the clean number above comes from a different
  court rather than from a better slice of this corpus.

  The same leak had a second mouth: legislation chunks carried `source_id`
  but no `sentence_id`, so every one of them became its own group and the
  codes' articles — whose text repeats verbatim across chunks — scattered
  across both sides too. Found and fixed separately.

* **It is not legal knowledge.** Asked for the content of art. 2086 c.c., both
  checkpoints answered fluently with the content of *other* articles — one
  described the restitution of goods, the other the `procura al difensore` of
  art. 83 c.p.c. They have learned the register, not the map from article
  number to text. Measuring that needs a verifiable-answer benchmark, which
  does not yet exist here.
* **Part of the gain is surface form.** The student has also learned how these
  documents are laid out, cited and punctuated. That is verticalization, but
  it is not knowledge, and the two get conflated exactly when the number
  flatters.
* **In-domain, not general.** Same courts, overlapping years as the training
  corpus. It answers "did this help on this domain", and nothing beyond.

One qualitative observation, reported as an observation because n=1 at
temperature 0.7: on the same prompt the untouched base **looped**, repeating
its own sentence two and a half times and fragmenting a token (`2 086`), while
both distilled checkpoints produced one complete period and stopped. Turning
that into a result needs fixed seeds over a handful of prompts, counting how
often each model enters repetition.

**The comparison between the two checkpoints isolates nothing.** step-28000 is
from the control arm (8-bit teacher, shared GPU) and step-8400 from the split
arm (BF16 teacher, separate GPUs): steps, teacher precision and node layout all
differ at once. The clean A/B exists only when the split arm reaches 28,000.

### Budget

Measured 2026-09-22, day 20 of 61. Snapshots in
[`measurements/`](measurements/), frozen daily because `sacct` forgets and
`sprio` never remembers.

| Quantity | Value |
|---|---|
| Node-hours allocated | 1,250 |
| Node-hours consumed | 256.0 (**20.0 %**) |
| Calendar elapsed | 495.7 h of 1,464 (**33.9 %**) |
| Node-hours lost to failed runs | 0.67 (job 56760964, ZeRO-3 OOM) |

**Consumption is running at 61 % of the rate the calendar demands.** On this
trajectory the allocation closes around 60 % used — roughly 490 node-hours
unspent. Under-use is a reportable outcome and counts against the next
request, so the reasons matter more than the number.

| Where the calendar went | hours | share |
|---|---|---|
| At least one job running | 236.0 | 47.6 % |
| Idle, **cluster full** | 98.0 | 19.8 % |
| Idle, **queue empty** | 161.7 | 32.6 % |
| Mean nodes while busy | **1.08** | |

**The empty queue is ours and nearly all of it is the first week**: 65h50m
from 02/09 to 04/09, 37h46m to 06/09, 42h35m to 08/09 — the days of the
non-existent model ids, the ZeRO-3 OOM and the pre-flight that checked the
wrong tokenizer. Those are in the incidents log with their costs. After
08/09 the empty-queue intervals are minutes, not days.

**The full cluster is not.** On 2026-09-22 the account's fairshare was
**0.751**, with effective usage 0.000086 against 0.000208 of shares — using a
third of what it was entitled to. Its jobs' priority of 141,963 decomposed as
QOS 120,000 + fairshare 18,783 + age 3,175, against a partition whose pending
priorities ran to **60,259,060**, with competitors at 212,303–275,734 and
**exactly one idle node** in `boost_usr_prod`, itself unresponsive. An account
under-using its share cannot close a 70,000-point gap by waiting: age
contributes three thousand.

**The 1.08 is the part we can still act on.** The allocation plan of
2026-09-11 said that averaging one busy node requires stretches at two or
three, because gaps are certain. That never happened: even while running, we
ran one node. The same 236 busy hours at two nodes would have delivered 470
node-hours rather than 256.

Two responses are in flight, and the report should say which worked: every
queued link cut from 24 h to 4 h, on the reasoning that a job no scheduler
can fit into a backfill window never runs on a saturated cluster; and an 8 B
student prepared as a second independent chain, which raises concurrency and
covers the other chain's queue gaps.

## Predictions, scored

The pilot's predictions were written down before it ran, in
[`v11-pilot-preregistration.md`](v11-pilot-preregistration.md). The results
section scores each one held / falsified / untested with the measured value
beside its threshold.

This is not decoration. Three predictions were already wrong in the first two
days, and each registered as a lesson only because a number had been stated
beforehand — the 60-hour Phase 1 estimate against 81 measured, the eval blamed
for a throughput loss it was not causing, and two ZeRO-3 levers that raised
the memory peak they were meant to lower. "We expected X and measured Y" is a
result; "we measured Y" is a data point.

### Scored so far

| | predicted | measured | |
|---|---|---|---|
| **P1** K per position | ≥ 64 | **64-65** | held |
| **P1** scoring vs generation | outside ±20 % | **2.73× faster** | held, for the wrong reason |
| **P3** mass retained at K=32, T=1 | ≥ 99 % | **93.66 %** | **falsified** |
| **P3** truncation KL at K=32, T=1 | < 0.01 nats | **0.0655** | **falsified** |
| **P6** KL | < 0.01 nats | **0.00616** | held |
| **P6** top-1 agreement | > 99 % | **95.43 %** | **falsified** |

**P3 is falsified, and it is the first result that changes a decision.** The
prediction was that formulaic legal Italian would be peaked enough for K=32 to
carry ≥ 99 % of the probability mass. Measured:

| K | T=1 | T=2 | T=4 |
|---:|---:|---:|---:|
| 16 | 90.84 % | 35.95 % | 0.97 % |
| 32 | **93.66 %** | 39.25 % | 1.36 % |
| 64 | 95.74 % | 42.95 % | 1.93 % |
| 128 | 97.27 % | 47.18 % | 2.80 % |

K=32 misses by more than five points, and **no K on the table reaches 99 %**,
including the K=128 the prediction dismissed as waste. Quadrupling K from 32 to
128 buys 3.6 points of mass for 4× the storage — the tail is long, not absent,
and that is the opposite shape from the one the reasoning assumed.

Truncating to the top-K and renormalising costs exactly
`KL(truncated ‖ full) = −log(retained mass)`, so the prediction's two
thresholds are one number, and 99 % mass is 0.01005 nats — 99 % would have
*failed* "KL < 0.01 nats" on its own. At K=64, T=1 the truncation error is
**0.0435 nats**: seven times the 0.00616 nats P6 measured for quantizing the
teacher to int8. The pilot spent a job establishing that the cheap teacher does
not distort the target, and the cache format built to hold its output distorts
it seven times more. Precision was being guarded at the wrong end of the
pipeline.

**The temperature result is the larger finding, and it was not predicted at
all.** Distillation runs at T > 1 — that is what softens the teacher into a
signal about the whole distribution rather than its argmax — and softening is
precisely what flattens the tail that top-K throws away. At T=4, K=128 retains
**2.8 %** of the mass: a truncation KL of 3.58 nats, against a student loss
around 1.27. The cache would be noise, not supervision. A top-K cache is
therefore not temperature-agnostic, and storing per-temperature normalisers —
which the format does — does not rescue it, because the mass that is missing
was never written down.

Two consequences, recorded here and carried into ADR-001:

* **A high-fidelity top-K cache is not affordable at the sizes Part 11 priced.**
  Reaching a truncation error comparable to P6's quantization error needs a K
  far beyond 128, and the storage estimate scales with it.
* **This shifts the A-vs-B decision toward design B**, the online split, which
  computes against the full distribution and never truncates. P4 predicted B
  wins at N=1 on node-hours; P3 says A's loss is not only in node-hours but in
  the signal itself.

*Caveat, and it matters for the size of the effect rather than its direction:*
this was measured on **Qwen3-30B-A3B-Base without the Phase-1 adapter**, which
was still training when the job ran. Continued pre-training on Italian case law
should make the teacher *more* peaked on that text, so these are a lower bound
on retained mass. Re-run with `--adapter` when Phase 1 finishes. The gap to
close at K=32, T=1 is 5.3 points, and the temperature collapse is too large for
an adapter to reverse.

*(2026-09-12, Qwen3-30B-A3B-Base, bf16, no adapter, transformers — no vLLM —
64 documents / 30,784 positions, seq_len 512.)*

**P1 holds, and the reasoning behind it was wrong.** The prediction expected
prompt scoring to be *slower* per position than generation-mode logprobs, with
memory as the binding constraint. It is **2.73× faster**: 1,074 positions/s
against 394. The falsification threshold was "within ±20 % of generation", and
2.73× clears it — in the opposite direction from the one the prediction
described.

The error is conceptual and worth naming. Scoring a prompt is prefill: every
position in one forward pass. Generation is autoregressive, one token at a
time. Prefill is bound to be faster per position, and "prompt_logprobs is
expensive" — true, relative to a prefill that does not compute them — got
confused with "slower than generation", which does not follow. A threshold
stated as a band rather than a direction is what let the prediction survive
its own reasoning being wrong, and that is an argument for stating thresholds
that way.

Two findings the probe was built to record rather than assume:

* **vLLM returns log-probabilities, not raw logits.** The values carry
  `decoded_token`, `logprob`, `rank`. ADR-001 Part 3 specifies a provider
  returning "RAW LOGITS, not log-probabilities"; that is now known to be
  unavailable from this backend, and the cache format's per-temperature
  normaliser is what absorbs the difference.
* **K is a property of the engine, not only of the request.** `max_logprobs`
  defaults to 20 and the engine *refuses* a larger request rather than
  truncating it. A cache at K=128 needs a teacher process configured for K=128
  before the first document is scored; raising K afterwards means standing the
  teacher back up.

*(2026-09-12, Qwen3-4B-Base, TP=1, 4 documents / 2,040 positions, seq_len 512.
A smoke run on the student-sized model, to separate the API question from the
30 B MoE on four GPUs. Throughput includes a Triton JIT spike on the first
batch — 2.19 it/s against 6.34 on the second — so the steady-state figure is
higher than the one reported.)*

*(P6: 2026-09-11, Qwen3-30B-A3B-Base at int8 against bf16, no adapter, 200
validation documents / 97,773 scored positions, seq_len 512.)*

Run twice, and the second run is worth reporting as a method result of its
own:

| | 20 docs, 10,186 positions | 200 docs, 97,773 positions |
|---|---|---|
| mean KL (nats) | 0.006125 | 0.006160 |
| top-1 agreement | 95.651 % | 95.430 % |
| top-5 agreement | 99.951 % | 99.955 % |

Ten times the data moved the KL by 0.6 % and top-1 by two tenths of a point.
So the first run was not a small sample producing a noisy number — these are
properties of the quantized model on this corpus, and a distributional
comparison of this kind converges on the order of ten thousand positions.
Worth knowing before sizing the rest of the pilot: the expensive run bought
confidence, not a different answer.

**The two halves disagree, and the disagreement is the finding.** Top-5
agreement is 99.95 % and the KL is 0.006 nats, so where the argmax flips, the
first two candidates were near-tied: the change is in which of two almost
equal probabilities wins, not in the distribution. Distillation optimises a
divergence against the teacher's probabilities, not agreement on its argmax,
and 0.006 nats against a student loss around 1.27 is about 0.5 % of the
signal.

So the question P6 was asked to answer — *did quantizing the teacher to make
it fit distort the target?* — is answered **no**, and v1.0's Phase-1 adapter
needs no asterisk.

But the prediction was posed badly, and that is recorded rather than quietly
repaired. Top-1 agreement was chosen because it is the half a reader can
interpret without information theory; it measures the stability of an argmax,
not the fidelity of a distribution, and on formulaic Italian legal text — full
of positions with two near-equiprobable continuations — the argmax is unstable
even between two copies of the same model. The threshold that should have been
pre-registered is KL alone, with top-1 reported as context. Per the
pre-registration's own scoring rule, a badly posed prediction is itself a
result.

## Incidents log

One line each, with the cost. This is the section that makes the report
worth more than a tidy methods description.

| Date | Incident | Cost | Fix |
|---|---|---|---|
| 2026-09-08 | Configured teacher/student ids (`Qwen3-32B-Base`, `Qwen3-7B-Base`) do not exist on the Hub | prefetch aborted, replanning | model survey, real ids |
| 2026-09-08 | `RE_CF` anchored with `\b` missed 59 codici fiscali while reporting a clean run | corpus rewrite before launch | shape-only match; sweep gates the launch |
| 2026-09-08 | Phase 2/3 launchers pointed at pre-rename checkpoint paths | would have failed a phase each | repointed |
| 2026-09-08 | Pre-flight verified the smoke model's tokenizer, not the job's | false green | `--tokenizer-model` |
| 2026-09-08 | ZeRO-3 dense-tuned limits OOM'd at first forward on the MoE teacher | 0.67 node-hours | `ds_zero3_moe.json` |
| 2026-09-08 | 20+ minutes of silent job, diagnosed by hand from `/proc` | ~30 min of operator time | job heartbeat |
| 2026-09-15 | `eval_steps: 1000` was config nothing read — no validation loop existed, so 18,000 steps of Phase 2 produced no held-out number and a flat training curve could not be interpreted | the whole v1.1 Phase 2 has no validation curve; unrecoverable after the fact | `evaluate()`, capped at `eval_max_batches` |
| 2026-09-15 | `save_steps: 1000` against a 24 h walltime: job 57353619 reached ~step 14,890 and its successor resumed from checkpoint-14000 | 890 steps, 2 h 40 m, per link boundary | `save_steps: 300` with `save_total_limit` bounding the directory |
| 2026-09-20 | A base-model student shipped with Qwen3's *Instruct* chat template, inherited from the base tokenizer. Recent llama.cpp auto-enables conversation mode when a template is present, so every prompt arrived wrapped in a format the model had never seen: it looped, echoed its input, emitted stray subword tokens, and read as broken | two wrong diagnoses (under-training, then a mis-mapped EOS) before the template was suspected | template dropped at export; the same file then continued a Cassation incipit into correct legal Italian citing art. 365 c.p.c. |
| 2026-09-20 | `quantize_to_gguf.sh` built only `llama-quantize` and `llama-cli`, so the one quantitative check — perplexity against the untouched base — needed a binary that was never compiled | the measurement went untaken while generations were judged by eye | `llama-perplexity` added to the build targets |
| 2026-09-20 | The same script defaulted `GGUF_NAME` to the literal `legal-it-7b` and `LCPP_DIR` to `$HOME` (50 GB on Leonardo) | every model in a series would have carried one stale, identical name | named after the source directory; `$WORK` preferred |
| 2026-09-20 | Merging a 4 B student in BF16 was OOM-killed on a login node — twice. "CPU only" means no GPU, not small: the merge holds gigabytes of weights and the GGUF conversion writes 8 GB, against whatever per-user memory a shared login node has left | two dead runs, ~40 min | both steps moved to `lrd_all_serial` with 30 G; the scripts' headers now say so |
| 2026-09-20 | The second of those kills landed at 91 % of writing the F16 GGUF, leaving a 7.3 GB truncated file. Nothing reported an error, and the next run's "F16 already present — skipping conversion" would have quantized the truncated model into one that loads and is silently wrong | caught before it propagated | conversion and quantization write a `.partial` and rename on success; the export writes a `.partial` directory and moves it into place only when complete |
| 2026-09-20 | A re-run after fixing an export silently reused the GGUF built from the *previous* export, so the corrected model was never converted and the smoke output was byte-identical — reading as "the fix did nothing" rather than "nothing ran" | one wasted diagnosis cycle | the conversion compares mtimes and reconverts when the source directory is newer |
| 2026-09-20 | `llama-cli` with stdin at `/dev/null` and no single-turn flag does not exit — it generates, returns to its `> ` prompt, reads EOF, reprints, and spins. Piped into `head`, the resulting SIGPIPE surfaced as exit 141, which the script reported as "the GGUF is malformed" about a model that had just written competent legal Italian | a false failure on a healthy model, on top of a hang | the single-turn flag is read from `--help` rather than guessed, output goes to a file instead of a pipe, and a timeout bounds the step |
| 2026-09-20 | `perplexity_compare.sh` defaulted to 8 threads regardless of the allocation, so two twenty-minute measurements ran oversubscribed on 4 cores | ~2× on two measurements | threads default to `SLURM_CPUS_PER_TASK`; the next run went from ~20 min to 6.5 |
| 2026-09-21 | `format_pretraining.py` splits train/val over **chunks, not documents**, so fragments of the same ruling sit on both sides. The first quality result was therefore reported as held-out when it is held-out per chunk and overlapping per document — the student was scored on passages whose neighbours it had trained on | the −48 to −54 % figure demoted to an upper bound; direction unaffected | caveat written into the report the day it was found; group by `sentence_id` before splitting for the next corpus; a clean number needs text the training never touched |
| 2026-09-20 | The export job packaged GGUFs that nothing evaluated. The first three perplexity measurements were run by hand and existed only in a terminal scrollback | not a record, and not citable; a run that stopped improving would have said so only when somebody next remembered to look | the export job now measures each new GGUF against a fixed base on a fixed held-out corpus and appends a row to `exports/perplexity.csv` |
| 2026-09-15 | Throughput reported as a cumulative mean since job start, never reset — printed an identical 1.48 for thirteen hours and could not have shown a slowdown | none yet; a latent blind spot on the metric used to size the chain | per-window rate, cumulative kept beside it |

## Administrative

- **The EuroHPC Final Report is obligatory**, not optional: the PI submits it
  within three months of the allocation completing, on the EuroHPC JU
  template, to EuroHPC Peer-Review, and failure to submit can disqualify
  future proposals from any member of the research group. This allocation
  ends 02/11/2026, so it is due by **02/02/2027**. This outline is its
  skeleton as well as Zenodo's — write once, submit twice. Allocation
  strategy and the backlog that fills it:
  [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md).
- Still to confirm with CINECA or the EuroHPC portal, not extractable from
  the published call PDFs: the exact Final Report template, and the required
  acknowledgement wording.
- Acknowledgement of the allocation (EHPC-AIF-2026PG01-1147) is required in
  any publication.
- Corpus stays private: pseudonymised, and `sentence_id` / `source_id`
  re-identify each ruling in a public archive. The report describes the
  pipeline; it does not ship the data.
