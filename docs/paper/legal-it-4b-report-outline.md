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

| Quantity | Value |
|---|---|
| BF16 student size | |
| Q4_K_M GGUF size | |
| Perplexity before / after quantization | |
| Tokens/s on the target consumer GPU | |

### Budget

| Quantity | Value |
|---|---|
| Node-hours allocated | 1,250 |
| Node-hours spent, by phase | |
| Node-hours lost to failed runs | 0.67 (job 56760964, ZeRO-3 OOM) |
| Queue wait time, total | |

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
