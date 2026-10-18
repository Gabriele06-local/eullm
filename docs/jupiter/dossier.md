# EHPC-AIF-2026PG01-1434 on JUPITER Booster — project dossier

*Porting and scaling a teacher–student verticalisation pipeline for European
domain-specific LLMs to Grace Hopper.* EuroHPC AI Factory Playground, JUPITER
Booster (JSC), 5,000 GPU-hours over two months, budget `e-aif-2026pg01-1434`.
Principal Investigator: Francesco Marchetti, I3K Technologies Srl (Milan).

This dossier is the running record of the allocation: what was set up, what
was measured, what went wrong and how it was fixed, and what the hours were
spent on. It is kept with the code (`docs/jupiter/dossier.md` in the EULLM
repository) and updated as the work goes; the working notes behind it are in
`docs/jupiter/allocation-plan.md`. Every figure below comes from a job on
JUPITER, named by its Slurm job id.

**Last updated: 16-10-2026, morning.**

## 1. At a glance

| | |
|---|---|
| Awarded | 06-10-2026; first job on 14-10-2026 |
| System | JUPITER Booster: 4× GH200 (H100 96 GB HBM3, 72-core Neoverse-V2 Grace) a node, whole nodes only |
| Spent so far | 21.3 GPU-hours by 15-10-2026 evening, and 13.0 more for the five jobs of the night to 16-10 (`tools/jupiter/budget.sh` gives the running total); planned to 15-12-2026, two months from activation (JuDoor shows 31-12-2026, kept as the margin): about 82 GPU-hours a day from 16-10, 20 node-hours, one node busy 85% of the time |
| Codes | EULLM Engine (Rust, llama.cpp), EULLM Forge (Python: PyTorch 2.9.1 of the site, Transformers 5.19, PEFT 0.21) |
| Data on JUPITER | Legislation (Normattiva, EUR-Lex) and public benchmarks; pseudonymised court decisions once JSC's data-protection addendum is signed, asked for on 15-10-2026 (§7) |

## 2. What the proposal committed to, and where each stands

| # | Deliverable | Leonardo baseline (A100 64 GB) | Status on JUPITER |
|---|---|---|---|
| 1 | The pipeline runs on aarch64 + GH200, unattended, evaluation included | runs end to end | Distillation, instruction tuning, generation and evaluation ran unattended overnight, job after job (§5.3, §5.5) |
| 2 | Distillation throughput per step, teacher in BF16 on one GPU | ~380 optimiser steps/h, teacher on 2 GPUs + student on a 3rd | **548–553 steps/h** with 2,048-token micro-batches, teacher alone on one GPU (§5.2) |
| 3 | Two runs per node (2 GPUs each) | 1 run per 3 GPUs | **Done**: two runs side by side at the same speed (§5.2, §5.3) |
| 4 | Instruction tuning time | 4B, 3,400 examples, 2 epochs: 46 min on one GPU | **24.4 min** for the 4B on 4,272 pairs, 2 epochs, one GH200: 0.34 s a pair against 0.81, 2.4× per GPU; the 8B 28 min; four runs side by side (§5.5) |
| 5 | A faster inference backend for synthetic data generation | 58 tok/s on 3 A100, 30B-A3B-Instruct in plain Transformers | **Done: 2,279 tok/s on one node (4 GH200), same model, EULLM Engine: 39× per node** (§5.4) |
| 6 | Quality on the fixed held-out set | e.g. the 5.11 → 5.92 perplexity regression it caught | Ten seed questions: no difference between the models (0.39–0.46 keyword coverage). A 217-question exam drawn from the text of the law, being sat by every model (§5.5) |
| 7 | Go/no-go on the Qwen3.5 family | — | Probe: the pair loads and generates as fast as Qwen3; a training step took 1.79 s on the reference kernels, the fast ones now installed; distillation on legislation queued (§5.6) |
| 8 | GPU-hours per distillation epoch and per complete vertical | — | Legislation, 3 epochs: **2.98 GPU-hours an epoch a run** unpacked, **0.52 packed** (§5.3) |
| 9 | Energy use and carbon footprint | — | GPU energy per job: generation, instruction tuning, distillation (§6) |

## 3. Log

**14-10-2026**
* Login-node probe of the environment (`forge/scripts/jupiter/probe_env.sh`):
  aarch64, `booster` hands out whole nodes (a job pays 4 GPU-hours an hour),
  modules GCC 14.3, CUDA 13, Python 3.13.5, PyTorch 2.9.1.
* EULLM Engine built for sm_90 on the login node. Two build failures fixed in
  the code: the CUDA static libraries were looked for only under
  `/usr/local/cuda` (the site's toolkit is a module), and the site's GCC does
  not produce position-independent code by default (the binary is now linked
  `-no-pie`).
* Job 2326280: compute nodes have no internet and no `git`; models and data
  are staged from a login node.
* Job 2326571: first engine comparison on GH200, EULLM Engine against
  llama-server at the same llama.cpp commit (§5.1).
* Jobs 2326997, 2326998: model-load investigation; GPU probe for Forge.
* Jobs 2327213, 2327214: loads four at once; second round of the runtime
  comparison (40 points). Job 2327297: Forge probe, stopped at its first
  import (venv path order, fixed).

**16-10-2026**
* JSC's data-protection assessment form for the pseudonymised court
  decisions: a first submission wrongly declared none, withdrawn and
  replaced (§7).
* EU acts in force in Italian (17,563 regulations and directives) being
  fetched outside JUPITER for a larger legislation corpus; the processors
  (`prepare_eurlex.py`, `prepare_legislation.py --all-acts`) are in the
  repository.
* Venv: flash-linear-attention 0.5.2 for Qwen3.5's linear attention (on the
  module's Triton 3.5.1); Qwen3.5-9B-Base downloaded; the held-out exam drawn
  from the legislation (217 questions). Queued: every model on the exam,
  instruction tuning on 9,223 pairs, the Qwen3.5 probe with the fast
  kernels, then the Qwen3.5 distillation (§5.7).

**15-10-2026, night**
* Job 2338844: third link; both legislation runs complete and merged.
  Job 2339283: the same runs packed, in 47 minutes (§5.3).
* Job 2340154: instruction tuning, four runs on one node, then the ten seed
  questions (§5.5). Job 2340155: 4,951 more pairs (§5.4).
* Job 2340254: the Qwen3.5 pair on GH200 (§5.6).

**15-10-2026**
* Job 2339281: 4,272 instruction pairs generated from the legislation corpus
  by EULLM Engine in 10 minutes of one node, 2,279 tok/s (§5.4).
* Job 2334585: first Forge numbers on GH200 (§5.2).
* Engine fix published: a model on GPFS stays memory-mapped (§5.1).
* Legislation corpus copied from Leonardo and split by article (7,064 chunks
  to train on, 234 held out).
* Jobs 2337122–2337123: the distillation failed at its first forward step
  (teacher inputs on the wrong GPU), fixed in `distill.py`. Job 2337859: the
  chain runs (§5.3).
* Sequence packing added to `distill.py`, and the EULLM Engine backend to the
  synthetic-data generator, both tested; JUPITER jobs written for each.

## 4. Environment on JUPITER

* **Forge** runs on the site's PyTorch module (torch 2.9.1, built for sm_90),
  with a virtual environment on top for Transformers 5.19.0, PEFT 0.21.2,
  Accelerate 1.15.0 and Datasets 5.1.0 — never a second torch from pip.
* **Models** in the project storage: Qwen3-30B-A3B-Base (teacher, 61 GB in
  BF16), Qwen3-4B-Base and Qwen3-8B-Base (students), and GGUF models for the
  engine. All Apache 2.0.
* **Job shape**: one node, two runs side by side, each on a GPU pair and its
  two Grace sockets (GPUs 0–1 with cores 0–143, 2–3 with 144–287), in
  two-hour links that resume from their newest checkpoint.
* The scripts: `tools/jupiter/` (engine), `forge/scripts/jupiter/` (Forge).

## 5. Results

### 5.1 EULLM Engine on GH200 (deliverable 5)

Against llama-server at the same llama.cpp commit, Q4_K_M, one GH200 a point,
tokens/s (EULLM Engine / llama-server; jobs 2326571 and 2327214):

| model | 1 request | 16 requests | 32 requests | 16 clients, 400 graded gsm8k |
|---|---|---|---|---|
| Qwen3-8B | 229 / 226 | 1,434 / 1,412 | | 1,156 / 1,066 |
| Qwen3-14B | 142 / 142 | 1,066 / 1,044 | | 783 / 738 |
| Qwen3-32B | 70 / 70 | 589 / 588 | 875 / 778 | 439 / 421 |
| Qwen3.6-27B (hybrid) | 75 / 74 | 513 / 394 | 700 / 392 | 432 / 343 |
| Qwen3.6-35B-A3B (MoE) | 218 / 216 | 888 / 723 | | 646 / 576 |

Equal on dense models one request at a time; ahead with many requests on the
hybrid and MoE models (up to 79% with 32 on Qwen3.6-27B), at the same
accuracy on the graded workload. One GH200 writes 888 tok/s with a
35B-A3B MoE, where Leonardo's synthetic-data generation ran at 58–64 tok/s;
the like-for-like run (same model, Qwen3-30B-A3B-Instruct-2507) is prepared
(§5.4).

**Model loading.** The engine first took 107–184 s to load where
llama-server seemed to take 2–12 s. Part of that was the measurement (the
runner did not count llama-server's load); the rest was real: the engine
read models into memory on network file systems, a rule measured right on
LUMI's Lustre, and on JUPITER's GPFS that is the slow way (job 2326997, one
16.8 GB model):

| | read into memory | memory-mapped |
|---|---:|---:|
| EULLM Engine | 53 s | 10 s |
| llama-server | 48 s | 4 s |

Four servers loading one 22 GB file at once: 34 s read in, 16 s mapped
(job 2327213). The engine now keeps models on GPFS mapped (commit
`ca0d19d`); Lustre keeps the rule.

### 5.2 The distillation step on GH200 (deliverables 2 and 3)

Job 2334585, the Leonardo recipe (Qwen3-30B-A3B-Base teacher in BF16,
Qwen3-4B-Base student with LoRA r128 and gradient checkpointing, 2,048-token
micro-batches, 16 per optimiser step), two runs side by side on one node:

| | JUPITER (GH200) | Leonardo (A100 64 GB) |
|---|---|---|
| GPUs per run | 2 (teacher alone on one, in BF16) | 3 (teacher split over two) |
| One micro-batch | 0.41 s | — |
| Optimiser steps an hour, per run | **548–553** | ~380 |
| Steps per GPU-hour | ~276 | ~127 (**2.2×**) |
| Teacher memory | 57 GiB of 95 | — |

Loading the 61 GB teacher takes ~3.5 minutes per job (two processes reading
it at once). The teacher generating with plain Transformers in BF16 on one
GH200: 13.9 tok/s alone, 113.5 with 8 sequences, 346.9 with 32.

### 5.3 First distillation on legislation

Two arms on one node, 4B and 8B students, the Base teacher, three epochs over
the legislation corpus (1,324 optimiser steps), in two-hour links.

* First attempt (jobs 2337122–2337123, 13 minutes of node time): both runs
  stopped at the first forward step. The teacher, alone on one GPU, gets no
  device hooks from Accelerate, and the batch stayed on the student's GPU.
  Fixed in `distill.py` (commit `ce822dc`) with a test.
* Jobs 2337859, 2337861, 2338844: both runs train, unattended, link after
  link. At 1 h 39 min the 4B was at step 560 and the 8B at step 500
  (evaluation at step 400: loss 0.989, KL 0.824, CE 1.376); the second link
  ended with both at step 1,180 of 1,324, and the third resumes from the
  checkpoint of step 1,100.
* **Throughput lesson.** The legislation chunks are short (~210 tokens), and
  one chunk a micro-batch ran at 1.5–1.6 micro-batches a second, ~340
  tokens/s against ~5,000 with 2,048-token micro-batches: the fixed cost of
  each forward was the whole cost. Sequence packing (several chunks in one
  2,048-token sequence, each attending only to itself; commit `ebc56c4`)
  was added and the same runs repeated packed (job 2339283).

| | unpacked (2337859, 2337861, 2338844) | packed (2339283) |
|---|---|---|
| Node time, both runs, 3 epochs | 4 h 28 min | **47 min** |
| GPU-hours an epoch, one run | 2.98 | **0.52** |
| Optimiser steps | 1,324 | 206 |
| 4B, held-out loss / KL / CE | 1.021 / 0.854 / 1.412 (step 1,200) | 1.013 / 0.888 / 1.304 (step 200) |
| 8B, held-out loss / KL / CE | 0.938 / 0.762 / 1.347 (step 1,200) | 0.923 / 0.790 / 1.232 (step 200) |

Packed reaches the same held-out loss in a sixth of the node time: KL to
the teacher slightly higher, cross-entropy on the text lower. Packing is
the default for short-document corpora from now on, for the models it is
correct for (not Qwen3.5, §5.6).

### 5.4 Synthetic instruction data with EULLM Engine (deliverable 5)

Job 2339281: one EULLM Engine server per GH200, 32 requests each,
Qwen3-30B-A3B-Instruct-2507 (Q8_0 GGUF), the generator Leonardo ran in plain
Transformers, with the same prompts, sampling and quality filters, on the
training split of the legislation corpus:

| | JUPITER, EULLM Engine | Leonardo, Transformers |
|---|---|---|
| GPUs | 4 GH200 (one node) | 3 A100 64 GB |
| Generation speed | **2,279 tok/s** | 58 tok/s (pilot, job 58559305) |
| Per GPU | ~570 tok/s | ~19 tok/s |
| Pairs | 4,272 accepted of 4,384 (97%) in 8.4 min of generation, 10 min of node | 6,000 estimated at 7–8 node-hours |
| Energy (GPUs) | 0.20 kWh, ~5,700 generated tokens per Wh | — |

39 times the throughput per node, 29 times per GPU. The 112 rejected pairs
were turned down by the filters: 98 referred to "the text" instead of
answering, 11 were about a single case, 3 were not valid JSON. The weights
differ in precision (Q8_0 against BF16); the quality of the pairs is
measured downstream, in the instruction tuning.

Job 2340155 asked the same passages again with three more seeds: 4,951 new
pairs in 10 min 52 s of node time (0.22 kWh), 9,223 in all. Fewer than three
times 4,272, because a short passage asked the same task gives the same
question, which the generator skips as already done; more pairs need more
text, not more seeds.

### 5.5 Instruction tuning (deliverable 4) and the first held-out check (6)

Job 2340154: Leonardo's stage 3 (LoRA r32, 2 epochs, the identity examples
mixed in) on the 4,272 pairs, four runs side by side on one node, one GH200
each; then each model merged and asked the ten legal-it seed questions
(keyword coverage), with Qwen3-4B-Instruct-2507 as the reference.

| run | training on one GH200 | keyword coverage | ended its turn |
|---|---|---|---|
| 4B distilled on legislation | 24.4 min | 0.385 | 10/10 |
| 4B Base, not distilled | 24.5 min | 0.410 | 9/10 |
| 8B distilled on legislation | 28.3 min | 0.447 | 10/10 |
| 8B Base, not distilled | 27.2 min | 0.455 | 10/10 |
| Qwen3-4B-Instruct-2507 (reference) | — | 0.448 | 3/10 |

Leonardo's figure is 46 minutes for 3,400 pairs on one A100: 0.81 s a pair,
here 0.34 s, **2.4 times per GPU**, and four runs at once on a node.

On the ten questions no model is ahead of another: the spread is what ten
questions give by chance, and distillation on 1.5M tokens of legislation
shows nothing there. Ten questions cannot tell, so the exam drawn from the
law (217 questions on article contents, deadlines and articles that do not
exist; reported in totals only) is now sat by every model (§5.7).

### 5.6 The Qwen3.5 pair on GH200 (deliverable 7)

Job 2340254, Qwen3.5-35B-A3B-Base teaching Qwen3.5-4B-Base (Apache 2.0,
same 248,320-token vocabulary), two pairs side by side on one node:

| | Qwen3.5 (2340254) | Qwen3 (2334585) |
|---|---|---|
| Teacher load / memory | 69 s / 64.6 GiB | ~200 s / 57 GiB |
| 2,048-token micro-batch | 1.79 s | 0.41 s |
| Optimiser steps an hour at 16 | 125 | 548–553 |
| Teacher generating, 1 / 8 / 32 sequences | 11.4 / 92.7 / 369.6 tok/s | 13.9 / 113.5 / 346.9 tok/s |

Transformers loads the multimodal checkpoints as text-only models, and
generation runs as fast as Qwen3's. Training was four times slower because
the linear-attention layers (three in four) ran Transformers' reference
PyTorch path: flash-linear-attention was not installed. It is now (0.5.2,
on the module's Triton). Packing is not used for Qwen3.5: its
linear-attention layers carry a state along the sequence that the
per-document mask does not reset.

### 5.7 Running on 16-10-2026

* Every model of §5.5 on the 217-question exam.
* Instruction tuning on the 9,223 pairs: the 4B distilled unpacked and
  packed, the 4B Base, the 8B packed (packed against unpacked, distilled
  against not, and more pairs).
* The Qwen3.5 probe again with the fast kernels, then the Qwen3.5
  distillation on legislation, 4B and 9B students, in two-hour links.

### 5.8 The held-out exam, graded (17-10-2026, deliverable 6)

Every model of §5.5–5.7 sat the 217-question exam twice: closed book, and
open book, each question with the articles BM25 finds for it in the
legislation in front. Qwen3-30B-A3B-Instruct-2507 graded every answer
against the text of the article (correct, partial, wrong; rubric v3, as on
Leonardo; score = correct + half the partial, over 217), four graders on
one node (job 2371747, 34 minutes, 0.44 kWh). Totals only: the questions
stay held out.

| model (instruction-tuned on the same pairs) | open book | correct |
|---|---:|---:|
| Qwen3.5-9B distilled on legislation | **0.811** | 175 |
| Qwen3.5-9B Base | 0.804 | 172 |
| Qwen3.5-4B Base | 0.781 | 168 |
| Qwen3-8B distilled, packed (9,223 pairs) | 0.767 | 165 |
| Qwen3-4B-Instruct-2507 (reference, as released) | 0.765 | 159 |
| Qwen3.5-4B distilled on legislation | 0.765 | 164 |
| Qwen3-8B distilled (4,272 pairs) | 0.737 | 158 |
| Qwen3-4B distilled, packed (9,223 pairs) | 0.730 | 157 |
| Qwen3-4B Base (4,272 / 9,223 pairs) | 0.717 / 0.714 | 153 / 152 |
| Qwen3-8B Base (4,272 pairs) | 0.712 | 154 |
| Qwen3-4B distilled (9,223 / 4,272 pairs) | 0.703 / 0.694 | 147 / 149 |

**The instruction pairs overlap the exam.** The exam was drawn from every
Normattiva article and the pairs from passages of the training split, with
no article left out of either. Traced back to their passages
(`pairs_exam_overlap.py`, 18-10-2026): 251 of the 9,223 pairs ask about an
exam article, and 105 of the exam's 153 articles have pairs about them. The
reference was trained on none of them, so every figure of ours above may be
inflated against it; the instruction tuning is rerun on the 8,972 pairs
that touch no exam article (§5.9), and only that comparison with the
reference stands.

Closed book every model, the reference included, scores 0.000–0.023: none
knows the articles by heart. Open book, the two Qwen3.5-9B models score
ahead of the reference (+10 questions of 217), pending the rerun without
the overlap. A
model distilled and the same model not distilled stay within a few
questions of each other in every pair: 1.5M tokens of legislation move
nothing measurable; the family and the size of the student do. Whether a
corpus a hundred times larger does is what norms-v2 asks (§5.9).

### 5.9 norms-v2: the codes with the EU acts in Italian (from 17-10-2026)

The Italian codes with 17,529 EU acts in force, in Italian, fetched from the
Publications Office (`prepare_eurlex.py`): 392,538 training chunks, ~220M
tokens. A fixed sample of 250,000 (~139M tokens) and 3,000 held out is the
corpus of a packed distillation chain, 4B and 8B students, one epoch, 7,224
optimiser steps, two-hour links.

* A link killed by its time limit while saving left a truncated
  `training_state.pt` (8B, checkpoint-1100), and every later link of the 8B
  failed at once reading it. The state is now written under a temporary name
  and renamed, and a truncated one is skipped for the one before
  (`18f0611`); the 8B resumed from checkpoint-1000 (job 2371817).
* Each link spent 50–70 minutes of its two hours loading the teacher (links
  2362188–2362191 and 2371817; 3.6 minutes for the first link, 12 for
  2371816), against 3.5 minutes on 14-10: the 61 GB read from `$PROJECT`
  twice, once per run, in many small reads at once, as fast as the shared
  file system allows at that hour. The four graders of job 2371747 loaded
  theirs quickly the same day. The teacher is now copied once per node into
  memory (`/dev/shm`, a few large reads) and both runs load it from there
  (`9d093b7`), from link 2375535: 49 s for the copy, the runs training
  after 4–5 minutes instead of an hour.
* Halfway (4B at step 3,400, 8B at 2,800, jobs 2381934–2381935), instruction
  tuned and examined open book: the 4B 0.712 against 0.714 for the 4B Base,
  the 8B 0.740 against 0.751 for the 8B Base. Open book the article is in
  front of the model, so the exam measures reading it, which distillation
  on the law is not expected to change; its effect is the held-out loss.

## 6. Energy

Slurm keeps no energy for jobs on JUPITER Booster (`ConsumedEnergy` empty), so
every Forge job samples the GPUs' power with `nvidia-smi` (every 1–5 s) and
sums it (`forge/scripts/jupiter/energy.py`). Idle, a node's four GPUs draw
~390 W together (job 2327297). The carbon footprint will be derived from the
figures below with the site's energy mix.

| job | what | GPU energy |
|---|---|---|
| 2339281 | 4,272 instruction pairs, EULLM Engine | 0.20 kWh (1,359 W mean) |
| 2340155 | 4,951 more pairs | 0.22 kWh (1,357 W mean) |
| 2340154 | four instruction-tuning runs and their evaluation | 1.21 kWh in 1.43 h (844 W mean: the evaluations leave the GPUs mostly idle) |
| 2340254 | Qwen3.5 probe | 29 Wh training loops, 15 Wh generating |

## 7. Data governance

* **What the proposal declared.** The project handles personal data:
  pseudonymised Italian court decisions (Corte di Cassazione 2021–2026,
  Consiglio di Stato 2017–2024), as the call's form required us to state,
  and we committed to a data processing agreement with the production site
  before any of it is uploaded. The Principal Investigator's Agreement
  (14-10-2026) is JSC's standard form, which excludes personal data without
  such an agreement.
* **The agreement is being sought.** On 15-10-2026 JSC was asked by e-mail
  for it; JSC answered with its Data Protection Team's assessment form, the
  step before the addendum to its terms of use. On 16-10-2026 the form was
  first submitted by mistake declaring no personal data; that submission is
  withdrawn by e-mail to the Data Protection Team and replaced by one for
  the pseudonymised decisions.
* **Until the addendum is signed** JUPITER holds legislation (Normattiva,
  EUR-Lex: public), public benchmarks, synthetic pairs generated from them
  and ruling summaries that carry nothing of the case, and no court
  decision is uploaded. The teacher on JUPITER is accordingly the Base
  model, without the continued-pre-training adapter trained on the
  decisions on Leonardo.
* **The measures declared, which bind the processing once allowed:** names
  of parties, lawyers and magistrates replaced; tax codes, VAT numbers,
  IBANs, addresses, phone numbers, e-mail addresses and dates of birth
  removed; surnames in case citations and links to the original documents
  removed; a document still holding a structured identifier rejected by an
  automatic check; generated examples about a single case, or still holding
  a placeholder, discarded; the data only on the project's storage, moved
  only over SSH, never in a repository or a third-party service, and deleted
  at the end of the project; decisions of 2025–2026 used for evaluation
  only, never for training.

## 8. Issues found and fixed

| when | issue | fix |
|---|---|---|
| 14-10 | Engine build could not find CUDA's static libraries in the site's toolkit | build looks in the toolkit `CUDA_PATH` names |
| 14-10 | Link error: site GCC does not make position-independent code | `-no-pie` for the engine binary |
| 14-10 | Benchmark runner counted models as installed from the catalogue listing | reads only the installed-models table |
| 14-10 | Load times compared unlike things (llama-server's load not counted) | load test timed from process start (`sbatch_load.slurm`) |
| 15-10 | Engine read models into memory on GPFS, 5× slower than mapping | GPFS models stay mapped (`ca0d19d`) |
| 15-10 | Forge venv: the module's older packages shadowed the venv's in jobs | venv first on the path in every job, imports checked before using a GPU |
| 15-10 | Teacher alone on one GPU received its inputs on the student's GPU | inputs moved to the teacher's device (`ce822dc`) |
| 15-10 | Short documents left the GPUs idle (~340 tokens/s) | sequence packing (`ebc56c4`): the same runs in a sixth of the time |
| 16-10 | Qwen3.5's linear attention ran on Transformers' reference PyTorch path, 4× slower than Qwen3 | flash-linear-attention 0.5.2 in the venv |
| 16-10 | flash-linear-attention 0.5.2 refuses the module's Triton 3.5.1 on Hopper (wrong gradients up to 3.7.0) | Triton 3.8.0 in a directory of its own for the Qwen3.5 jobs (`9f04bd6`) |
| 17-10 | A link killed while saving left a truncated checkpoint; every later link failed on it | atomic save, truncated state skipped on resume (`18f0611`) |
| 17-10 | Loading the teacher took 50–70 minutes of a two-hour link | teacher copied once per node into memory (`9d093b7`) |
| 18-10 | 105 of the exam's 153 articles had instruction pairs about them | overlap traced (`d746b11`); instruction tuning rerun on the 8,972 pairs without it |

## 9. Resource use

| job | what | node time |
|---|---|---|
| 2326280 | engine comparison, first try (models missing) | 2 min 27 s |
| 2326571 | engine comparison, 22 points + conversations | 17 min 23 s |
| 2326997 | model-load test, six loads | 10 min 9 s |
| 2326998 | GPU probe for Forge | 1 min 15 s |
| 2327213 | model-load test, four servers at once | 2 min 0 s |
| 2327214, 2327215 | engine comparison, 40 points | 15 min 54 s |
| 2327297 | Forge probe, stopped at the first import | 1 min 22 s |
| 2334585 | Forge probe: distillation step, generation, energy | ~5 min 42 s |
| 2337122, 2337123 | distillation, stopped at the first forward | 13 min 16 s |
| **completed** | | **~69.5 min of node time ≈ 4.6 GPU-hours** |
| 2337859, 2337861 | distillation chain on legislation, links 1–2 | 3 h 43 min |
| 2339281 | synthetic instruction data, EULLM Engine | 10 min 2 s |
| 2338844 | distillation chain, link 3 | 44 min 52 s |
| 2339283 | distillation on legislation, packed | 47 min 12 s |
| 2340154 | instruction tuning, four runs, evaluation | 1 h 27 min 30 s |
| 2340155 | 4,951 more instruction pairs | 10 min 52 s |
| 2340254 | Qwen3.5 probe | 5 min 14 s |

## 10. Next

1. norms-v2 to the end of its epoch, its checkpoints instruction-tuned
   and sitting the exam as they come (deliverables 2, 3, 6).
2. Qwen3.5-9B, the best student so far, on norms-v2 (deliverable 7).
3. A corpus large enough for the remaining hours: more legislation
   (EUR-Lex in Italian, the rest of Normattiva) and ruling summaries from
   Leonardo; generated pairs grow with the text, not with seeds (§5.4).
4. One complete vertical unattended, corpus to GGUF to evaluation, timed
   (deliverables 1, 8).
