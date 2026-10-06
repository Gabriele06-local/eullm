# Allocation plan — EHPC-AIF-2026PG01-1434 (JUPITER Booster)

> **5,000 GPU-hours on JUPITER Booster (JSC), two months**, AI Factory
> Playground. Awarded 06-10-2026; JSC communicates the start and end dates.
> Proposal: *Porting and scaling a teacher–student verticalisation pipeline
> for European domain-specific LLMs to Grace Hopper*. Technical Assessment:
> accepted, "codes not installed — to be installed by the applicant".
> Sibling allocations: Leonardo Booster EHPC-AIF-2026PG01-1147 (Forge, ends
> 02-11-2026, [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md))
> and LUMI-G EHPC-DEV-2026D09-278 (engine, inference only,
> [`../lumi/allocation-plan.md`](../lumi/allocation-plan.md)).

## What we committed to

This is the Forge pipeline, not the engine campaign: distillation of a large
MoE teacher into a small student, instruction tuning, GGUF export and
automated evaluation, moved from Leonardo to GH200 and measured against the
Leonardo baseline. The proposal names its outputs, and the Final Report and
the Large Scale application after it are judged on exactly these:

| # | deliverable | Leonardo baseline (A100 64 GB), from the proposal |
|---|---|---|
| 1 | the pipeline runs on aarch64 + GH200, unattended, evaluation included | runs end to end |
| 2 | distillation throughput per step, teacher in **BF16 on one GPU** | ~380 optimiser steps/h, 4B student, 30B-A3B teacher split over 2 GPUs + student on a 3rd |
| 3 | **two runs per node** (2 GPUs each) | 1 run per 3 GPUs |
| 4 | instruction tuning time | 4B, 3,400 examples, 2 epochs: 46 min on one GPU |
| 5 | a faster inference backend for synthetic data generation | 58–64 tok/s, 30B-A3B instruct in plain Transformers |
| 6 | quality on the fixed held-out set, against the base model and Leonardo | e.g. the 5.11 → 5.92 perplexity regression it caught |
| 7 | go/no-go on the **Qwen3.5** family for the next models | — |
| 8 | GPU-hours per distillation epoch and per complete vertical | the sizing of the Large Scale application |
| 9 | energy use and carbon footprint, "as the call requires" | — |

Multi-node training is explicitly out of scope. Maximum 8 GPUs at once
(two nodes), 2 TB of storage, 1 TB of transfer.

Item 5 is where the engine enters: EULLM Engine serving the teacher's GGUF
with continuous batching on GH200 is the natural "faster backend", and the
proposal says the engine is "validated on the aarch64 host with CUDA" here.

## The number that governs everything

5,000 GPU-hours over ~61 days is **~82 GPU-hours a day: one node with all
four GPUs busy around the clock uses 96**. The allocation is spent by keeping
one node busy for about 52 of the 61 days, or two nodes for half of them.

Leonardo taught the cost of a slow start: its first two weeks ran at ~12
node-hours a day against 20 needed, because whole-node jobs of 4–24 h were
rarely placed, and only 2-hour resumable chains fixed it. The award letter
says the same thing from the other side: usage is expected to begin within
the first month, the AI Factory may allocate "according to a pre-defined
consumption pattern", and a prioritisation policy applies. Ask JSC for both
on the first day, and measure placement before committing to a job shape.

## Before the start date

1. **Account.** JuDoor registration, SSH key, 2FA; join the project when JSC
   sends it. Login is `login.jupiter.fz-juelich.de`.
2. **Data processing agreement — the blocker for Italian court decisions.**
   The proposal declares personal data (pseudonymised decisions are still
   personal data) and promises a DPA with the hosting site *before any
   personal data is uploaded*. Request it from JSC now. Until it is signed,
   only legislation (Normattiva, EUR-Lex: public domain) and public
   benchmarks go to JUPITER, which is enough for every throughput
   measurement (deliverables 1–5, 8) — throughput does not depend on which
   text is trained on.
3. **Environment probe**, `forge/scripts/jupiter/probe_env.sh`: aarch64,
   GPU count and memory, whether nodes are exclusive, how PyTorch is reached
   and whether its build has sm_90, transformers/peft/accelerate beside it,
   internet on compute nodes, energy accounting. Login node first, then one
   15-minute GPU job. Nothing else is written for JUPITER until it has run.
4. **Engine for aarch64 + sm_90.** The published `eullm-linux-arm64-cuda`
   binary targets sm_86/89/120, not Hopper; build on JUPITER with
   `CUDAARCHS=90`, as `tools/lumi/build_engine.sh` does for gfx90a on LUMI
   (the Cray-wrapper and libclang lessons from LUMI apply if JSC's
   environment is similar).

## Calendar, from the start date

| weeks | work | GPU-h (cumulative) |
|---|---|---:|
| 1 | probe; Forge venv on top of the site's PyTorch; one-GPU smoke (instruction tuning on legislation); teacher loads in BF16 on one GPU; placement probe (which job shapes start, how fast) | ~200 |
| 1–2 | **baseline replication**: the Leonardo distillation config, teacher BF16 on one GPU, student on another, two runs per node — steps/h against 380 (deliverables 2, 3); instruction tuning against 46 min (4) | ~1,000 |
| 2–3 | **data generation** with EULLM Engine on GH200 against 58–64 tok/s (5); quality of the generated set unchanged | ~1,600 |
| 3–6 | **production runs** on the Leonardo recipe, now with BF16 teacher; held-out evaluation of every checkpoint (6); in parallel, **Qwen3.5** teacher/student pair (7) | ~4,000 |
| 7–8 | one **complete vertical unattended**, corpus to GGUF to evaluation, timed end to end (1, 8); energy figures (9); Final Report and Large Scale sizing | 5,000 |

Every week's runs are chains of short resumable jobs, as on Leonardo, until
the placement probe says JUPITER rewards something else.

## Job shape — to be fixed by the probe, not assumed

* **Unit of work: one run = 2 GPUs** (teacher, student). If nodes are
  exclusive — likely on a JSC booster, to be confirmed — a job is one node
  running **two runs side by side**, each pinned to its two GPUs and its two
  Grace sockets' cores (`CUDA_VISIBLE_DEVICES`, `numactl`/`taskset`).
* **Duration:** the shortest link the queue places quickly; Forge resumes
  from checkpoint, including the position in the data, so short links cost
  minutes.
* **Two nodes at most** (8 GPUs): the second node is for the Qwen3.5 track
  or a second vertical, never for a multi-node job.
* The Leonardo rules in `forge/CLAUDE.md` ("the shape of a job that starts")
  are the template; JUPITER gets its own version and its own
  `test_*_job_shape.py` once the probe has spoken.

## Lines not to cross

* **Personal data only after the DPA**, and only pseudonymised, as declared.
  Nothing from the corpus leaves the project storage; it is never in a
  repository.
* **Only permissive base models** (Qwen Apache-2.0, as declared); the
  Qwen3.5 pair is checked against the Hub licence before download.
* **No training on LUMI**: the same proposal told EuroHPC that LUMI DEV-278
  "covers inference-engine work only; no model training runs on it". JUPITER
  is where training runs.
* **Every hour leaves a result** tied to a deliverable in the table above.

## Final Report

The portal has the slot (*Final Report Upload* in the consolidated forms).
It reports the table above, filled: measured, not estimated — and the
energy figures. Its deadline and template are to be confirmed with EuroHPC
(the AI Factory rule has been three months after the end).

## Open questions for JSC (first message after the account)

1. Start and end dates, and the "pre-defined consumption pattern".
2. Node sharing and billing: is a 2-GPU job billed 2 or 4 GPU-hours?
3. Maximum walltime and the job-size/priority policy on `booster`.
4. The DPA: template and who signs.
5. Energy accounting per job (Slurm `ConsumedEnergy` or site tooling).

Sources: [JUPITER technical overview (JSC)](https://www.fz-juelich.de/en/ias/jsc/jupiter/tech),
[GH200 node figures](https://www.nextplatform.com/2025/06/11/peeling-the-covers-off-germanys-exascale-jupiter-supercomputer/),
[a JUPITER Booster build/run recipe](https://hipace.readthedocs.io/en/latest/building/platforms/jupiter_jsc.html).
