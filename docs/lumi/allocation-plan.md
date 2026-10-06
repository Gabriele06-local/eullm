# Allocation plan — EHPC-DEV-2026D09-278 (LUMI-G)

> 4,500 node-hours (18,000 GPU-hours) on LUMI-G, `project_465003366`,
> **12-09-2026 → 12-03-2027**. EuroHPC Development Access for the **Engine**,
> not for Forge. What the machine is, how the ROCm build works and what was
> measured on 12-09-2026: [`lumi-g.md`](lumi-g.md). The sibling plan for the
> Leonardo AI-Factory allocation, whose arithmetic this one repeats:
> [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md).
>
> Updated 05-10-2026. Last measured consumption: **0.0%** (12-09-2026) —
> `tools/lumi/status.sh` recounts it from `sacct`.

## The number that governs everything

Six months against 4,500 node-hours is one whole node busy every day. The
first of those months went by with almost nothing spent, so from 05-10-2026:

| | |
|---|---:|
| calendar elapsed | 12.6% (23 of 181 days) |
| straight-line target to date | ~570 node-h |
| spent | ~0 |
| days left | 158 |
| needed from now | **28.5 node-h/day = 1.2 nodes around the clock** |

**One node is no longer enough.** Queue waits come off whatever is running,
so the campaign runs on **two nodes in parallel** until the deficit is
recovered (`status.sh` shows when), then one.

Unused budget is the outcome that has to be explained afterwards; used budget
explains itself if every hour left a measurement behind. That is the design
rule for everything below: the node is never idle, and nothing runs on it
that does not produce a result file with its provenance.

Two facts about LUMI shape how:

- **Single-GCD work cannot spend this allocation.** `small-g` and `dev-g`
  bill 0.125 node-hours per GCD-hour: one GCD busy until March is ~470
  node-hours, a tenth of the budget. The rest has to be whole nodes on
  `standard-g` (48 h walltime).
- **A whole node bills all eight GCDs** whatever runs on it. A job measuring
  one configuration at a time — what every script before the campaign runner
  did — pays for seven idle GCDs.

`small-g` billing: the charge is the largest of GCDs, cores/8 and memory/64
GB. A one-GCD job asking for 16 cores or 128 GB pays for two.

## How the node is kept busy

[`bench/campaign/`](../../bench/campaign/README.md) — tested against a
stand-in engine in CI, run on LUMI by `tools/lumi/sbatch_campaign.slurm`:

- **A queue of points on scratch**, expanded from campaign specs
  (`tools/lumi/campaigns/`). Several jobs drain it at once — two nodes now —
  and a job that hits the walltime puts its unfinished points back.
- **Every GCD always has a point.** Single-GCD points run eight at a time;
  wider points take aligned groups (a pair on one MI250X module, half the
  node, all of it); the next point starts the moment devices free up. A wide
  point that is waiting reserves its devices and narrower ones only take them
  meanwhile if they will be done in time.
- **Workload points stretch to fill.** Sustained-load points run between a
  minimum and a maximum duration and take exactly the time that is free,
  which is what keeps the end of each 48-hour job from being billed idle.
- **Every server is pinned** to the seven cores LUMI documents as closest to
  its GCD, and every result records the cores, the other points on the node
  at the time, and `neighbours-control` measures the same points alone — so
  packing is itself measured rather than assumed harmless.
- **Each job leaves its own usage evidence**: `<job>.summary.json`, the share
  of the job each GCD had work, and `<job>.node.jsonl`, the raw HBM and
  utilisation samples. That is the page of the Final Report that says the
  hours were used.

Workload points are not filler. They send the public, pinned GSM8K, ARC-Easy
and ARC-Challenge sets at a fixed concurrency for hours, grade the first
pass, compare every later pass with it (greedy decoding, prompt cache off: an
answer that changes under load is a bug), and record throughput, latency and
HBM per minute. That is objective (6), stability under sustained load, plus
the accuracy each memory and speed configuration costs — the other half of
objective (1) that a throughput number alone does not give.

## What we committed to

The accepted proposal asks five questions and names the metrics it will be
read against (time-to-first-token, prompt-processing and generation
throughput, HBM use, model-loading time, scaling efficiency, stability under
sustained load; dense and MoE; one device to a full node).

| WP | question | campaign | engine work it waits on |
|---|---|---|---|
| 1 | size × quant × context → memory, throughput | c01 dense/moe/long-context, c02 quant | — |
| 2 | batching as concurrency grows | c01 batch axes; chunked prefill before/after | merge `feat/engine-roadmap` |
| 3 | multi-GPU without disproportionate overhead | c01 ref/scale, c02 235B split vs 2×4 | `--split-mode` incl. tensor |
| 4 | CUDA vs ROCm | the same specs on Leonardo / JUPITER | — (runner is site-neutral) |
| 5 | loading and on-prem transfer | cold/warm load in every result; c02 large MoE | `--moe-cache` on HIP |
| 6 | stability under sustained load | soak groups, 2-24 h | `/metrics` (nice to have) |
| 7 | multi-node ("may be evaluated") | c04, bounded | RPC backend build |

## Campaigns and the budget

`campaign.py plan` prints what each spec costs at most:

| spec | points | node-h | needs |
|---|---:|---:|---|
| `c01-node-baseline` | 205 | ~115 | catalog models only: **runs today** |
| `c02-quant-large-moe` | 75 | ~130 | ~1.2 TB pulled from Hugging Face first |
| ~~`c05-finetune`~~ | — | — | withdrawn from LUMI on 06-10-2026, see below |

Measured honestly, the matrix the proposal describes is cheap: ~250
node-hours a pass, nearly all of it the soaks. What spends 4,500 is doing it
**again for every engine change** — which is the development cycle the
proposal describes ("frequent releases and daily code iterations… experiments
will track exact Git revisions"). A **round** is the same specs planned under
a new label (`ROUND=<engine version> campaign_setup.sh`): every point measured
again, so each engine change has a before and an after on identical
workloads.

| item | node-h |
|---|---:|
| round 1: c01 + c02 on the released 0.7.20 build | ~250 |
| rounds on engine milestones, ~1 every 9 days (~14 × ~250) | ~3,500 |
| c03: row and tensor split, replicas × split, once the engine exposes them | ~300 |
| c04: multi-node exploration, 2-4 nodes, bounded | ~300 |
| iteration on small-g/dev-g, reserve | ~150 |
| **total** | **4,500** |

A round needs a reason: a release, a merged branch that touches inference, a
llama.cpp bump, a build flag (RCCL, HIP graphs) — or, once, a deliberate
same-binary repeat to measure day-to-day and node-to-node spread. The soak
`max_duration_s` is the other lever if rounds run short of the pace.

## Engine work, before the hours (no allocation needed)

The four gaps [`lumi-g.md`](lumi-g.md) found by reading the code are still
open on `main` at 0.7.20, and each one ends a round with a before/after.

0. **Merge `feat/engine-roadmap`.** It is 22 commits ahead of `main` and
   carries measured prompt and answer times in every response (`5febfeb` — on
   `main`, `prompt_eval_duration` is hard-coded to 0, so the runner's
   server-side prefill rate is null until this lands) and chunked prefill
   between decode steps (`04d08ca`, roadmap 0.7-D), an objective (2) result.
1. **The banner must report the live backend, not the compiled one.** A
   `rocm` binary that finds no device still prints `GPU backend: ROCm` and
   then bills node-hours at CPU speed. The runner's `--fit-strict` default
   catches the out-of-memory side of this; nothing catches a missing device.
2. **`--split-mode`, `--tensor-split`, `--main-gpu`, device selection in
   `RuntimeOpts`.** The vendored bindings expose all of it, including
   llama.cpp's **experimental `Tensor` split** — real tensor parallelism, the
   one arrangement that could make a *single* request faster across GCDs. The
   12-09 conclusion that splitting adds no throughput was measured on layer
   split only. This is c03.
3. **The harness (WP0)** — done in `bench/campaign`: one schema
   (`eullm.bench/1`), TTFT, prefill and decode rates, cold/warm load, HBM per
   device, repeats with their spread, and provenance (engine version and
   binary hash, repository revision, ROCm version, cores, neighbours). It
   runs on CUDA unchanged (`--backend cuda --bind none`); what is missing is
   the `sbatch` wrapper for Leonardo and JUPITER, a copy of
   `sbatch_campaign.slurm` with their partitions. A `/metrics` endpoint
   (roadmap 0.7-B) would let a soak see queue depth over time, not just at
   the end.
4. **A placement policy: the smallest split that fits, then replicas.** The
   12-09 numbers give the rule (replicas 4.07× on four GCDs, layer split
   0.98×); c01 and c02 measure it on more models. Turning it into `--fit` on
   a per-device plan, with the engine serving N pinned replicas behind one
   endpoint, is roadmap 1.0-B arriving early, for a measured reason.
5. **NUMA binding** — the runner already pins every server; a round with
   `--bind none` is the before/after that says whether the engine should do
   it itself.
6. **An RCCL build**, `GGML_HIP_RCCL=ON`, and `GGML_HIP_GRAPHS`: experiment
   binaries, each one a round, never the default.
7. **`--moe-cache` on HIP.** It refuses any non-CUDA device and more than one
   GPU, and its pinned-memory patch is CUDA-only. Objective (5) on MoE.

Items 0-2 are October. Items 4-6 December. Item 7 January.

## The CUDA half

Leonardo was not awarded for this project. In order:

1. **Leonardo (again) and JUPITER**, requested by the PI in early October,
   decision pending. The campaign runner and the specs run there as they are;
   only the `sbatch` wrapper changes.
2. **Leonardo AI-Factory, before 02-11-2026.** Measuring how the legal-it
   GGUFs that allocation produced serve on an A100 is evaluation of its own
   output, and its queue needs filling too.
3. **The A100 numbers already held** ([`../cineca/leonardo.md`](../cineca/leonardo.md)),
   measured by hand on 04-09-2026 before two llama.cpp bumps.
4. **The RTX 5070 Ti** — the "smaller on-premises system" of objective (5).

## Models

Permissive licences only, as everywhere in the project, checked against the
Hub on 05-10-2026: Qwen and gpt-oss Apache-2.0, DeepSeek-V3.1 MIT. No Llama.

- **c01, catalog** (Q4_K_M): Qwen3 4B/8B/14B/32B, Mistral-Small-24B,
  Qwen3.6-27B, Qwen3.6-35B-A3B; plus the 12-09 reference model,
  `unsloth/Qwen3.8-27B-GGUF` UD-Q8_K_XL (29.3 GiB), so its rows repeat
  exactly. It was gone from LUMI by 05-10 and is pulled again; the engine now
  names it `qwen3.8-27b-gguf-ud-q8_k_xl`, the 12-09 rows
  `qwen3.8-27b-ud-q8_k_xl` — the same file.
- **c02, from Hugging Face**: Qwen3 8B/14B/32B Q8_0 against c01's Q4_K_M;
  Qwen3-30B-A3B Q4_K_M and Q8_0; gpt-oss-20b and -120b (MXFP4, 11 and 59
  GiB); Qwen3-235B-A22B Q4_K_M (132 GiB: 4 GCDs, so one split or two
  replicas of it) and Q8_0 (233 GiB); Qwen3-Coder-480B-A35B Q4_K_M (270 GiB);
  DeepSeek-V3.1 Q4_K_M (378 GiB — only a whole node runs it at all).

About 1.3 TB on `/scratch` in total, inside the 4 TB asked for. Compute nodes
have no network: `campaign_setup.sh` pulls on a login node and leaves the
missing ones blocked, never failed.

The workload sets are GSM8K, ARC-Easy and ARC-Challenge (4,867 graded
questions). MMLU is not: its pinned source,
`people.eecs.berkeley.edu/~hendrycks/data.tar`, answers 404 as of
05-10-2026 — which also breaks `sbatch_autobench.slurm`'s default `SETS`.

## Calendar

Paced on 28.5 node-hours a day to 12-03-2027.

| month | engine | on the machine | target node-h |
|---|---|---|---:|
| Oct | merge `feat/engine-roadmap`, live banner, split controls, Leonardo/JUPITER wrapper | **c01 now on two nodes**; c02 as its models arrive; round 2 on the merged roadmap branch | 750 |
| Nov | anomalies: KV quant that gains nothing on gfx90a, batching that pays 3.6× here and 1.7× on A100 | c03 split modes; rounds on each release; the CUDA half if granted | 850 |
| Dec | placement policy, served replicas, RCCL and HIP-graphs binaries | before/after rounds; **long soaks queued 20-12 → 06-01**, when nobody is watching and the queue still runs | 880 |
| Jan | `--moe-cache` on HIP, load path | rounds; c04 multi-node | 880 |
| Feb | release candidate | rounds on it | 800 |
| Mar | release | final round to 12-03; data to Zenodo; Final Report draft | 340 |

## The engine's trainer (`c05-finetune`): withdrawn from LUMI

**Not run on this allocation, as of 06-10-2026.** The JUPITER proposal
(EHPC-AIF-2026PG01-1434, submitted to the same Joint Undertaking at the end
of September) describes this allocation in writing: *"This allocation covers
inference-engine work only; no model training runs on it."* A trainer
benchmark is engine work, but it trains weights, and a statement made to
EuroHPC is not reinterpreted after the fact. The `c05` points were taken off
the queue on 06-10; whatever ran on the night of 05-10, before the conflict
was noticed, is reported as such in the Final Report and not used. The spec
stays in the repository for a machine where training is declared.

What it was, for the record. `eullm finetune` is new engine code: llama.cpp's trainer (ggml-opt), which
the engine did not expose, behind one command that trains an F32 GGUF on a
text file and writes a GGUF the engine serves. `c05` measures it on one GCD
the way the other campaigns measure inference: tokens per second, HBM, and
the does-not-fit boundary by model size (Qwen3 0.6B/1.7B/4B Base), optimizer
(AdamW, SGD), training window (512-2048) and which tensors train (all, or
attention only); then whether the held-out loss falls as it should over
three epochs at three learning rates. The text is GSM8K's training split
(public, MIT); the trained models are deleted when the point ends.

It was planned as software engineering and benchmarking of the runtime, on
public data, with no model as an output; the sentence above settles it.

## Lines not to cross

- **This is a development allocation for the engine, inference only.** The
  workload is public benchmark sets, graded; nothing it produces feeds Forge,
  and no training of any kind runs here — not Forge, and not the engine's own
  trainer (`c05`, withdrawn). That is what EHPC-AIF-2026PG01-1434 told
  EuroHPC about this allocation.
- **Every hour leaves a result.** A round without an engine change, a point
  that measures nothing new — those are the hours that are hard to explain.
  More rounds tied to more engine changes are not.

## Final Report

The portal already has the slot (page 11 of the consolidated forms: *Final
Report Upload*). To settle while writing, not after:

- **Licence.** The application says Apache-2.0 in three places; the repository
  is AGPL-3.0-or-later since August 2026. Both open source; the report must
  describe the repository as it is.
- **Where the CUDA numbers came from**, per the section above.
- **The `c05` points that ran on 05-10/06-10** before being withdrawn: how
  many, that they trained small public models on public text as an engine
  benchmark, that the models were deleted, and that the track was stopped
  because of the declaration in EHPC-AIF-2026PG01-1434.
- **The deadline and template**: confirm with EuroHPC; the AI-Factory rule is
  three months after the end (12-06-2027 here), and it is reasonable to
  assume the same.

## Decisions needed

1. **Two nodes in parallel** until the deficit is recovered — the default of
   `submit_campaign.sh`.
2. **The CUDA half on Leonardo AI-Factory** before 02-11, or wait for the
   Leonardo/JUPITER decision.
3. **MMLU**: find a pinned mirror for ReflexBench, or leave it out (the
   campaigns already do).
