# Allocation plan — EHPC-DEV-2026D09-278 (LUMI-G)

> 4,500 node-hours (18,000 GPU-hours) on LUMI-G, `project_465003366`,
> six months. EuroHPC Development Access for the **Engine**, not for Forge.
> What the machine is, how the ROCm build works and what was measured on
> 12-09-2026: [`lumi-g.md`](lumi-g.md). The sibling plan for the Leonardo
> AI-Factory allocation, whose arithmetic this one repeats:
> [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md).
>
> Written 05-10-2026. Last measured consumption: **0.0%** (12-09-2026).

## The number that governs everything

Six months of calendar against 4,500 node-hours is **one full node busy
continuously, every day, until the end**. From 05-10-2026:

| if the window ends | days left | node-hours/day | full nodes busy 24/7 |
|---|---:|---:|---:|
| 28-02-2027 (started with early-September access) | 146 | 30.8 | 1.3 |
| 31-03-2027 (started 01-10, as requested) | 177 | 25.4 | 1.1 |

The end date is still the one unknown, and it moves the pace by a fifth. The
Puhuri project page shows it; `lumi-allocations` does not.

Two consequences follow, and both are already known from Leonardo.

**Single-GCD work cannot spend this allocation.** `small-g` and `dev-g` bill
0.5 GPU-hours per GCD-hour, which is 0.125 node-hours. One GCD kept busy
every hour until 31-03-2027 is ~530 node-hours — 12% of the budget. That is
where iteration belongs, and it is cheap precisely because it does not
count. **The other 88% has to be full-node work**, on `standard-g` (whole
nodes, 48 h walltime) or on `small-g` asking for all 8 GCDs (same price,
3-day walltime, up to 4 nodes).

**The queue must never be empty.** Every hour without a full-node job queued
is budget that expires, and an allocation returned mostly unused is visible
in the Final Report. A full-node job should be queued or running at all
times from now on, the way Leonardo's recovery made it a rule
([`../cineca/allocation-recovery.md`](../cineca/allocation-recovery.md)).

Note on `small-g` billing: the charge is the largest of GCDs, cores/8 and
memory/64 GB. A one-GCD job asking for 16 cores or 128 GB pays for two.

## What we committed to

The accepted proposal asks five questions and names the metrics it will be
read against (time-to-first-token, prompt-processing and generation
throughput, HBM use, model-loading time, scaling efficiency, stability under
sustained load; dense and MoE; one device to a full node). Each question
becomes a work package. Engine work comes first and costs no allocation.

| WP | question | what has to exist | what spends hours |
|---|---|---|---|
| 0 | — | one benchmark harness, one result schema, both sites | nothing (dev-g only) |
| 1 | model size × quant × context → memory, throughput | WP0 | the measurement matrix |
| 2 | continuous batching as concurrency grows | WP0 | concurrency sweeps, profiling |
| 3 | multi-GPU without disproportionate overhead | split controls, placement policy | arrangements × before/after |
| 4 | CUDA vs ROCm | a CUDA source (see below) | identical runs, both sites |
| 5 | loading and on-prem transfer | load-time metrics | cold/warm loads, big MoE |
| 6 | stability under sustained load | load generator | 48 h full-node soaks |
| 7 | multi-node (proposal: "may be evaluated") | RPC backend build | bounded exploration |

## Engine work, before the hours (no allocation needed)

The four gaps [`lumi-g.md`](lumi-g.md) found by reading the code are still
open on `main` at 0.7.20, and the first three gate everything else.

0. **Merge `feat/engine-roadmap` first.** It is 22 commits ahead of `main`
   and carries two things the harness needs: measured prompt and answer
   times in every response (`5febfeb` — on `main`, `prompt_eval_duration` is
   hard-coded to 0 and `eval_duration` includes the prefill), and chunked
   prefill between decode steps (`04d08ca`, roadmap 0.7-D), which is itself
   a before/after experiment for objective (2).
1. **The banner must report the live backend, not the compiled one.** A
   `rocm` binary that finds no device still prints `GPU backend: ROCm` and
   then bills node-hours at CPU speed (`banner.rs`, `inference/mod.rs`).
   Count the devices `ggml_backend_dev_count()` actually returns and refuse
   to serve on a cluster when a GPU build finds none. Cheapest insurance on
   the list.
2. **`--split-mode`, `--tensor-split`, `--main-gpu`, device selection in
   `RuntimeOpts`.** The vendored bindings expose all of it
   (`with_split_mode`, `with_main_gpu`, `with_devices`), including
   llama.cpp's **experimental `Tensor` split** — real tensor parallelism,
   and the one arrangement that could make a *single* request faster across
   GCDs. The 12-09 conclusion that splitting cannot add throughput was
   measured on layer split only; `Row` and `Tensor` have never run on LUMI.
   Objective (3) cannot be studied through a runtime that only lets
   llama.cpp layer-split silently.
3. **The harness (WP0).** Every LUMI script already prints one
   `BENCH_RESULT {json}` line; none records the engine's git revision or
   version, TTFT, prefill tok/s, peak HBM or load time. What is needed is one
   versioned schema all of them emit, with: model-load time (cold and warm),
   per-device HBM sampled during the timed window, the 1→2→4→8 device sweep,
   and the provenance that makes a number reproducible — engine revision,
   llama.cpp build, ROCm/CUDA version, compile flags, Slurm job id. The same
   script must run unchanged on CUDA: **no engine sbatch exists for Leonardo
   yet** (the A100 numbers of 04-09 were taken by hand), so
   `tools/leonardo/sbatch_bench.slurm` is part of this item. A `/metrics`
   endpoint (roadmap 0.7-B) belongs here too: a 48 h soak needs the engine to
   report its own queue depth and latencies over time, not just at the end.
4. **A placement policy: the smallest split that fits, then replicas.** The
   12-09 measurement gives the rule — replicas scaled 4.07× on four GCDs,
   layer split 0.98× — and today it lives in a Slurm script. Extending
   `--fit` from one summed VRAM figure with one flat reserve to a per-device
   plan ("fewest devices that fit, as many replicas as remain", llama.cpp's
   multi-device `common_fit_params` is already wrapped in the bindings), with
   the engine serving N device-pinned replicas behind one endpoint, is the
   engine change objective (3) should end with. It is roadmap 1.0-B (worker
   pool with explicit GPU assignment) arriving early, for a measured reason.
5. **NUMA binding per GCD.** LUMI's GCDs and CPU NUMA domains are not
   numbered alike; host threads for a replica should sit on the domain
   closest to its GCD. Measure first (WP3), make it default only if it pays.
6. **An RCCL build as an experiment binary**, `GGML_HIP_RCCL=ON`, never the
   default: the product reasoning in `build.rs` stands, the project measures
   whether it should. Same for `GGML_HIP_GRAPHS`.
7. **`--moe-cache` on HIP.** It refuses any non-CUDA device and more than one
   GPU (`fit.rs`), and its pinned-host-memory patch is CUDA-only. Porting it
   is objective (5) on MoE: the same expert cache that lets a 5070 Ti run a
   model larger than its VRAM, measured on hardware where the answer can be
   checked against the model fully resident.

Items 0-3 are October. Items 4-6 are December, after WP1-3 have said where
the time goes. Item 7 is January.

## The CUDA half

Leonardo was rejected for this project, so objective (4) needs NVIDIA
numbers from elsewhere. In order of preference:

1. **Leonardo AI-Factory, before 02-11-2026.** That allocation exists to
   produce legal-it-4b/8b; measuring how those GGUFs serve on an A100 with the
   engine they ship in is evaluation of its own output, and that allocation
   needs its queue filled anyway. Run the WP0 harness there on the legal-it
   GGUFs plus the reference models, then the identical files on LUMI. **This
   is why the harness is the first deliverable: it has four weeks.**
2. **The numbers already held**, A100 measured by hand on 04-09-2026
   ([`../cineca/leonardo.md`](../cineca/leonardo.md)) — usable, but they
   predate two llama.cpp bumps and the common schema.
3. **A EuroHPC Benchmark Access request on an NVIDIA system** (MareNostrum 5
   ACC, JUPITER) in November, with the LUMI data as the justification — which
   is exactly what the proposal said these results were for.
4. **The RTX 5070 Ti workstation** — not a peer of the MI250X, but it is the
   "smaller on-premises system" of objective (5), and every WP5 result
   should end with a run there.

## Models

Permissive licences only, as everywhere in the project: Qwen (Apache-2.0),
DeepSeek (MIT), GPT-OSS (Apache-2.0), Mistral's Apache-2.0 releases. No
Llama. Exact tags come from the catalog at run time; the shape of the set is:

- **dense, one GCD**: 4B, 8B, 14B, 32B (Q4_K_M, Q8_0, BF16 where it fits)
- **dense, two GCDs**: 32B BF16 (~64 GB) — the smallest real split case
- **MoE, one GCD**: Qwen3-30B-A3B, GPT-OSS-20B
- **MoE, several GCDs**: GPT-OSS-120B (2), Qwen3-235B-A22B (Q4 ~3, Q8 ~5),
  Qwen3-Coder-480B-A35B (Q4 ~5-6)
- **MoE, the whole node**: DeepSeek-V3.x 671B Q4 (~400 GB of 512 GB HBM).
  The model that only a full node runs at all, and the clearest single
  demonstration of objective (3) on MoE.
- **our own**: legal-it-4b and -8b — the on-prem target of objective (5), and
  the files the CUDA half is measured on.

Large GGUFs go to `/scratch/project_465003366` from a login node (compute
nodes have no network). Roughly 2-3 TB in total, inside the 4 TB asked for;
`/flash` only temporarily for the load-time comparison, since it bills 3×.

## Budget

| item | node-hours | where |
|---|---:|---|
| iteration: builds, smoke, single-GCD profiling | 250 | dev-g, small-g |
| WP1+2 matrix, 1-8 GCDs, 3 repeats per point | 900 | standard-g |
| WP3 arrangements: layer/row/tensor split, replicas, RCCL, NUMA, before/after | 700 | standard-g |
| large MoE (235B / 480B / 671B), load + throughput + concurrency | 500 | standard-g |
| WP5 loading: cold/warm, Lustre vs flash, mmap vs read, MoE cache | 300 | standard-g |
| WP6 soaks: 48 h full node, monthly, plus the holiday window | 700 | standard-g |
| nightly regression on every engine/llama.cpp change, ~2 h/night | 300 | standard-g |
| WP7 multi-node exploration, 2-4 nodes, bounded | 300 | standard-g |
| final campaign with the release engine | 400 | standard-g |
| reserve | 150 | |
| **total** | **4,500** | |

Repeats are not padding: a point measured once cannot carry a confidence
interval, and 12-09 showed a 0.2% spread is achievable, which is what makes a
2% effect reportable. Packing eight independent single-GCD points on one node,
one per GCD, is the efficient way to run the matrix — after one run that
checks neighbours do not perturb each other.

## Calendar

Paced for a 31-03-2027 end; if Puhuri says 28-02, March's work moves into
February and every month's target rises by a fifth.

| month | engine | on the machine | target node-h |
|---|---|---|---:|
| Oct | merge `feat/engine-roadmap`, live banner, split controls, harness v1 + Leonardo twin, LUMI `status.sh` (pace and idle-queue alarm) | now, with the existing scripts: repeat the replica rows, 1/2/4/8 sweep with per-device utilisation; then the first row/tensor split runs; stage the big MoE; **CUDA half on Leonardo before 02-11** | 500 |
| Nov | anomalies: KV quant that gains nothing on gfx90a, batching that pays 3.6× here and 1.7× on A100 | WP1+2 matrix; chunked prefill before/after; first 48 h soak; nightly regression starts; Benchmark Access request | 850 |
| Dec | placement policy, served replicas, NUMA binding, RCCL binary | WP3 before/after; **long soaks queued 20-12 → 06-01**, when nobody is watching and the queue still runs | 850 |
| Jan | load path: staging, parallel per-device load; `--moe-cache` on HIP | WP5; large-MoE campaign; multi-node exploration | 850 |
| Feb | release candidate | re-run the matrix on it; soak | 800 |
| Mar | release | final campaign; data to Zenodo; Final Report draft | 650 |

## Lines not to cross

- **This is a development allocation for the engine.** Soaks drive the
  engine with public benchmark prompts and throw the answers away. Using the
  hours to generate Forge training data, or to train anything, would turn it
  into production for a different project — the kind of thing a Final Report
  cannot explain.
- **No work invented to burn hours.** Everything in the budget table answers
  one of the five questions or makes an answer reproducible. If a line stops
  doing that, it comes out and the hours go to repeats or soaks.

## Final Report

The portal already has the slot (page 11 of the consolidated forms: *Final
Report Upload*). Three things to settle while writing, not after:

- **Licence.** The application says Apache-2.0 in three places; the repository
  is AGPL-3.0-or-later since August 2026. Both are open source and neither
  affects the award, but the report must describe the repository as it is.
- **Where the CUDA numbers came from**, per the section above, stated plainly.
- **The deadline and template**: confirm with EuroHPC; the AI-Factory rule is
  three months after the end, and it is reasonable to assume the same.

## Decisions needed

1. **The window end date** (Puhuri). Everything above is paced on a guess.
2. **The CUDA half on Leonardo** before 02-11: agree that measuring the
   legal-it GGUFs with the engine is in scope there.
3. **Benchmark Access on an NVIDIA system** in November: worth the
   application, or is the Leonardo data enough?
