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

## What the login probe found (14-10-2026)

`forge/scripts/jupiter/probe_env.sh` on a login node (`jpbl-s02-04`):
aarch64 Neoverse-V2, 72 cores, and a GH200 of its own (97,871 MiB, 900 W).
Project `e-aif-2026pg01-1434`, which is also the budget account; `$PROJECT`
is `/e/project1/e-aif-2026pg01-1434` (6.2 PB free), `$SCRATCH`
`/e/scratch/e-aif-2026pg01-1434`. Partitions `booster` (default) and
`largebooster`; `booster` hands out whole nodes (`OverSubscribe=EXCLUSIVE`),
so **a job pays for all four GPUs of its node** and two runs per node is the
shape (deliverable 3), default time 1 h, no maximum. Modules: GCC/14.3.0,
CUDA/13, NVHPC/25.9, Python/3.13.5, PyTorch/2.9.1 (built with GPU support);
the system Python is 3.9 with none of the training packages, so the Forge
venv goes on top of the PyTorch module. Apptainer is there but needs the
`container` group. huggingface.co answers from the login node. The GPU probe
(one node, 15 minutes) is still to run; the engine's comparison job prints
the compute node's GPUs and whether it reaches the internet.

## The engine on GH200 (deliverable 5)

`tools/jupiter/` builds and measures the engine there as `tools/lumi/` does on
LUMI, and in the same terms: `build_engine.sh` (Rust from rustup in
`$PROJECT/$USER`, CUDA for sm_90, a check that the binary carries Hopper
code), `install_runtimes.sh` (llama-server at the same llama.cpp commit),
`setup.sh` (models, workload sets, c07's `rt27-*` points planned for the
build) and `submit.sh` (one node: the points four at a time, then the
five-turn conversations of `multiturn_probe.py`). The same points ran on the
MI250X on 14-10-2026 (`r-27b-next10`), so the two machines compare row by
row; a job asks for an hour of one node and took 17 minutes (1.2 GPU-hours).

The first build (14-10-2026) stopped at the link with "could not find native
static library `cudart_static`": the crate that finds CUDA for the build
reads no `CUDA_PATH` on Linux and knows only `/usr/local/cuda`, while the
CUDA/13 module's toolkit is under its own prefix. `build.rs` now looks in the
toolkit `CUDA_PATH` names first (backlog H4-H, the same gap Leonardo's Spack CUDA
hit), and `build_engine.sh` checks before building that the toolkit has the
static library.
The second stopped one step later, at "relocation R_AARCH64_ADR_PREL_PG_HI21
... recompile with -fPIC" in `ggml-cuda.cu.o`: rustc links the binary
position-independent, the host code nvcc compiles is not, and the site's GCC,
unlike Ubuntu's, does not make it so by default. `build_engine.sh` links the
`eullm` binary alone with `-no-pie` (`cargo rustc ... -- -C link-arg=-no-pie`),
as the release job for x64 CUDA and Leonardo do.

The first job (2326280, 14-10-2026, 2.5 minutes of one node) blocked all 22
points: none of the three models had been pulled. `setup.sh` and the runner
took a model for installed when its id was any word `eullm list` printed,
and with an empty store the command prints the EU catalog, where all three
are; both now read only the table of installed models. The same job showed
that **compute nodes do not reach the internet** (huggingface.co: no answer)
and have no `git`: models and workload sets are fetched on a login node, and
`submit.sh` hands the job the checkout's commit. Its four GPUs: GH200 120GB,
97,871 MiB each, 680 W power limit, driver 610.57.04.

### First measurements (job 2326571, 14-10-2026, `next10`)

One GH200 a point, four points at a time, the c07 `rt27-*` points measured
on LUMI's MI250X the same day (`r-27b-next10`, one GCD); EuLLM and
llama-server at the same llama.cpp commit (b11482), Q4_K_M, tokens/s:

| requests | 27B EuLLM | 27B llama-server | 32B EuLLM | 32B llama-server | 27B EuLLM on a GCD | 27B llama-server on a GCD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.1 | 74.4 | 69.9 | 69.5 | 38.3 | 35.3 |
| 4 | 204 | 188 | 199 | 200 | 67.8 | 61.5 |
| 16 | 513 | 394 | 564 | 585 | 186 | 132 |
| 32 | 700 | 392 | 875 | 778 | 240 | 111 |
| 16 clients, 400 graded gsm8k | 432 | 343 | | | 165 | 126 |

The engine's lead on the hybrid Qwen3.6-27B is not the MI250X's: on the
GH200 it writes 30% more than llama-server with 16 requests, 79% more with
32 (where llama-server stops scaling, 394 to 392, as on the GCD) and 26%
more on the graded workload, at the same accuracy (0.9425 and 0.95, three
questions in 400). The first token with 32 requests: 187 ms against 5,970.
On the dense Qwen3-32B the two are level up to 16 and EuLLM 12% ahead at
32. A 16,000-token prompt is read at the same speed (27B: 2,430 tokens/s
against 2,398; 32B: 2,002 against 1,915). A GH200 is 2-3 times a GCD here
(half an MI250X).

Five turns over a 6,000-token document (`multiturn_probe.py`, one GPU):
from the second turn Qwen3.6-27B reads each prompt in 53-69 ms, llama-server
in 79-84; Qwen3.6-35B-A3B in 33-49 ms against 40-52. The first turn, the
whole prompt, about 2.2 s for both on the 27B.

**Loading**: the report's "load s" gave EuLLM 107-184 s and llama-server
2-12 s, and the first turn of the 35B-A3B's first conversation waited 169 s
for EuLLM. The two columns do not measure the same thing: the runner times
the warm-up request once the server answers its readiness check, and
`eullm serve` loads the model in that request, while llama-server loads it
before `/health` answers, so llama-server's 2-12 s holds no load at all (the
same holds for every runtime comparison of load times from the runner). That
EuLLM is slow is still likely: 107 s for a 17 GB file is about 160 MB/s,
where LUMI's Lustre gave 0.7-1.3 GB/s. The suspect is the engine's rule for
network file systems (`fit::read_in_whole_on_gpu`: a model whole on the GPUs
is read into memory instead of mapped), measured right on LUMI's Lustre and
applied here to `$PROJECT` if `statfs` calls it a network file system
(JSC's storage is IBM Storage Scale, GPFS; to be confirmed): llama.cpp's read-in
path is synchronous 1 MiB reads through four pinned buffers, from one
thread. `tools/jupiter/sbatch_load.slurm` times six loads of one model on
one GPU from process start to first answer: EuLLM by default and with
`--mmap`, llama-server mapped, read in (`--load-mode none`) and with direct
I/O, and EuLLM by default again.

### Where it stands (14-10-2026) and what is next

On `$PROJECT/$USER`: the checkout, the engine `next10`
(`eullm-target/next10/release/eullm`, commit 1fdf9c6), llama-server b11482
(`campaign/runtimes/`), Qwen3.6-27B, Qwen3-32B and Qwen3.6-35B-A3B
(`eullm-models/`), the queue with round `r-gh200` done (`campaign/`). Spent:
about 1.3 GPU-hours (jobs 2326280 and 2326571). Next, in order:

1. EuLLM's load time on JUPITER (above): first, free, the existing server
   logs of job 2326571 (whether the rule fired, and the time between
   "Loading model" and "Allocating context"); then
   `tools/jupiter/sbatch_load.slurm`, one node for 10-20 minutes.
2. The GPU probe for Forge (`forge/scripts/jupiter/probe_env.sh` as a
   15-minute batch job; it now loads the site's PyTorch module, without
   which the system Python 3.9 has no torch), then the Forge venv on top of
   the PyTorch module and week 1 of the calendar below.
3. The DPA with JSC, before any personal data.

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
