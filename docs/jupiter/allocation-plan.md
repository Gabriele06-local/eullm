# Allocation plan — EHPC-AIF-2026PG01-1434 (JUPITER Booster)

> **5,000 GPU-hours on JUPITER Booster (JSC), two months**, AI Factory
> Playground. Awarded 06-10-2026; JSC communicates the start and end dates.
> Proposal: *Porting and scaling a teacher–student verticalisation pipeline
> for European domain-specific LLMs to Grace Hopper*. Technical Assessment:
> accepted, "codes not installed — to be installed by the applicant".
> **Activated 14-10-2026**, after the Principal Investigator's Agreement
> (version 23.06.2025) was signed and sent — see "Lines not to cross" for
> what it binds.
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
2. **No personal data on JUPITER: the rulings stay on Leonardo.**
   The proposal answered "Does your proposal involve handling of personal
   data?: Yes", and the call's own form required the PI to confirm the data
   controller's responsibilities and noted that "they will need to make a
   data processing agreement with the production site"; we answered that we
   would sign one "before any personal data is uploaded, and we will do
   so". The Principal Investigator's Agreement signed on 14-10-2026 is
   JSC's standard form and says the system is used "not to process any
   personal data": that is the default, and the DPA is the way the call
   itself provides out of it. Pseudonymised court decisions are personal
   data, so they do not go to JUPITER without one -- and on 14-10-2026 we
   chose not to ask for one: the rulings are processed on Leonardo, and
   only what `export_clean_cards.py` lets out leaves it. Legislation
   (Normattiva, EUR-Lex: public domain) and public benchmarks are what
   JUPITER trains and evaluates on, which covers every
   throughput measurement (deliverables 1–5, 8): throughput does not depend
   on which
   text is trained on. Case-law work stays on Leonardo, with one way out:
   the ruling cards, which hold the principles and not the case, after
   `forge/scripts/export_clean_cards.py` has run on Leonardo — development
   rulings, special-category and criminal appeals and any card still naming
   a party, place or amount left out, and made-up references (numbers
   50000–59999, which no real ruling has) in place of the real ones, with no
   table back. Exams on real rulings stay off JUPITER too.
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
applied here to `$PROJECT`, which `statfs` calls GPFS (magic 0x47504653):
llama.cpp's read-in
path is synchronous 1 MiB reads through four pinned buffers, from one
thread. `tools/jupiter/sbatch_load.slurm` times six loads of one model on
one GPU from process start to first answer: EuLLM by default and with
`--mmap`, llama-server mapped, read in (`--load-mode none`) and with direct
I/O, and EuLLM by default again.

The server logs of job 2326571 say where the time goes. Every EuLLM load
read the file in ("its file is on GPFS"), and all of it is spent between
"Loading model" and "Allocating context", in llama.cpp's load: the sizing
before it and the context after take milliseconds. Alone on the node,
Qwen3.6-27B (16.8 GB) took 62 s and, after a dozen loads of the same file on
that node, still 57 s (about 280 MB/s: GPFS kept none of it for the next
load), and Qwen3.6-35B-A3B (22.1 GB) 166 s (130 MB/s). With other servers
loading at the same time it took 88-172 s: three servers of the 27B that
started at 19:05:53, 19:06:25 and 19:06:47 UTC all finished at 19:08:15.6,
within 10 ms of one another, the later ones waiting for the first's reads of
the same file (16.8 GB in 142 s, 118 MB/s for the node); a 32B and a 27B
loading together ended 2.6 s apart. llama-server's logs hold no load lines:
llama.cpp's own information lines are its verbosity 4, and llama-server
shows 3 unless given `-lv 4`.

Slurm keeps no energy for these jobs (`ConsumedEnergyRaw` empty for 2326280
and 2326571), so deliverable 9 has to come from our own `nvidia-smi` power
sampling in every job. Both jobs started three minutes after submission;
`AllocTRES` bills a node as `billing=288` (its 288 cores).

**The answer (job 2326997, `sbatch_load.slurm`, Qwen3.6-27B, GPU 0, one
load at a time, seconds from process start to the first answer):**

| case | how llama.cpp reads the file | s |
|---|---|---:|
| EuLLM, default (first on the node) | read in (the rule) | 53.4 |
| EuLLM `--mmap` | mapped | 9.9 |
| llama-server, default | mapped | 3.9 |
| llama-server `--load-mode none` | read in | 47.8 |
| llama-server `--load-mode dio` | direct I/O | 360.2 |
| EuLLM, default, again | read in | 53.1 |

Reading in costs the same in either server, 48-53 s against 4-10 mapped,
and the same the second time (GPFS keeps nothing for it); direct I/O is
seven times worse again. The slow part is llama.cpp's read-in path on GPFS
(synchronous 1 MiB reads from one thread), not EuLLM, and EuLLM chose it by
the network-file-system rule, written for Lustre.

The rule exists because on Lustre several servers *mapping one file at
once* were the slow case. On GPFS they are not (job 2327213,
`EULLM_LOAD_CASES=together`, Qwen3.6-35B-A3B, 22.1 GB, four servers one per
GPU loading at once): mapped first, on a file the node had not read, every
one loaded in 15.7-15.9 s; read in, right after, in 34.0-34.2 s. Alone and
read in, job 2326571 had taken 166 s for the same file. So on GPFS a model
stays mapped, alone or four at once: the change proposed for
`fit::read_in_whole_on_gpu` keeps the network rule for Lustre only (to be
made in one branch, `engine/` being shared with LUMI's).

### The c07 runtimes on GH200, second round (job 2327214, 14-10-2026)

`r-gh200-rt`: Qwen3-8B, Qwen3-14B, Qwen3-32B and Qwen3.6-35B-A3B
(Q4_K_M), EuLLM against llama-server, one GH200 a point: 40 points in
14 min 47 s of one node (the 20 Ollama points blocked: no Ollama here).
Tokens/s:

| model | 1 request | 4 | 16 | 16 clients, 400 graded gsm8k (accuracy) |
|---|---|---|---|---|
| Qwen3-8B | 229 / 226 | 603 / 600 | 1,434 / 1,412 | 1,156 / 1,066 (0.92 / 0.9225) |
| Qwen3-14B | 142 / 142 | 391 / 382 | 1,066 / 1,044 | 783 / 738 (0.95 / 0.96) |
| Qwen3-32B | 70.3 / 70.2 | 201 / 200 | 589 / 588 | 439 / 421 (0.96 / 0.9475) |
| Qwen3.6-35B-A3B | 218 / 216 | 485 / 476 | 888 / 723 | 646 / 576 (0.9325 / 0.94) |

(EuLLM / llama-server.) Dense models are level on throughput; on the
graded workload EuLLM is 4-9% ahead, and on the MoE 35B-A3B 23% ahead with
16 requests and 12% on the graded run. A 16,000-token prompt is read at the
same speed by both (decode 211 tok/s on the 35B-A3B, 124 on the 14B). For
deliverable 5: one GH200 writes 888 tok/s with a 35B-A3B MoE under EuLLM,
where Leonardo's synthetic-data generation ran at 58-64 tok/s with the
30B-A3B in plain Transformers. Not yet the same model or precision (Q4_K_M
GGUF against BF16): the like-for-like number is the teacher probe's
generation in Transformers on GH200, then the same teacher as GGUF under
EuLLM. The "load s" column again compares EuLLM's load with llama-server's
warm-up request only.

### The first Forge numbers (job 2334585, 15-10-2026)

`sbatch_teacher_probe.slurm` on one node, the Forge venv on the site's
PyTorch (`teacher_probe.py`: random tokens, no corpus here yet, and the
teacher without Leonardo's continued-pre-training adapter):

* **Distillation step, two runs side by side** (deliverables 2 and 3):
  the teacher (Qwen3-30B-A3B-Base, BF16, 56.9 GiB, 57.8 at peak) alone on
  one GPU, the LoRA r128 student (Qwen3-4B-Base, 33.4 GiB at peak) with
  checkpointing on the next, `cutoff_len` 2048. One micro-batch: median
  0.411 s on GPUs 0-1 and 0.407 s on GPUs 2-3, at the same time: **548 and
  553 optimiser steps an hour** at Leonardo's 16 micro-batches a step,
  against ~380 there with the teacher split over two A100 and the student
  on a third. Per GPU-hour: ~276 steps against ~127, 2.2 times. To be
  confirmed by `distill.py` itself on real text, with the adapter.
* **The teacher generating in plain Transformers**, BF16, one GPU, 256
  tokens a sequence: 13.9 tok/s alone, 113.5 with 8 sequences, 346.9 with
  32 (deliverable 5's baseline; Leonardo: 58-64 tok/s).
* **Loading the teacher**: 201 s for 61 GB, two processes reading it at
  once from GPFS, 12.8 s for the student: about 3.5 minutes of every job
  link, to be counted in the link length.
* **Energy**: 38.4 Wh for the 3.9 minutes of phase 1, mostly the load
  (mean 586 W for the four GPUs), and 15.2 Wh for the 1.4 minutes of
  generation (628 W). The method works; the per-step figure comes with
  the real run.

Spent on JUPITER by 15-10-2026 noon: 56 minutes of node time in nine
jobs, about **3.8 GPU-hours** of 5,000.

## What the GPU probe found (job 2326998, 14-10-2026)

`probe_env.sh` on a booster node (`jpbo-016-09`), as a batch job:

* 288 cores (four Neoverse-V2 of 72), four GH200 of 97,871 MiB (95.0 GiB to
  torch), 680 W each, all four joined by NVLink (NV6 between every pair).
  GPU *i* sits by cores 72*i*-72*i*+71 and NUMA node *i* (its HBM is NUMA
  node 4, 12, 20, 28): the two runs of a node pin to GPUs 0-1 with cores
  0-143 and GPUs 2-3 with cores 144-287.
* `booster`: whole nodes (`OverSubscribe=EXCLUSIVE`), default time 1 h, no
  maximum, 5,884 nodes (5,257 allocated and 27 idle when it ran).
* `module load PyTorch` alone works (Stages/2026, a Core package): Python
  3.13.5, torch 2.9.1+cu128 built for sm_80/90/100/120,
  `torch.cuda.is_available()` true with four devices, BF16 supported.
  transformers, peft, accelerate, datasets and safetensors are not in it:
  `forge/scripts/jupiter/setup_venv.sh` puts them in a venv on top of the
  module (`--system-site-packages`, torch and numpy held at the module's
  versions by a constraint), and `forge/scripts/jupiter/env.sh` loads both.
* The teacher (30B-A3B, ~61 GiB in BF16) fits one GPU: two runs per node.
* Compute nodes: no huggingface.co. Apptainer 1.5.4 is installed.
* No `ConsumedEnergy` for any job in seven days.

### Where it stands (14-10-2026) and what is next

On `$PROJECT/$USER`: the checkout, the engine `next10`
(`eullm-target/next10/release/eullm`, commit 1fdf9c6), llama-server b11482
(`campaign/runtimes/`), Qwen3.6-27B, Qwen3-32B and Qwen3.6-35B-A3B
(`eullm-models/`), the queue with round `r-gh200` done (`campaign/`). Spent:
about 1.3 GPU-hours (jobs 2326280 and 2326571). Next, in order:

1. EuLLM's load time: four servers loading at once on GPFS, mapped against
   read in (`EULLM_LOAD_CASES=together`), then the change to
   `fit::read_in_whole_on_gpu` in `engine/` (shared with LUMI's branch).
2. Forge: the venv is made (`setup_venv.sh`, 14-10-2026: torch 2.9.1 from
   the module, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0,
   datasets 5.1.0 in the venv; `forge/venv-known-good-2026-10-14.txt`) and
   Qwen3-30B-A3B-Base and Qwen3-4B-Base are in `forge/hf_home`.
   `sbatch_teacher_probe.slurm` is the first GPU job: two distillation
   micro-batch loops side by side (the teacher in BF16 alone on a GPU,
   Leonardo's LoRA student on the next), the teacher generating with plain
   Transformers, and the node's power every second. Then week 1 of the
   calendar below, with the training libraries checked against Leonardo's
   for the baseline.
3. The data JUPITER trains on: legislation, public benchmarks and the
   ruling cards `export_clean_cards.py` lets out on Leonardo (no DPA, by
   choice: no personal data here).

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

## Job shape

* **Unit of work: one run = 2 GPUs** (teacher, student). Nodes are
  exclusive (probe, 14-10-2026), so a job is one node running **two runs
  side by side**: GPUs 0-1 with cores 0-143, GPUs 2-3 with cores 144-287
  (`CUDA_VISIBLE_DEVICES`, `taskset`/`numactl`).
* **Duration:** the shortest link the queue places quickly; Forge resumes
  from checkpoint, including the position in the data, so short links cost
  minutes.
* **Two nodes at most** (8 GPUs): the second node is for the Qwen3.5 track
  or a second vertical, never for a multi-node job.
* The Leonardo rules in `forge/CLAUDE.md` ("the shape of a job that starts")
  are the template; JUPITER gets its own version and its own
  `test_*_job_shape.py` once the probe has spoken.

## Lines not to cross

* **No personal data before the DPA** (signed PI agreement, 14-10-2026;
  the proposal's Privacy & Data Governance answers): no court decisions,
  pseudonymised or not, and nothing derived from them that could carry a
  name — OPD prompt files with case-law rows included — only legislation,
  public benchmarks and the cards `export_clean_cards.py` lets out. After
  the DPA, still only pseudonymised text, as declared. Nothing from the
  corpus is ever in a repository.
* **What the proposal promised about the data, which binds us after the
  DPA too**: names of parties and lawyers, tax codes, VAT numbers, IBANs,
  addresses, phone numbers, emails and dates of birth replaced; links to
  the original documents removed; surnames in case citations removed;
  generated examples that still hold a placeholder or anything the
  anonymiser would remove discarded; generated questions about a single
  case filtered out; the corpus kept only on project storage, moved only
  over SSH, never in a repository or a third-party service; raw data not
  needed for training deleted at the end of the project; models evaluated
  on 2025–2026 decisions never used for training.
* **Export control and the AI Act** (same agreement): no work on the
  excluded topics, and the AI Act's obligations for general-purpose models
  apply to what we release; JSC is told of any relevant change of research
  topic.
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
4. (Not asked, by choice on 14-10-2026: no DPA, so no personal data on JUPITER.)
5. Energy accounting per job (Slurm `ConsumedEnergy` or site tooling).

Sources: [JUPITER technical overview (JSC)](https://www.fz-juelich.de/en/ias/jsc/jupiter/tech),
[GH200 node figures](https://www.nextplatform.com/2025/06/11/peeling-the-covers-off-germanys-exascale-jupiter-supercomputer/),
[a JUPITER Booster build/run recipe](https://hipace.readthedocs.io/en/latest/building/platforms/jupiter_jsc.html).
