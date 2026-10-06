# Experts in RAM: the half of Strata llama.cpp lacks — implementation plan

**Status:** phases 1 and 2 measured on the reference PC; phase 2's pinning goes through `--no-mmap`, since the driver refused to pin the mapped file. Phase 6 written (patch `0003`, `LLAMA_MOE_PREFETCH=1`) and measured: with four slots a 33,200-token prompt reads 24-42% faster, to the same answer, and six or eight slots no faster than four · 6 October 2026. Written against `feat/moe-cache` at d1e0904, where llama.cpp is 6b7b03a: b11370 plus PR #29887, the expert cache. Line numbers refer to that tree. Strata's design and figures come from its paper (Strata v0.1.35); the speeds come from the reference PC: RTX 5070 Ti 16 GB on PCIe 4.0 x16, Ryzen 9 5950X (16 cores, AVX2), 64 GB of DDR4.

**How our changes are carried.** As patch files in `engine/vendor/llama-cpp-rs/llama-cpp-sys-2/patches/`, which the build script applies to a copy of the submodule (`llama_patches.rs`): the submodule stays at 6b7b03a, and nothing has to be pushed to the mirror for a change to build. `0001` gives CUDA a way to pin host memory on request; `0002` is phases 1 and 2 in the cache; `0003` is phase 6, in ggml's scheduler.

---

## 0. Where we are

Qwen3.8-Flash-Next IQ2_XS on the reference PC, measured with `bench/speed_check.py`:

| Engine | Writes (tokens/s) | Reads a prompt (tokens/s) |
|---|---:|---:|
| Strata, MTP on | 85.8 | 2,320 |
| EuLLM 0.7.20, the usual split | 21.5 | 256 |
| llama.cpp + PR #29887, 8,000 MiB cache | 49.4 | 206 |
| the same, with llama.cpp's MTP (2 drafts) | 36.3 | 196 |
| EuLLM, `--moe-cache auto` | 43.6 | 210 |
| EuLLM, `--moe-cache auto --no-mmap` (experts pinned) | 58.1 | 451.5 |

With `--moe-cache auto` the GPU was busy 59% of the time and the CPU idle.

**Plain decoding is level.** Strata's paper gives 47-57 tokens/s without its MTP layer (finding 2, on an RTX 5070 with DDR5). The cache brings llama.cpp to 49.4.

**MTP is where the gap is, and why it does not pay for us.** Strata gets 1.6-1.8× from its MTP layer; llama.cpp lost 15-20% with it. A check of three tokens routes them to up to 30 experts per layer. In llama.cpp every one of those not in VRAM is copied over PCIe before the GPU can start, so a check costs about 2.5 single steps and yields 2.05 tokens (52% of the drafts kept). In Strata the extra experts go to the CPU, which computes them while the GPU works, so a check costs little more than a step.

**What Strata does that llama.cpp does not** (paper §3.1-3.4, findings 2, 4 and 9):
1. Every expert lives in one *pinned* arena: the CPU computes any of them in place, and the GPU can pull any by DMA at the full PCIe rate.
2. Per layer, the experts are split three ways. Those in the VRAM cache run on the GPU. A share of the misses is copied in by the GPU's copy engine while everything else runs (55% of them for the i-quants). The rest are computed by the CPU's cores. The GPU adds the results up.
3. The per-layer handshake is a "doorbell" in pinned memory that the CPU spins on, and the whole 48-layer pass is one captured CUDA graph: no driver synchronisation per layer.
4. The cache starts from a profile recorded on other prompts (50% of the experts served from VRAM) and follows the conversation (72%).

## 1. What the cache does per MoE layer today

- The scheduler gives a `MUL_MAT_ID` whose experts are in host memory to the GPU, reading the experts from a view of the cache's bank instead (`ggml/src/ggml-backend.cpp`, in the backend assignment and the split builder).
- Before that split runs, it reads the router's choice back to the host and waits for the GPU to drain to do so (`ggml_backend_synchronize`, ggml-backend.cpp:1974-1976). The GPU is idle until the next step is queued.
- `prepare` (`src/llama-moe-cache.cpp`) runs the LRU and copies each miss into its slot with `ggml_backend_tensor_set_async`, from the expert tensor as llama.cpp loaded it: an mmap of the GGUF, pageable memory, which the driver stages before the DMA.
- The GPU then computes every expert of the layer from the bank.

## 2. Phases

### Phase 1 — Measure a decode step (1-2 days)

Timers in the cache's callbacks, reported per step and per layer: time waiting on the read-back, time planning, bytes and time uploading, misses. Report the hit rate while the server runs, not only at exit; today it is printed when the context is freed, at a log level EuLLM hides. Measure 1-token steps and 3-token MTP checks.

**Deliverable:** a table of where a step goes, on the reference PC. It decides how much phases 2 and 3 can win, and in which order.

**Written** (patch `0002`): `LLAMA_MOE_CACHE_STATS=N` prints to stderr, every N steps of up to 8 tokens, the time per step split into the time until each layer's router is read back, the time in the cache's `prepare` and the rest; MiB copied and the share of experts found in VRAM; and the copy time and rate, from one step in 8 that waits for its copies. Per-layer figures are the per-step ones divided by the layers the line names. What it cannot see: the scheduler's own wait at each read-back is inside "to the routers", not apart from the GPU's computing.

**Measured** (4 October, reference PC, `--moe-cache auto` = 8 GiB, Qwen3.8-Flash-Next IQ2_XS). The experts stayed in pageable memory: the driver refused to pin the mapped file (below). Windows of 64 one-token steps:

| Window | ms per step | Up to the routers (48 layers) | In the cache | After | MiB copied | Experts in VRAM | Copying |
|---|---:|---:|---:|---:|---:|---:|---:|
| first, cache cold | 25.1 | 14.5 | 9.1 | 1.5 | 83.6 | 87.3% | 10.0 ms at 9.1 GB/s |
| the six after | 20.1–23.4 | 13.7–14.0 | 5.0–8.0 | 1.4–1.5 | 46–76 | 88.5–93.0% | 4.7–9.7 ms at 9.0–9.2 GB/s |

- The cache works: nine experts in ten are found in VRAM once it is warm.
- Copying the rest is a quarter to a third of a step, at 9 GB/s, because the driver stages it out of pageable memory. From pinned memory PCIe 4.0 x16 carries up to about 25 GB/s: phase 2.
- Two thirds of a step pass before each layer's routing is read back: the GPU computing the 48 layers, plus the waits at each read-back. How much of that is waiting decides phase 4, and is the next thing to measure.
- A step of 20–21 ms is 48–50 tokens/s inside llama.cpp, the same as llama-server with the same cache (49.4). `speed_check` sees 44 through EuLLM, so 1–2 ms per token go to EuLLM's own work around the decode: to look at separately.

### Phase 2 — Pin the experts (small; a candidate for a first upstream PR)

Register the expert ranges of the mmap with CUDA. `ggml_backend_cuda_register_host_buffer` exists (`ggml/src/ggml-cuda/ggml-cuda.cu`:5021, behind `GGML_CUDA_REGISTER_HOST`), and llama.cpp never calls it.
- Pin only the experts: about 35 GB for IQ2_XS out of 64 GB. Leave the 27 GB of n-gram tables (PLE) pageable: a token reads 16 rows of them.
- Expected: misses and prompt-reading copies at the full PCIe 4.0 rate instead of through a staging buffer. It helps every model with experts in RAM, cache or not.
- Risks: the pinned pages are read in at load (35 GB up front). The RAM left must hold everything else. Windows refuses one 34-43 GB pinned range (Strata finding 13), so pin in several.

**Written** (patches `0001` and `0002`): CUDA's `ggml_backend_pin_host_buffer` / `ggml_backend_unpin_host_buffer` procs pin on request and say why they could not (read-only registration, which a read-only mmap needs). The cache pins the pages of the experts it copies from when it is created, in page-aligned ranges merged where tensors share a page (the load line says how many), only if that leaves a quarter of the RAM and at least 8 GiB to the rest, and unpins them when the context is freed. Experts loaded without mmap are in CUDA's pinned host buffer already and are left alone. `LLAMA_MOE_CACHE_PIN=0` turns it off for the comparison.

**On the reference PC the driver refused** (`host experts not pinned: operation not supported`): registering a read-only file mapping with `cudaHostRegister` is not something Linux drivers reliably allow, and the usual answer is to copy the file into memory that is pinned. llama.cpp already does that when the model is not mapped: weights overridden to the CPU then go to the backend's pinned host buffer instead of the mapped file (`llama-model-loader.cpp`, which warns that CPU overrides with mmap are slower and suggests `--load-mode none`; EuLLM hides llama.cpp's log, so it was never seen). EuLLM's `--no-mmap` asks for that.

**Measured** (same PC and model): 33.02 GiB of experts in pinned memory; writes 44.4 → 58.1 tokens/s (+31%), a prompt read 211 → 451.5 tokens/s (2.1×). `free` shows the pinned experts as shared memory (33 GB), with 24 GB of the 62 still available.

The step table with the experts pinned (micro-batch 512, 8 GiB cache):

| Window | ms per step | Up to the routers (48 layers) | In the cache | After | MiB copied | Experts in VRAM | Copying |
|---|---:|---:|---:|---:|---:|---:|---:|
| first, cache cold | 20.3 | 17.9 | 0.5 | 1.8 | 85.3 | 87.0% | 3.8 ms at 24.0 GB/s |
| the six after | 17.0–18.8 | 15.4–16.7 | 0.2–0.5 | 1.4–1.9 | 48.6–78.4 | 88.1–92.6% | 1.9–3.8 ms at 23.5–23.9 GB/s |

The copies now run at the bus's speed, and asynchronously: the host only queues them (0.3 ms "in the cache" instead of 5-8), and the GPU waits for them on its stream, so their 2-4 ms moved into "up to the routers". Without them that is 13.5-14 ms, as before: the GPU computing the 48 layers plus the waits at the read-backs. CUDA graphs stay on with the cache (one per split, captured once), so it is not kernel launches.

### Phase 3 — The CPU computes the misses, in parallel with the GPU (the core; weeks)

Built on the PR's hooks, so that it stays one change on top of it.
1. `prepare` splits the layer's (token, expert) pairs into hits and misses. Hits keep their slot. Misses point to a slot kept at zero, which adds nothing on the GPU.
2. A CPU worker pool computes each miss in place, from the host tensor: gate and up, the activation, down, all with ggml's own CPU dot products. Each result is weighted by its routing weight and summed per token into a pinned buffer. The layer's input and the routing weights are read back with the ids, a few kilobytes per token.
3. In the graph, `build_moe_ffn` (`src/llama-graph.cpp`:2365-2380) adds a per-layer input, the CPU's partial sum, to `moe_out` when the cache is on. The scheduler joins the worker and uploads the sum before the split that reads it. A split boundary is forced between the experts' matmuls and that add, so the GPU's hits are queued before the host waits for the CPU.
4. Start with every miss on the CPU. Then measure a share copied by DMA instead, as Strata does.
5. **Correctness:** the same greedy output as the cache alone, within float tolerance. With MTP, the same output with real drafts and with deliberately wrong ones.

**Target:** MTP paying off with the experts in RAM, at 70 tokens/s or more on the reference PC.

### Phase 4 — Fewer synchronisations

Replace the per-layer read-back wait with an event the host waits on while the GPU keeps going, or with Strata's doorbell. This is deep in ggml's scheduler: only after phase 1 says what the 48 waits per token cost.

### Phase 5 — A cache that starts warm

Fill the cache at startup from a profile of the experts used most, and keep that profile between runs. Strata fills 50% from a profile and reaches 72% by following the conversation. PR #26824, closed, kept such a profile beside the model file.

### Phase 6 — Reading the prompt

Copy layer N+1's experts while layer N computes, in larger blocks, borrowing the cache's VRAM as the staging area during a prompt. That is Strata §3.5, and its 2,320 tokens/s against our 206-822.

**Measured with the experts pinned** (`--no-mmap`, 33,200-token prompt): a larger micro-batch copies the experts fewer times, and takes VRAM from the cache, which `--fit` sizes from what is left:

| `--n-ubatch` | Expert cache | Writes (tokens/s) | Reads a prompt (tokens/s) |
|---:|---:|---:|---:|
| 512 | 8.00 GiB | 58.1 | 451.5 |
| 2048 | 6.75 GiB | 54.9 | 963.8 |
| 4096 | 5.25 GiB | 47.1 | 1,239.5 |

At 4,096 a micro-batch takes 3.3 s, of which copying 33 GiB at 24 GB/s is at most 1.5 s. Copied while the previous layer computes, the reading would be bound by the computing alone: about 1,800-2,300 tokens/s by that arithmetic, depending on how many experts a micro-batch leaves unused, against Strata's 2,320. This is the largest gain left on reading.

**Written** (patch `0003`, in ggml's scheduler, off unless `LLAMA_MOE_PREFETCH=1`). Nothing upstream does it at b11393: open PR #21067 prefetches whole weights and regressed time to first token, and draft #28414 (expert slots) has open correctness problems, so neither is carried.
- A MUL_MAT_ID whose experts are in host memory, of at least `LLAMA_MOE_PREFETCH_MIN_TOKENS` tokens (512, where nearly every expert is used), reads them from one of `LLAMA_MOE_PREFETCH_SLOTS` slots in VRAM (2 to 8, default 4, each the size of the largest expert tensor) instead of from the compute buffer. The slots are placed before ggml-alloc runs, which then leaves them alone, at the same addresses every micro-batch, so CUDA graphs stay valid.
- A second instance of the GPU backend, with a stream of its own, copies each staged tensor whole, as soon as its slot's previous reader is queued. Two events per slot: the reading split waits for `copied` on the GPU, and the next copy into the slot waits for `freed`, recorded after the reading split. The host waits for neither: no read-back of the router and no drain of the GPU per layer on that path.
- Below the threshold, with the MoE cache's decode steps, with several GPUs or with pipeline parallelism, the scheduler copies as before. Anything it needs and does not find (events, a second stream, room in VRAM for the slots) turns it off with one line on stderr, `moe prefetch: off, <why>`, and gives the slots back; when on, `moe prefetch: on, 4 slots of N MiB ...` once.
- CUDA only. The other backends with events hold on to what an asynchronous copy or an event takes until the next full synchronization, which comes once per graph: Vulkan creates an event per record, Metal keeps a host copy of the data per copy, thousands of each per long prompt.
- Pinned experts only: those in the GPU's pinned host buffer, where `--no-mmap` puts them (the default with `--moe-cache` when the RAM allows). From pageable memory the copy into a slot holds the host until the slot's previous reader has run, and the copies and the splits alternate again; a mapped model, even one the expert cache pinned, keeps the usual copies.
- Costs: the slots' VRAM (4 × 256 MiB on Qwen3.8-Flash-Next IQ2_XS), kept from the first long prompt on and taken from what `--fit` leaves free. They are made only if a twentieth of the card, at least 512 MiB, stays free beside them for what a compute allocates as it goes (the CUDA pool, cuBLAS), which aborts the process when it finds no room; otherwise the line on stderr says how much was needed, and lowering `--moe-cache` by the difference makes room. The padding past each copy is cleared by the kernels that read it (MMQ, MMVQ), as in the compute buffer.

**Measured** on 4 October (reference PC, Qwen3.8-Flash-Next IQ2_XS, the 33,200-token prompt of the table above), with `bench/prefetch_check.sh`: the same server flags with `LLAMA_MOE_PREFETCH=0`, then `=1`, each asked the same long question twice, greedy and with `cache_prompt: false`, then measured with `bench/speed_check.py`. In every run the four answers were the same, token for token (checksum `b2941bfb`).

| `--n-ubatch` | expert cache | slots | reading, tokens/s: off → on | writing, tokens/s: off → on |
|---|---|---:|---|---|
| 4096 | 5.25 GiB (`auto`) | 2 | 1,252.7 → 1,387.0 (+10.7%) | 52.8 → 48.8 |
| 2048 | 6.75 GiB (`auto`) | 2 | 944.7 → 1,084.9 (+14.8%) | 51.6 → 51.4 |
| 4096 | 5.25 GiB (`auto`) | 3 | 1,261.2 → 1,491.4 (+18.3%) | 52.8 → 48.8 |
| 4096 | 4.5 GiB (fixed) | 2 | 1,249.2 → 1,373.6 (+10.0%) | 50.3 → 46.1 |
| 4096 | 4.5 GiB (fixed) | 4 | 1,228.3 → 1,743.1 (+41.9%) | 48.9 → 50.0 |

- **Four slots are the step that pays,** hence the default of 4. The likely reason is the order of the copies: a layer's three expert tensors are each read by a split of their own, and the third split goes on through the next layer's attention, the longest stretch of a micro-batch. With four slots the next layer's three copies all have a free slot while that split computes; with three, the third waits for that very split to end, and with two, the second does too. At 1,743 tokens/s a 4096-token micro-batch takes 2.35 s, against 3.3 s without and the 1.8 s of computing alone: about a third of the copying still shows. Six and eight slots (the cap is 8 now) are the next measurement, at a fixed cache small enough to leave them room (`PREFETCH_SLOTS="4 6 8" STEPS=prefetch tools/gpu_night.sh`).
- **Writing** moved by −8% in three runs and +2% in the fourth. The slots are not on the writing path, which reads its experts through the expert cache (a MUL_MAT_ID of 512 tokens or more is staged, an answer's steps are of one token), and both servers of a run had the same cache (the `--fit` line of their logs: 5.25 and 6.75 GiB both times). So either the order the two servers run in (the prefetch one always second) or run-to-run noise of the size seen elsewhere on this machine: the same model wrote 111.4 and 120.5 tokens/s in two starts during the MTP measurements. `ORDER="1 0" bench/prefetch_check.sh` runs the pair the other way round, which tells the two apart.
- Optional: `nsys profile` of one micro-batch shows how much of the copying still waits for a slot.

**Measured again** on 5 October, six and eight slots and the order reversed, all at a fixed cache of 3,584 MiB (room for eight slots), `--n-ubatch 4096`, same model and prompt (`tools/gpu_night.sh`, `STEPS=prefetch`, nothing else on the GPU). Every answer the same again (`b2941bfb`).

| slots | reading, tokens/s: off → on | writing, tokens/s: off → on |
|---:|---|---|
| 4 | 1,163.2 → 1,494.2 (+28.5%) | 42.8 → 42.9 |
| 6 | 1,175.8 → 1,471.9 (+25.2%) | 42.9 → 43.1 |
| 8 | 1,247.6 → 1,480.5 (+18.7%) | 45.4 → 43.0 |
| 4, the prefetch server first | 1,207.0 → 1,497.7 (+24.1%) | 42.8 → 43.4 |

- **More than four slots buys nothing:** with the prefetch every setting reads 1,470-1,500 tokens/s. The default of 4 stays.
- **Writing does not move,** in either order: the −8% of 4 October was the order or noise, not the prefetch.
- The gain is smaller than 4 October's +42% at four slots, with a smaller cache (3.5 GiB against 4.5) and with one run each: whether the cache or the day makes the difference is not known. A run made while another night ran on the same GPU (the morning of 5 October) is left out: its four-slot server could not pin the experts and wrote 4.3 tokens/s.

## 3. Upstream

- Phase 2 is small and helps any MoE with experts in RAM: a candidate for a llama.cpp issue, then a PR.
- Phase 3 extends PR #29887's design. Discuss it with its author on that PR, with phase 1's numbers in hand.
- llama.cpp's rules apply to anything posted there (`CONTRIBUTING.md`, `AGENTS.md`):
  - code written with AI help must be disclosed, and the person submitting must be able to explain every line;
  - issues, PR descriptions and replies must be written by a person;
  - a feature starts as an issue.
- Until then our changes are patch files on top of 6b7b03a (`llama-cpp-sys-2/patches/`), regenerated at every weekly bump that breaks them; each one turns into a commit for an upstream PR as it is.

## 4. Risks

- Patches in ggml's scheduler conflict with upstream often: every weekly bump has to carry them.
- PR #29887 may change before it is merged, and our hooks with it.
- On AVX2 the i-quant dot products run at about 5 GB/s per core (Strata finding 7). Sixteen cores would exceed what DDR4 delivers, so on the reference PC RAM bandwidth caps the CPU half.
- The PR supports one GPU only.

## 5. Next step

1. ~~A q8_0 KV cache with `--n-ubatch 2048`~~ Measured: little to gain on this model. Only one layer in four has attention, so its KV cache is about 1 GiB at a 40,960-token context and `--fit` already charges it that way; q8_0 gave the expert cache 0.25 GiB more (7.00 GiB at 2,048), for 55.8 tokens/s writing and 935.7 reading. At 4,096, 53.2 and 1,195.1. `--n-ubatch 2048` without it stays the balance: about 55 writing, about 960 reading.
2. llama.cpp's MTP with the experts pinned (llama-server, `--load-mode none`): a check of three tokens copies more experts, which cost 2.5 steps at 9 GB/s and costs far less at 24. If drafting pays now, loading the MTP head from its own file in EuLLM is a smaller job than phase 3.
3. ~~`--no-mmap` on by itself with `--moe-cache`~~ Done: a load with a cache reads the model into memory when the RAM can spare the experts (`--mmap` to keep the mapping) and reads prompts 2,048 tokens at a time unless `--n-ubatch` says otherwise.
4. ~~Phase 6: copy the next layer's experts while a prompt's micro-batch computes.~~ Written (`LLAMA_MOE_PREFETCH=1`, patch `0003`) and measured: 1,743 tokens/s with four slots, against 1,228; six and eight slots read no faster than four (5 October). Next: a flag in place of the environment variable, and whether it is on by default, which needs `--fit` to keep the slots' VRAM out of an `auto` cache and one measurement of what that smaller cache costs the writing.
5. The GPU's busy time per step, to split the 13.5-14 ms before the routers into computing and waiting: phase 4 if the waiting is large.
