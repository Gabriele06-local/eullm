# Strata, piece by piece: why it is faster on the same PC

**Status:** a study, 6 October 2026. Strata's source read, without building or running it: 0.1.35, the version measured on the reference PC, and 0.1.40.1, the latest. Its paper (v0.1.35) gives the figures taken on its authors' PC: RTX 5070 12 GB, Ryzen 5 7600, DDR5.

EuLLM at `feat/engine-roadmap` cd18876: llama.cpp 6b7b03a (b11370 plus PR #29887) with patches `0001`-`0004`.

The reference PC: RTX 5070 Ti 16 GB on PCIe 4.0 x16, Ryzen 9 5950X (16 cores, AVX2), 64 GB of DDR4. The model: Qwen3.8-Flash-Next IQ2_XS, the same GGUF for both engines.

Paths are Strata 0.1.35's unless marked: `generate.cpp` is `src/program/generate.cpp`, `prefill.cpp` is `src/prefill/prefill.cpp`. The phases are those of `docs/moe-offload-plan.md`.

---

## 1. The question

On the same PC, with the same model file and the same script (`bench/speed_check.py`):

| | Writes (tokens/s) | Reads a prompt (tokens/s) |
|---|---:|---:|
| Strata 0.1.35 | 85.8 | 2,320 |
| EuLLM, the defaults with a cache (5.75 GiB of experts in VRAM, micro-batch 2048, phases 6 and 6b) | 53.2–54.9 | 1,509 |

Are the measurements right? How does Strata do it, and why don't we? In short:
- **The measurements.** The reading figure is sound. The writing figure is Strata's best case: greedy, drafting always on, a warm repeat. Its plain writing has never been measured on this PC (section 2).
- **Writing.** Plain writing is level. Strata leads with speculative decoding through its MTP layer: a check of up to four tokens costs it about two steps and yields 3.3 tokens (section 3).
- **Reading.** Strata leads on the size of the chunk: 8,192 tokens at a time, in VRAM its expert cache lends only while a prompt is read (section 4).
- **The pieces.** Nothing in either is out of reach on llama.cpp. Five pieces are missing, and one of them may be a difference in llama.cpp's MTP graph. Section 5 lists them, section 6 orders them, and section 7 says what to measure first.

## 2. Are the measurements right?

| What | Comparable? | Effect on Strata's figure |
|---|---|---|
| PC, model file, script | Yes. Strata reads the experts straight from the same GGUF file, at their offsets in it (`tests/data/native_experts/iq2_xs.txt`): gate and up IQ2_S in 34 layers, IQ2_XXS in 11 and IQ1_M in 3; down Q2_0 everywhere. They are the same 33.02 GiB that EuLLM pins. | none |
| Token counts | They are each server's own `usage`. Strata counts the end-of-turn token, but a 256-token story stops on length, so there is none. | none |
| Reading | A full read: the document starts with a fresh number, so nothing is reused. Strata's time includes tokenizing in Python and refilling its cache after the prompt. | none; if anything, lower |
| Temperature 0 | Greedy is MTP's best case. Strata drafts the most likely token and keeps it only if it equals the token the model picks, as EuLLM's `--mtp` does. | Its chat (0.6, thinking on, long answers) showed 49.7–62 tokens/s, and how much of the drop is the temperature is not known. On a 9B, EuLLM kept 58% of drafts at 0 and 56% at 0.8 (`docs/roadmap-engine-0.7-1.0.md`, 0.8-Z2). |
| Warm repeat | `speed_check` times the second of two identical requests on every server. Strata ignores `cache_prompt: false`: the second request resumes from the first one's checkpoint and reads about 5 of the request's ~25 tokens. It reads those a window at a time, about 16 ms a token by its own comment (`generate.cpp:5076-5081`). Its cache also moves towards that story's experts every 4 windows, and EuLLM's LRU keeps that story's experts too. | about 0.3 s of a 3 s request: up to a tenth of 85.8 (an estimate). The first run's speed, which `speed_check` prints, gives the size of it. |
| Drafting always on | The server refuses to start without `--spec` and `--mtp` (`generate.cpp:3838-3843`). Its plain decoding is a window of one token. | No plain figure on this PC. The paper's is 47–57 tokens/s on its own PC (finding 2). |
| CPU held at 3.4 GHz that day | Strata computes part of every step's experts on the CPU; EuLLM computes none. | if anything, lower |

Our own figures have two known biases:
- **A long prompt's last micro-batch.** When it is under 512 tokens it is copied without the prefetch: about 1 s of a 22 s read, the same for every setting.
- **Server starts.** Two starts of the same server write up to 5–8% apart (0.8-Z2), which is why every comparison here runs both orders.

**Verdict:**
- 2,320 is a fair figure.
- 85.8 is right, but it is Strata's best case, and perhaps a tenth of it is the warm repeat.
- Whether Strata's plain writing on this PC is above our 53–55 is not known.

Section 7 gives the runs that settle it.

## 3. How Strata writes

**Windows, not steps.** Every step checks T tokens: the last one accepted plus up to three drafts (setup writes `--spec 4 --spec-min-p 0.5 --mtp`).
- T is 1 on a request's first window, and whenever the draft layer's first guess has a probability under 0.5 (`generate.cpp:5385-5394`).
- The T tokens go through the 48 layers, the head and the argmax as one captured CUDA graph. There is one graph for each T from 1 to 8 (`include/strata/core/verify.hpp:1-11`, `src/core/verify.cpp:887-948`).

**Each layer's experts go three ways** (`src/core/expert_source.cpp:1649-1811`):
- **In the VRAM cache:** the GPU computes them.
- **55% of the misses, those with the lowest routing weight:** the GPU computes them too. A copy kernel inside the graph reads them over PCIe from the pinned copy of every expert into at most 16 staging slots. The 55% holds for i-quant packs on a link of 20 GB/s or more, which is probed at start, and is lower on a slower link (`generate.cpp:1716-1736`).
- **The rest:** the CPU computes them in place in RAM.
  - On a 5950X, 15 workers pinned one per physical core, plus the host thread.
  - Each missed expert is computed once for all of the window's tokens.
  - One token uses ggml's own dot products; two or more use Strata's AVX2 kernels. On i-quants that is about 5 GB/s per core.

A window costs the dense weights read once for all its tokens, plus the union of its tokens' misses on the CPU. That union is 1.75×, 2.4× and 3.05× one token's misses for T = 2, 3 and 4 (`verify.hpp:10-11`).

**No driver call inside the layers.**
- The GPU writes each layer's choice of experts to mapped memory and bumps a counter.
- The host spins on that counter, plans, starts the CPU's work and raises flags in mapped memory.
- The GPU waits on those flags with one-thread kernels inside the graph (`src/kernels/cuda/elementwise.cu:276-295`; `verify.cpp:691-797, 1062-1121`).

That is one launch and two synchronizations per window, about 5 µs a handoff. The GPU does wait for the CPU: 21.1 ms of a 42.0 ms IQ2_XS window (paper, Table 5).

**The cache.**
- It is filled at start from a profile that ships with the model.
- Every 4 windows, in each layer, an expert outside the cache used at least twice replaces one inside it used less. There are up to 96 swaps a time, and the counts fade by ×0.7.
- The swaps are copied on a side stream, and an expert is used only once its copy has landed. Nothing is copied in on demand while a window runs (`generate.cpp:2851-2889, 4507-4576`).

It finds 72–78% of the experts in VRAM on a 12 GB card (paper). EuLLM's LRU (PR #29887) finds 88–93% in `speed_check`'s story. The price is that every miss is copied before its layer computes: 2–4 ms of an 18 ms step.

**The draft layer.** The GGUF has no MTP layer.
- Setup downloads the layer's 31 tensors from Qwen's BF16 checkpoint, about 5 GB (`tools/mtp_fetch.py`).
- It repacks them, all in VRAM (`include/strata/core/mtp.hpp`):
  - the projections in Q8_0;
  - the 512 experts in Q2_0, 708 MB;
  - the model's head cut to a subset of the vocabulary (106,299 tokens, about 180 MiB).
- The drafts are argmaxes taken one after another, stopped at the first with a probability under 0.5. They cost 2.2 ms a window (Table 5).

**What it yields.**
- The probe keeps 0.89, 0.86 and 0.85 of greedy drafts at steps 1 to 3 (`mtp.hpp:4`).
- Strata's own measurements of IQ2_XS give 3.20–3.29 tokens a window at 4K–32K.
- Table 5, IQ2_XS at 4K on its authors' PC: a 42.0 ms window (GPU 14.7, CPU experts 21.1, drafting 2.2, other 4.0) yields 3.28 tokens. That is 12.8 ms a token, against 17.5–21 ms for its plain decoding (47–57 tokens/s).
- On the reference PC, 85.8 tokens/s is 11.7 ms a token.

**Where llama.cpp's MTP differs.**
- **llama.cpp:** its graph for this model normalizes the main model's hidden state one hyper-connection stream at a time (`src/models/qwen4exp.cpp:581` in our pin).
- **Strata:** it follows the vLLM implementation it was transcribed from, which normalizes all four streams together: one RMS over 10,240 values (`src/core/mtp.cpp:442`).
- **Strata 0.1.40:** it added llama.cpp's way as an opt-in, `--mtp-hnorm stream`, and published no result.

It is the one difference in the draft graph we know of. llama.cpp's MTP yielded 2.04 tokens per check of three on this model, where Strata's per-step rates would give 2.66 (1 + 0.89 + 0.89 × 0.86).

**The one llama.cpp measurement, and what was wrong with it.** llama-server, 3 October:
- 49.4 tokens/s without drafts at an 8,000 MiB cache;
- 36.3 and 35.1 with two drafts at 7,000 and 6,000 MiB, the cache cut to make room for the draft layer;
- about 15–20% under what those two caches give without drafts, by interpolation between 4,000 and 8,000 MiB.

Three conditions held it back:
- the experts were mapped, so every copy was staged by the host at 9 GB/s;
- the cache was smaller with drafts than without;
- the draft layer's experts (2.7 of its 3.85 GB, in Q8_0) were left in RAM with `-cmoed`, so the CPU computed them from pageable memory.

Of that measurement only the acceptance still stands: 52% of the drafts kept.

## 4. How Strata reads

**8,192 tokens at a time.** `--prefill auto` takes the largest chunk of 8,192, 6,144, 4,096 and so on whose buffers fit in the expert cache's last slots. It must leave at least 128 slots, and may lend at most 90% of them when the experts are pinned (`generate.cpp:3745-3770`).
- For a 32K prompt on a 16 GB card that is 8,192 tokens, with about 4.5 GiB lent. That is the sum of its buffers; an RTX 5090's log shows 3,421 slots, 4.62 GiB.
- The paper (§3.5) still describes 2,048-token chunks with FP16 experts through cuBLAS. The code of 0.1.35 does what follows.

**The loan.**
- The cache is filled hottest first, so the slots it lends hold its coldest experts.
- For the prompt those experts are marked absent and streamed like any other.
- After it, the same slots are refilled from the pinned copy in about 180 ms, and the cache is as it was, byte for byte (`generate.cpp:5739-5799`, `src/core/expert_cache.cpp:319-343`).
- Outside a prompt no VRAM is held for reading, so the cache is larger while writing.

**Streaming** (`prefill.cpp:1140-1248, 1783-1811`).
- Every expert not in VRAM is copied once per chunk: gate, up and down in one copy, on a stream of its own.
- A thread issues the copies up to 384 experts ahead of the computing, about 0.9 of a layer.
- The experts already in VRAM are read where they are.

**Kernels.**
- The experts go through llama.cpp's own MMQ kernels, compiled from llama.cpp (`src/prefill/moe_mmq.cu:1-3`): the same ones llama.cpp runs for us. IQ1_M (3 layers) goes through FP16 and cuBLAS.
- The dense weights go through cuBLAS in FP16.
- The sparse attention has kernels of Strata's own.
- No expert runs on the CPU while reading.

**What the bus carries.**
- **Strata:** about 30 GB per 8,192-token chunk by our count from its cache's size, or 3.7 MB a token. That is 1.25 s at 24 GB/s of a 3.5 s chunk, so the GPU's computing (about 0.43 ms a token) sets the pace.
- **EuLLM, at 2,048 with phase 6b:** 14 MB a token, 1.23 s per micro-batch of 1.36 s, so the bus sets the pace.

**0.1.40.** It adds int8 tensor-core kernels of its own for the experts:
- the default for the Q2_0 pack since 0.1.36: 32K prompts went from 2,170 to 2,653 tokens/s on a 5070;
- opt-in for the IQ packs: +3% at 32K.

## 5. The puzzle

| # | Piece | Strata | EuLLM | Have it? |
|---|---|---|---|---|
| 1 | Every expert pinned in RAM | its own pinned copy (`cudaHostRegister`) | CUDA's pinned buffer, `--no-mmap` (phase 2) | ✓ |
| 2 | A VRAM cache of the most used experts | profile, then swaps every 4 windows: 72–78% | LRU (PR #29887): 88–93% | ✓ |
| 3 | Plain writing | 47–57 (paper, its own PC) | 53–55; 58.1 with an 8 GiB cache | ✓ level; on this PC, to measure |
| 4 | Copies overlapped with computing while reading | a thread, 384 experts ahead | the prefetch, 4 slots (phase 6) | ✓ |
| 5 | Experts already in VRAM not copied while reading | ✓ | phase 6b | ✓ |
| 6 | Expert kernels while reading | llama.cpp's MMQ | llama.cpp's MMQ | ✓ the same; whether our computing per token matches is under piece 7 below |
| 7 | **Large chunks that cost the writing nothing: VRAM lent by the cache during a prompt** | 8,192 tokens, about 4.5 GiB lent | 2,048, with 1.1 GiB of compute buffer and 1 GiB of slots held for good | ✗ |
| 8 | **A draft layer that guesses well** | 3.28 tokens per check of four | llama.cpp's graph: 2.04 per check of three | ✗ (the norm?) |
| 9 | **A small draft layer, in VRAM, from a file of its own** | 0.9 GB: Q2_0 experts, a vocabulary subset | `--mtp` reads the layer only from the model's own GGUF; Flash-Next's is a separate 3.85 GB Q8_0 file | ✗ |
| 10 | **A cheap check: the CPU computes the misses while the GPU computes the hits** | 45% on the CPU, 55% by the copy kernel | every miss copied before its layer computes; the CPU idle | ✗ phase 3 |
| 11 | No host synchronization per layer | a doorbell, one graph per window | 48 read-backs of the router per step | ✗ phase 4, worth unknown |

Pieces 1 to 6 are in place; 7 to 11 are missing, and 7 and 10 are the large ones. None needs anything llama.cpp lacks:

| # | What it takes | Effort |
|---|---|---|
| 7 | A buffer the cache lends to the scheduler for the length of a prompt, then refills: the largest patch so far, in llama-context, the cache and the CUDA backend | a week or more |
| 8 | One line of the graph, if the norm is the cause | hours, plus a llama-server run |
| 9 | EuLLM loading a second file (llama.cpp already loads such a file: `src/models/qwen4exp.cpp:179`) and a quantization | a day or two |
| 10 | Phase 3 | weeks |
| 11 | Phase 4, deep in ggml's scheduler | weeks |

### What each should give

These are estimates, to be replaced by the measurements of section 7.

- **7, the loan.**
  - Reading: at 8,192 tokens a micro-batch carries 0.17 ms a token of copying, so the computing sets the pace.
  - Our computing is a micro-batch's time less its copy, without the prefetch. Phase 6's runs give 0.29–0.34 ms a token at 2,048 and 0.43–0.50 at 4,096, which brackets Strata's 0.43.
  - That gives between 2,000 tokens/s (0.5 ms a token) and 3,300 (0.3), around Strata's 2,320.
  - The figure grows with the micro-batch, and it should not. Either some step of llama.cpp's graph for this model costs more per token as the micro-batch grows, or fewer than all the experts are copied at 2,048. The second is plausible: at 512 a micro-batch took 1.13 s, less than the 1.48 s of copying them all.
  - The 6,144 and 8,192 runs of section 7 settle both questions.
  - Writing: the 2.1 GiB held for reading go back to the cache, which wrote 58.1 tokens/s at 8 GiB against 53–55 at 5.75.
- **8 and 9, a good draft layer.** To write at 85.8 tokens/s, each token has 11.7 ms.
  - A check yielding 3.28 tokens may cost 38 ms, about two of our steps.
  - A check yielding 2.04 may cost only 24 ms. That is one step plus the extra copies of its misses, with nothing left for drafting.
  - Acceptance comes first.
- **10, phase 3.**
  - A step's copies, 2–4 ms of 17–19, leave the GPU's path: about 15–25% faster plain writing.
  - A check of four tokens loses three times as much copying.
- **11, phase 4.** It gives whatever part of the 13.5–14 ms before the routers is the GPU waiting rather than computing. That is unknown; the GPU's busy time while writing (section 7) gives it.

## 6. Order

One piece at a time, each measured before the next:

1. **The measurements of section 7.** No code; they replace the estimates above.
2. **The loan (piece 7).** The rest of the reading road, and a larger cache for writing.
3. **The draft layer (pieces 8 and 9).** Check the norm on llama-server, then `--mtp` from a file of its own with Q2_0 experts.
4. **Phase 3 (piece 10).** Sized by what a check costs once step 3 runs.
5. **Phase 4 (piece 11).** Only if the GPU idles.

Steps 2 and 3 can swap: step 3 is smaller, and step 2's gain is surer.

## 7. What to measure first

**Strata, as installed (0.1.35).**

Its `strata-iq2_xs.json` takes engine flags in `args` and environment variables in `env`. Add `"env": {"STRATA_DECODE_TIMING": "1", "STRATA_PREFILL_TIMING": "1"}`. Its log then prints, for every request:
- a window's time, split into the GPU, the CPU's experts, the drafting and the rest;
- a prompt's time by phase.

Then four runs of `speed_check`:

| Run | Change | What it gives |
|---|---|---|
| A | none | the timings behind 85.8 and 2,320 on this PC, and the first run's speed |
| B | `--spec-min-p 1.0` instead of 0.5 | windows of one token, its plain writing (the draft layer still guesses once a window) |
| C | A plus `--prompt-cache 0 --adapt-swaps 0` | no warm repeat |
| D | A, with `speed_check --temperature 0.7` | the temperature's cost |

**EuLLM.**

| What | How | What it decides |
|---|---|---|
| The GPU's busy time while writing | `nvidia-smi dmon -s u` during `speed_check`, or `nsys` | piece 11 |
| Reading with larger micro-batches | `--n-ubatch 6144` and `8192` with a small fixed cache (`MOE_CACHE=1024`) | the ceiling the loan would reach |
| Writing on a fresh server | a server that has read no long prompt yet | whether phase 6b costs 1.5 tokens/s of writing |

**llama.cpp's MTP on Flash-Next, with the experts pinned.**

`bench/mtp_test_d.sh` with llama-server built from our pin:
- the experts pinned;
- the cache the same size with and without drafts;
- the draft layer in VRAM, with its experts requantized to Q2_0.

It measures the drafts kept and the speed with two and with three drafts; then the same with the norm over all four streams.
