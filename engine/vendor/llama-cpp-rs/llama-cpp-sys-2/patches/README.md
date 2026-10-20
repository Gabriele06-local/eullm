# EuLLM's patches to llama.cpp

Changes EuLLM needs in llama.cpp before upstream has them. The build script
applies every `*.patch` here, in name order, to a copy of the `llama.cpp`
submodule under `OUT_DIR` and builds from that copy (`../llama_patches.rs`);
the submodule itself is never modified.

Each file is a `git diff` against the commit the submodule is pinned to, with a
description of the change above the first `diff --git` line. To make one:

```bash
cd engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp
# edit, build and test the change in place, then:
{ printf 'what the change does, and why\n\n'; git diff -- <files>; } > ../patches/NNNN-short-name.patch
git checkout -- <files>        # the submodule goes back to the pinned commit
```

A later patch may change a file an earlier one changed; it is then a diff
against the tree with the earlier patches applied.

Keep new declarations out of the headers every backend's sources include
(`ggml.h`, `ggml-backend.h`, `ggml-impl.h`): a patch that changes one rebuilds
all of them, the CUDA kernels among them, on every machine that builds it and
in every release build, where sccache finds none of them cached. `0003`
declares its one function in `ggml-backend-prefetch.h`, a header of its own,
for that reason.

When the submodule moves, `cargo test --test llama_patches` in `engine/` says
which patches still apply. Regenerate the ones that do not against the new
commit, and delete the ones upstream has taken.

| Patch | What it does |
|---|---|
| `0001-cuda-pin-host-memory-on-request.patch` | CUDA procs to pin and unpin host memory on request (read-only registration), reporting why pinning failed |
| `0002-moe-cache-pin-host-experts-and-step-statistics.patch` | The MoE expert cache pins the experts it uploads from (`LLAMA_MOE_CACHE_PIN=0` to compare), and `LLAMA_MOE_CACHE_STATS=N` reports where a decode step goes |
| `0003-moe-prefetch-experts-on-a-second-stream.patch` | With `llama_context_params.moe_prefetch_slots` set (EuLLM's `--moe-prefetch`), the experts in host memory of a MUL_MAT_ID of 512 tokens or more are copied into that many VRAM slots, 2 to 8, on a second stream of the GPU while the splits before it compute (phase 6 of `docs/moe-offload-plan.md`); one GPU of the CUDA backend (CUDA, or ROCm since 14-10-2026), experts in pinned host memory (a model loaded without mmap), slots only where they leave VRAM free for the compute; off unless set. A staged copy skips the copy callback of `llama-context` |
| `0004-moe-prefetch-reads-cached-experts-from-vram.patch` | The prefetch copies the experts the MoE cache holds from its banks in VRAM, on the GPU's own stream, and only the others over the bus (`llama_moe_cache::sched_lookup`; `LLAMA_MOE_PREFETCH_FROM_CACHE=0` to compare); a long prompt reads about 9% faster (phase 6b) |
| `0005-penalties-by-index-on-the-whole-vocabulary.patch` | The repeat, frequency and presence penalties change the logits of the recent tokens at their own index when the candidates are the whole vocabulary in order, as `llama_sampler_sample` builds them: `penalty_last_n` lookups instead of one per token of the vocabulary (1,196 to 2 µs per token on 248,320 tokens), the same logits; any other array is scanned whole, as before |
| `0006-cuda-one-warp-per-row-for-short-bf16-matrix-vector-products.patch` | One-token bf16 matrix-vector products with at most 512 columns and at least 4,096 rows (the low-rank "up" projection of the hyper-connections, 320 to 10,240) give each row to a warp instead of a block of 160 threads with a shared-memory reduction: 17 to 9 µs, twice a layer. Same fp32 arithmetic, the sum in another order; `GGML_CUDA_MMVF_SHORT_ROWS=0` restores the generic kernel |
| `0007-models-one-ssm-out-product-for-all-tokens.patch` | Qwen3.5/3.6, Qwen3-Next and qwen4exp shape the input of a linear-attention layer's output projection (`ssm_out`) as one matrix of all the ubatch's tokens instead of one per sequence: a decode step of n sequences read the weights n times through the matrix-vector kernel, now once (0.71 ms a call at 32 sequences on an MI250X GCD, 22% of the GPU time) |
| `0008-moe-cache-evicts-the-most-recent-slot-when-a-pass-overflows.patch` | The MoE cache evicts its least recently used slot only while that slot's expert went unused for a whole pass over the layers; when every slot was used within the last pass (a step wants more experts than the cache holds), a miss takes the most recently used slot outside the current layer, so the slots filled early in a step stay instead of every expert being copied again (`LLAMA_MOE_CACHE_EVICT=lru` to compare). `LLAMA_MOE_CACHE_STATS` counts steps of up to 32 tokens and the copies into a slot used within the pass; `LLAMA_MOE_CACHE_TRACE=FILE` writes the experts every layer selected, for `bench/moe_cache_sim` |
| `0009-moe-cache-evicts-by-decaying-counts.patch` | The MoE cache keeps a count for every expert of every layer, cached or not, that each selection raises by 1 and that halves every 4 passes; a miss takes, among the stale slots, the one with the lowest count, and when every slot was used within the last pass, the lowest count among the slots used earlier in the pass. Replayed on the routing of Qwen3-235B and Qwen3-Coder-480B from LUMI-G, 1-4 more points of the experts in VRAM than 0008, never fewer; on one MI250X GCD 3-6% faster than 0008 wherever it copies less (`LLAMA_MOE_CACHE_EVICT=pass` for 0008's eviction, `lru` for upstream's) |
| `0010-meta-split-weights-copied-as-gathered-rows.patch` | With `--split-mode tensor`, a weight split along axis 0 (`attn_output`, `ffn_down`, `ffn_down_exps`: part of every row on each device) is gathered row by row into a host buffer and copied to each device in pieces of 64 MiB, instead of one 2D copy from pageable memory; on LUMI-G's MI250X tensor loads ran at 100-230 MB/s, 10-50 times slower than layer (`GGML_META_GATHER_ROWS=0` restores the 2D copy) |
