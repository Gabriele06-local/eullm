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
| `0002-moe-cache-pin-host-experts-and-step-statistics.patch` | The MoE expert cache pins the experts it copies from (`LLAMA_MOE_CACHE_PIN=0` to compare), and `LLAMA_MOE_CACHE_STATS=N` reports where a decode step goes |
| `0003-moe-prefetch-experts-on-a-second-stream.patch` | With `llama_context_params.moe_prefetch_slots` set (EuLLM's `--moe-prefetch`), the experts in host memory of a MUL_MAT_ID of 512 tokens or more are copied into that many VRAM slots, 2 to 8, on a second stream of the GPU while the splits before it compute (phase 6 of `docs/moe-offload-plan.md`); one CUDA GPU, experts in pinned host memory (a model loaded without mmap), slots only where they leave VRAM free for the compute; off unless set |
| `0004-moe-prefetch-reads-cached-experts-from-vram.patch` | Where a context has both the expert cache and the prefetch, the experts the cache holds are copied into the prefetch's slots from its banks in VRAM, on the GPU's own stream, the others over the bus on the copy stream (phase 6b): `ggml_backend_sched_set_moe_prefetch_lookup`, answered from the cache's LRU; `LLAMA_MOE_PREFETCH_FROM_CACHE=0` copies everything over the bus, to compare |
