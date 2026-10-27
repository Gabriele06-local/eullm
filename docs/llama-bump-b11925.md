# llama.cpp b11925: what moved, and how to check it before it lands

Branch `chore/llama-cpp-b11925`, from `main` at be0cf9e (release 0.7.95). The
llama.cpp submodule moves from b86d2f0 (7 October) to b11925 (27 October, 443
commits), the latest tag on the `eullm/llama.cpp` mirror that day. `llama-cpp-rs`
stays at 0.1.159: 0.1.160 only adds WebGPU, Emscripten, PowerPC and Android
build changes, none of which EuLLM builds.

Nothing here is merged until the reference PC and the three sites have run it
(sections 3 and 4).

## 1. What changed

### Patches (`engine/vendor/llama-cpp-rs/llama-cpp-sys-2/patches`)

| Patch | At b11925 |
|---|---|
| `0001` pin host memory | applies as it was |
| `0002` pin the host experts, step statistics | rewritten for the cache across several GPUs (#30112): the pages are pinned once, through the backend of the first GPU that caches experts, which registers them as portable for every GPU of that backend |
| `0003` prefetch on a second stream | the scheduler is now made in `llama_context::sched_new`, which sets the prefetch; the scheduler's new realloc callback kept, beside the prefetch's placement check |
| `0004` prefetch reads the cache | the same move; the prefetch still turns itself off with more than one GPU, so the multi-GPU cache never meets it |
| `0005`, `0006`, `0010` | apply as they were |
| `0007` one `ssm_out` product | **dropped**: upstream does the same (#30823) |
| `0008` + `0009` eviction | **one patch**, `0008-moe-cache-evicts-by-pass-and-decaying-counts`, replacing upstream's new 50/50 SLRU (#30637). Replayed through `bench/moe_cache_sim` (the SLRU added to it), the SLRU finds what the LRU found: the same where a step fits, and 0% of the experts in VRAM where it overflows the cache (Qwen3-235B at 32 sequences, Qwen3-Coder-480B at 8 and 32), where this patch finds 73%, 60% and 33% |
| `0011` pool keeps VRAM free | **halved**: upstream now drops a context's CUDA graphs when its pool frees memory (`invalidate_cuda_graphs`), so the patch's own per-device flush counter is gone and its threshold flush calls upstream's function |
| `0012` meta split loads | the loader's buffers are a vector now: one line |

### The wrapper and the Rust side

- **The chat output parser** (llama.cpp #30210) is a C++ object with no saved
  form. It used to cross to Rust as a string and be rebuilt on every parse; it
  now crosses as an owned handle, `llama_cpp_2::model::ChatParser`, freed with
  its last clone. Whether reasoning is split out of the content is now decided
  when the prompt is rendered (`reasoning_format = AUTO`, as before), no longer
  at parse time.
- **The rendered prompt** is a list of parts (#30389), template text and user
  input; the wrapper joins them (`str()`), which is what the prompt was before.
  llama.cpp marks the user's parts so that special tokens inside them are not
  parsed; EuLLM does not use that yet (a later change, not this bump).
- `llama_perf_context_data` has two new counters; `engine/src/llama_archs.rs`
  regenerated: eight more architectures (`beam`, `deepseek41`, `flint`,
  `kolibri1`, `maion-coder`, `muse-spark`, `sensenova_u1`, `vllm-decision2`).

### What upstream added that EuLLM can use next (not in this branch)

- Rejection sampling for speculative drafts (#30095, #30213, #30663) in
  `common_speculative`, which `--mtp` goes through.
- n-gram speculation types and a saved n-gram cache (#22055).

## 2. Checked here (cloud container, 4 cores, no GPU)

- CPU build; `cargo test`: 756 unit tests and the 11 `llama_patches` tests pass.
- `cargo clippy --all-targets`: no new warning. `cargo check --features multimodal`: clean.
- The two CUDA files the patches change (`ggml-cuda.cu`, `mmvf.cu`) compile
  with nvcc 13.4 for sm_89, CUDA graphs on.
- The real-model tests (`--ignored real_model`, release build, CPU): 35 of 35.
  Generation on Gemma-3-270M, MTP on Qwen3.5-0.8B-MTP (drafts do not change
  the answer), the hybrid state on Qwen3.5-0.8B, the reasoning block and the
  chat parser on Spark-X2.5-1.7B, decisions on Qwen3-0.6B Q8_0 and
  stories260K F32, embeddings on Qwen3-Embedding-0.6B, finetune on stories260K
  F32. Two decision tests need what their own comments say: an F32 model (on
  Q8_0 the batched and separate results differ by rounding) and a process of
  their own (the backend is initialised once per process).

Not checked here: any GPU run, the full CUDA, ROCm, Vulkan and Metal builds,
Windows.

## 3. On the reference PC (RTX 5070 Ti)

Build the branch next to the installed 0.7.95:

```bash
git fetch origin && git checkout chore/llama-cpp-b11925
git submodule update --init --recursive
cd engine && cargo build --release --features cuda
```

Then the same model and flags on each binary, one server at a time:

1. **Speed and the same answer**: start each binary with the model and run
   `python3 bench/speed_check.py --url http://127.0.0.1:11434/v1` against it.
   Writing and reading speeds, and the answer's hash. Hashes may differ where
   llama.cpp's numerics moved; the text should still read the same.
2. **The expert cache**: `bench/prefetch_check.sh` with each binary on
   Qwen3.8-Flash-Next IQ2_XS (`SETTINGS="0 4"`), and one run with
   `LLAMA_MOE_CACHE_STATS=64`: the share of experts in VRAM should be the
   0.7.95 one or better.
3. **MTP**: Qwen3.5/3.6 with `--mtp 2`, the share of drafts kept in the log.
4. **Chat with reasoning and tools** (the rewritten parser): a reasoning model
   (Qwen3, Spark) and one tool call through Open WebUI or `/v1/chat/completions`
   with `tools`: reasoning in `reasoning_content`, the call in `tool_calls`.

## 4. On the three sites

Each site builds this branch into a build directory of its own, so that no
running job's binary changes, and runs it beside the current engine:

- **LUMI-G**: `bash tools/lumi/build_engine.sh` from a checkout of this branch,
  with `CARGO_TARGET_DIR` set to a new directory; then the smoke job and one
  round of the usual campaign. ROCm is where `0011` came from: a long run should
  show the pool staying under the VRAM (`GGML_CUDA_POOL_KEEP_FREE_MB` lines in
  the log) and no NaN.
- **JUPITER**: `bash tools/jupiter/build_engine.sh llama-b11925` (the label
  names the build directory); `sbatch_compare.slurm` against the current build.
- **Leonardo**: the jobs use the released binary; this branch is built from
  source there (`docs/cineca/leonardo.md`), or tested once it is released.

The branch reaches each site's own branch through `main`, once the PC and the
sites agree.
