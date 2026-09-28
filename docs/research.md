# Research and experiments

We invest some engineering time in evaluating new techniques before deciding whether to ship them. Both outcomes live here: what we tested and set aside, and what we tested and now ship. Each subsection says which.

### TurboQuant KV cache compression — tested, on hold

Between Q1 and Q2 2026 we tested integrating TurboQuant (Google Research, ICLR 2026) — a Walsh-Hadamard rotation + Lloyd-Max codebook approach to KV cache quantization — via the [AmesianX/llama.cpp](https://github.com/AmesianX/llama.cpp) fork (v1.5.3). We shipped three experimental TurboQuant variants in v0.5.x (Linux/macOS/Windows). The reproducible benchmarks (Qwen3-8B at 264 k context on a 16 GB RTX 5070 Ti, ~77 tok/s; full quality runs on the LM Eval Harness) are archived under [`bench/results/turboquant_20260329_224511/`](../bench/results/turboquant_20260329_224511/) and the engineering write-ups under [`docs/turboquant-quality-report.md`](turboquant-quality-report.md) and [`docs/turboquant-kv-stress-report.md`](turboquant-kv-stress-report.md).

**Why it's not in v0.5.8 onwards:**

- **TurboQuant as a whole is not in upstream llama.cpp** — three independent PRs ([#21089](https://github.com/ggml-org/llama.cpp/pull/21089), [#23617](https://github.com/ggml-org/llama.cpp/pull/23617), [#23962](https://github.com/ggml-org/llama.cpp/pull/23962)) are either stalled, closed, or rejected, and the main maintainer has voiced skepticism about marginal quality gains over the standard Q4_0 KV cache at the same bit-width. The *rotation* half of the idea did land upstream, separately and without the codebook — see the next subsection.
- Our integration depends on a fork maintained by a single individual (`AmesianX`); production exposure to a single-maintainer fork that may diverge or be archived isn't a trade-off we want to ship under a "sovereign" engine claim.
- The TurboQuant variant build was the long-pole of every CI release (multi-hour Windows CUDA TurboQuant) for a feature whose practical advantage over standard quantized KV cache (`--cache-type-k q4_0 --cache-type-v q4_0`) hasn't been clearly established in our quality runs.

None of that has changed. What did change is that a narrower derivative — the rotation without the codebook — was merged upstream by the maintainer himself and reached us for free, which is the outcome this section was holding out for. It is described below.

The R&D code lives in git history at tag [`EuLLM-v0.5.7`](https://github.com/eullm/eullm/releases/tag/EuLLM-v0.5.7); the corresponding binaries remain downloadable from that release for anyone who wants to reproduce.

### Hadamard rotation of Q/K/V — upstream, shipped, on by default

[llama.cpp #21038](https://github.com/ggml-org/llama.cpp/pull/21038), *"llama : rotate activations for better quantization"*, was merged upstream on **1 April 2026**, authored by llama.cpp's lead maintainer. It is the Walsh-Hadamard half of the TurboQuant idea without the Lloyd-Max codebook: Q, K and V are multiplied by an orthonormal Hadamard matrix before the KV cache is written and rotated back after attention. Rotation spreads the outlier channels that make a low-bit KV cache lossy, and because the matrix is its own inverse it changes nothing mathematically — the cache stays in the ordinary `q4_0` / `q8_0` formats, readable by any llama.cpp.

It reached us exactly as the section above predicted, through a submodule bump and no engineering of our own. The llama.cpp we pin (`7ab4ee7`, b11100) carries it in `src/llama-kv-cache.cpp`, with a Fast Walsh-Hadamard Transform kernel for CPU, CUDA/HIP, Metal, Vulkan and SYCL, so every published EuLLM binary — CPU, CUDA, ROCm, Vulkan, Metal — contains it.

**It engages only where it can help.** The rotation is enabled per cache, at model load, when the KV type is quantized **and** the head dimension is a multiple of 64:

| KV cache | rotation |
|---|---|
| `--cache-type-k q4_0 --cache-type-v q4_0` (or `q5_0`, `q5_1`, `q8_0`) | **on** |
| `f16` — the default | off: nothing to de-outlier, and no cost paid |
| any head dimension not a multiple of 64 | off |
| `LLAMA_ATTN_ROT_DISABLE=1` in the environment | off everywhere — the A/B switch |

The server log states which branch was taken, at load time:

```
llama_kv_cache: attn_rot_k = 1, n_embd_head_k_all = 128
llama_kv_cache: attn_rot_v = 1, n_embd_head_k_all = 128
```

What we have **not** done is measure it on our own models: no perplexity or KL-divergence run yet compares rotated against unrotated `q4_0` KV. Until that exists, the honest claim is that the technique ships and is active, not that we have quantified what it buys.
