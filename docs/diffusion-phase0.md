# Diffusion LLMs, phase 0: measure, then decide

**Decision: we are not integrating a diffusion path into the engine.** Measured
17–18 September 2026 on Leonardo (A100-SXM-64GB) and LUMI-G (MI250X, one GCD),
at a cost of roughly 0.2 GPU-hours per site. The conditions that would reopen
the question are named at the end, and they are two, not one.

This document exists because the alternative was leaving the result in a chat
log. The scripts that produced it are `tools/lumi/sbatch_diffusion.slurm` and
`tools/leonardo/sbatch_diffusion.slurm`, with the matching
`build_diffusion_cli.sh` next to each; they emit `BENCH_RESULT {...}` in the
same shape as the autoregressive benchmarks, so both arrangements collect with
one grep.

## What was measured, and what was not

`llama-diffusion-cli` from the llama.cpp we vendor (`4d917609`, b10818), built
per site: HIP `gfx90a` on LUMI, CUDA `sm_80` on Leonardo. Dream-v0-Instruct-7B
Q4_K_M ([bartowski](https://huggingface.co/bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF),
quantised with llama.cpp), timestep schedule, `--diffusion-eps 0.001`,
confidence-based algorithm, temperature 0.2, seed 42.

This is the **uncached** path, which is not an oversight but what the code is:
there is not one line of KV-cache handling in `examples/diffusion/diffusion.cpp`.
Every step is a whole-sequence `llama_decode`. For Dream and LLaDA 1.x that is
inherent — bidirectional attention over a sequence whose token values change
each step means every position's K/V changes with them.

Not measured, and the gap matters when reading the rest: the LLaDA rows of the
sweep produced no result in the output captured, and LLaDA 2.x — the block
diffusion family, where confirmed blocks are frozen and past blocks cache
*exactly* rather than approximately — cannot be loaded at all by this llama.cpp.
`llm_arch_is_diffusion` knows `DREAM`, `LLADA`, `LLADA_MOE` and `RND1`
(`src/llama-arch.cpp:1091`). No `llada2`.

## Measurements

| site | max_length | steps | ms/step | sampling/step | host share | forward/step | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 | 512 | 256 | 166.91 | 58.67 | **35.2%** | 108.24 | 11.98 |
| MI250X | 512 | 256 | 218.31 | 40.93 | **18.7%** | 177.38 | 9.16 |
| MI250X | 256 | 128 | 113.20 | 20.20 | 17.8% | 93.00 | 17.67 |

GPU utilisation, sampled by the job during the timed window: 61.3% (A100,
512), 79.5% and 56.1% (MI250X, 512 and 256). `tok/s` counts `max_length` and
therefore includes the prompt — an upper bound by roughly the prompt's share,
about 4% here. The relative comparisons are unaffected: same prompt everywhere.

**The A100 is 1.31× faster per step.** Worth setting beside the autoregressive
result for the same two machines, where LUMI came in about 6% *behind*
Leonardo: the two machines order differently on the two workloads, because
autoregressive decode at batch 1 is bandwidth-bound while this is not. That
divergence is the kind of thing the EuroHPC allocation exists to characterise,
and it is more interesting than either number alone.

## Against autoregressive, which is the comparison that decides

On the MI250X, single stream, from `docs/lumi/lumi-g.md`:

| | model | tok/s |
|---|---|---:|
| autoregressive | Qwen3.8 **27B** Q8_K_XL | **27.3** |
| diffusion, uncached | Dream **7B** Q4_K_M | **9.16** |

Three times slower on a model four times smaller at half the bit width. At
equal model size the gap would be substantially worse. This is the number the
decision rests on.

## The structural result

This is the part worth citing, independently of the verdict.

**Host-side sampling is 18–35% of every step, and it is site-dependent.** The
fraction is not a property of the method: 35.2% on the A100, 18.7% on the
MI250X, where the sampler is also *faster in absolute terms* (40.93 against
58.67 ms) despite the slower GPU. Sampling is host work — confidence
computation, candidate selection — so its cost tracks the CPU, not the
accelerator. GPU utilisation of 61.3% on the A100 says the same thing from the
other side: the device idles while the host works.

**Both terms scale linearly with sequence length.** Doubling `max_length` from
256 to 512 on the MI250X moved ms/step by 1.93× and sampling/step by 2.03×.
That is the signature of an uncached full-sequence decode, and it says the
sampler's cost grows with `max_length` — not with how many positions are
actually being unmasked.

**Output length is the physical micro-batch.** `diff_params.max_length =
params.n_ubatch` (`examples/diffusion/diffusion-cli.cpp:205`), so generating
1024 tokens requires a 1024-token micro-batch on every step. This is a hard
ceiling on output length, not a tuning parameter, and it is independent of
speed.

Together these explain a decomposition published in
[ggml-org/llama.cpp discussion #22972](https://github.com/ggml-org/llama.cpp/discussions/22972),
where the block-wise KV cache was worth **1.02×** and a sampler submitting only
the active block **5.93×**. If the host term is a fifth to a third of each step
and scales with `max_length` rather than block size, then at block 32 over 512
positions the sampler change removes sixteen times more host work than the
cache can remove device work. The cache was not disappointing; it was aimed at
the smaller term.

## Ceilings: what the sampler bounds, and what it does not

If the forward pass were free, the **full-sequence sampler in `diffusion-cli`**
would still hold throughput to:

| site | sampler term | ceiling with a free forward pass |
|---|---:|---:|
| A100 | 58.67 ms/step | **34.1 tok/s** |
| MI250X | 40.93 ms/step | **48.9 tok/s** |

**These bound `diffusion-cli`'s sampler, not the diffusion path.** A sampler
that submits only the active block divides that term by `max_length / block`.
Projected from our own measurements at block 32 — arithmetic, **not verified**:

| site | sampler term, block 32 | share of the current step |
|---|---:|---:|
| A100 | 3.67 ms/step | 2.2% |
| MI250X | 2.56 ms/step | 1.2% |

At which point the host term stops binding and the ceiling becomes whatever a
cached forward pass costs — a quantity we have not measured, and the reason no
throughput figure is projected here. Quoting 34 and 48.9 tok/s as "the limit of
diffusion" would be wrong twice over: they are one implementation's sampler,
and that implementation is the one the upstream work replaces.

## Reopening conditions

Both, not either:

1. **The block-wise KV cache merged into llama.cpp upstream.** Not carried as a
   patch on our mirror. `README.md` records what that costs, in the TurboQuant
   section: production exposure to an unmerged fork maintained by one person is
   a trade this project has already made once and undone.
2. **`llada2` in master.** Without it the first condition buys a cache for Dream
   and LLaDA 1.x, where bidirectional attention means the cache is approximate
   and every position's K/V still changes each step. Block diffusion is the
   family where confirmed blocks freeze and past blocks cache exactly, and it is
   the model that would justify the path at all.

When both hold, the cost is a submodule bump, someone else maintains the code,
and these scripts re-run unchanged against it. That is the asset this phase
leaves behind, more than the numbers.

## Reproducing

```bash
git init llama.cpp && git -C llama.cpp fetch --depth 1 \
    https://github.com/eullm/llama.cpp 4d9176092d00586775af140581bb0b558ddc4389
git -C llama.cpp checkout FETCH_HEAD
bash build_diffusion_cli.sh          # finds llama.cpp beside it
sbatch sbatch_diffusion.slurm        # LUMI: --account=<project>; Leonardo: --time=00:30:00
```

Both job scripts preflight the two `GGML_ASSERT`s on block divisibility, refuse
a Leonardo run whose log carries `failed to initialize CUDA` — the failure mode
that yields a plausible-looking number measured on the CPU — and label `tok/s`
an upper bound when the prompt cannot be counted, rather than publishing it
silently.

Job IDs: Leonardo `58045060`, LUMI `22129806`.
