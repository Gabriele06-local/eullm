# llama.cpp bump to the merged MoE cache: where the work stands (8 October 2026, night)

Branch `feat/llama-moe-cache`, from `feat/engine-roadmap` at 88f3a33 (PR #741 is separate and was green).

## Done (builds with CUDA on the reference PC, RTX 5070 Ti)
- Submodule pinned to `b86d2f0` (upstream master with PR #29887, merge commit d6cf9ac). No release tag contains the PR yet (latest checked: b11479): move the pin to the first tag that does, when it exists.
- `wrapper_common.cpp`: `common_chat_parse` now takes a `common_chat_input`.
- Patches: `0001` unchanged; `0002` (pin the host experts, step statistics) rewritten for the new `llama-moe-cache.cpp` (stats are timed inside `copy()`; a new step starts when the layer index does not grow); `0003` (prefetch on a second stream) ported to the new scheduler (the MoE-cache entries are gone upstream; a staged copy skips the copy callback); `0004` ported on branch `feat/prefetch-reads-moe-cache` (the prefetch reads the experts the cache holds from its banks, `llama_moe_cache::sched_lookup`; 8 October: 1,516 vs 1,387 tokens/s reading, same answer).
- Found while measuring: `--fit` from 0.7.40 counted `per_layer_token_embd.weight` (a 28.8 GB lookup table in part 2 of Qwen3.8-Flash-Next IQ2_XS, always in RAM) as VRAM, so `--moe-cache auto` found no room and the server ran at 181 tokens/s reading, 6.4 writing. Fixed in `fit.rs` (`is_host_only_tensor_name`), with a test.

## Update (8 October, morning)
Measured with the fit fix (`bench/prefetch_check.sh`, `SETTINGS="0 4"`): prefetch off 986.9 read / 56.1 write tokens/s, cache 6.75G; on 1,389.7 / 57.3, cache 5.75G; the same answer in both (8e1cf173; the old binary's b2941bfb differs only because llama.cpp's numerics moved). Against the old binary: writing +10%, reading with the prefetch -9% (the lost 0004). `cargo test` (712), clippy (with and without multimodal) and the CI on the PR are green; the generated architecture list was regenerated (`LC_ALL=C tools/gen-llama-archs.sh`: without `LC_ALL=C` it sorts by the local language and fails the sorted test). Docs and CHANGELOG are updated. Still open: the 0004 gain on the prefetch path, the rest of the list below, and `bench/prefetch_check.sh` / `bench/fresh_check.sh` still know a `:bus` setting that no longer does anything.

## To do (as written the night before)
1. Measurement, same script for both binaries: `SETTINGS="0 4" bench/prefetch_check.sh BINARY MODEL` (model `~/work/Strata/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`).
   - Old binary (6b7b03a, before the bump) reference, measured tonight: prefetch off 968.4 read / 51.1 write tokens/s, cache 6.25G; prefetch on 1527.4 / 53.9, cache 5.50G; same answer (b2941bfb).
   - New binary: not measured with the fit fix yet. The first runs (181 / 6.4) were the fit bug above.
   - Compare the checksum of the answers, the cache sizes, and what stderr says about `moe prefetch` and `llama_moe_cache`.
2. `cargo test` and `cargo clippy --no-deps -- -D warnings` in `engine/` (the `llama_patches` test applies 0001-0003 to the pin), and the CI build of the CUDA jobs.
3. Re-check the docs that describe 0002-0004 (`patches/README.md` table, `docs/moe-offload-plan.md` section 3 and 4, `docs/engine-guide.md`), the CHANGELOG, and `LLAMA_MOE_PREFETCH_FROM_CACHE` (gone with 0004).
4. Then, as a separate commit: re-vendor llama-cpp-rs 0.1.159 (LlamaVocab refactor; 29 call sites in `engine/src`; our additions in the vendored crate and its build.rs go back on top).
5. Move the pin to the release tag when it exists; the mirror (`eullm/llama.cpp`) was synced and already holds `b86d2f0`.

## Resuming
On the PC, in `~/work/eullm`: `claude --continue` reopens the last session with this conversation, or `claude --resume` to pick it. A new session can start from this file.

## Update (8 October, evening) — stopped for a storm
Branches, all pushed: `feat/llama-moe-cache` (bump to the merged MoE cache, PR), `feat/llama-cpp-rs-0.1.159` (re-vendor, CI green), `feat/prefetch-reads-moe-cache` (patch `0004` ported: 1,516 vs 1,387 tokens/s reading; stacked on the re-vendor branch).

Validated on the re-vendor build: Flash-Next prefetch 0 and 4, `--mtp 2` on Qwen3.5-9B (68% of drafts kept), bge-m3 embeddings (1024), chat on three endpoints, multimodal. `deepseek-math-7b-instruct.Q8_0` aborts the server at its first prompt (an uncaught `std::out_of_range` from `llama_vocab::byte_to_token`, through the C API); it does the same on release 0.7.6, so it is not from the bump.

Stock llama-server against the patched one (`bench/llama_server_compare.sh`, builds in `~/work/llama-compare/{stock,patched}`, same pin, patched = 0003+0004 + a `--moe-prefetch` flag in `common/arg.cpp`): first pass, one run per variant in two orders, was too noisy to settle `0004`. A second pass of six rotated rounds (`~/work/llama-compare/rep1..6`) was stopped after two rounds: stock 950 read; 0003 alone 1,395 and 1,408; 0003+0004 1,534 and 1,572 (+10%); writing 53-54 for both patched variants (the earlier dip was noise). Rounds 3-6 are still to run, with the desktop closed (Chrome, Thunderbird and gnome-shell were using the GPU). Before any PR to llama.cpp: those rounds, a second model, the flag and the code cleaned for upstream; the PR text goes to the user first, nothing is posted without them.

Resume: `cd ~/work/eullm && git checkout feat/prefetch-reads-moe-cache`; rerun `bench/llama_server_compare.sh` with `ORDER=...` as in `~/work/llama-compare/rep.sh`.

## Update (8 October, afternoon)
Six rounds of stock vs patched llama-server finished (table in `docs/moe-offload-plan.md`): reading 1,010.7 stock, 1,387.9 with `0003`, 1,544.8 with `0003`+`0004`; writing the same in all; the same answer in 18 of 18. A Radeon RX 6700 XT now drives the desktop (the RTX 5070 Ti idles at 15 MiB, 16 W); a 5-minute load before and after showed the same temperatures (75/76 C max, fans 48/49%). Linux Vulkan build 0.7.40 downloaded to `~/work/vulkan-test`, not yet run. Next: a long answer with and without the cache (a report on the merged PR says the cache loops on long texts), the Vulkan test on the 6700 XT, then the PR text for the prefetch (the PR author removed the scheduler changes from #29887: expect that to be the hard part).

## Update (8 October, evening) — Vulkan on the RX 6700 XT
Release 0.7.40 Linux Vulkan build (`~/work/vulkan-test`, `GGML_VK_VISIBLE_DEVICES=0` picks the Radeon, RADV, no matrix cores), Qwen3-14B Q4_K_M (9.3 GB, all layers on the card), a 3,000-token prompt:
- flash attention on (the engine's default, `auto-detect`): 12.5 tokens/s writing, **29 reading**;
- `--no-flash-attn`: 18.6 writing, **345 reading**.
Qwen3.5-9B Q4_K_M with flash attention on: 5.0 / 38. So on RDNA 2 under Vulkan the flash-attention path costs most of the prompt-reading speed. To decide: turn flash attention off by default on Vulkan where there are no cooperative matrices (needs a check on a newer llama.cpp than 0.7.40's, and on another Vulkan card), or at least say it in `docs/platforms.md`.
Also found: the board's second long slot (the chipset's) negotiates PCIe 3.0 x2 here, not x4; the card in it measures the same as in x16 once the model is loaded. Long-text check of the MoE cache (reports of looping on the merged PR): 4,000 tokens, temperature 0, with and without the cache on Flash-Next IQ2_XS, neither loops; they diverge after 230 characters at a near tie, as greedy runs do when the numerics move.

## Update (9 October) — flash attention on and off, Vulkan on the RX 6700 XT
EuLLM 0.7.40 Vulkan build, default context (4,096), a 3,000-token prompt, two rounds in opposite orders, greedy answers (each model's answer is the same in both rounds and differs between on and off, as numerics do). Qwen3-14B Q4_K_M with `--no-fit --gpu-layers 99` (the automatic fit left 3 of 40 layers in RAM with the desktop on the other card); Gemma-4 E4B Q4_K_M; Qwen3.6-35B-A3B Q4_K_M with `--cpu-moe`.

| model | flash attention | read tokens/s (round 1, round 2) | write tokens/s |
|---|---|---|---|
| Qwen3-14B | on | 43.7, 42.0 | 20.7, 20.4 |
| Qwen3-14B | off | 246.4, 470.0 | 33.2, 32.8 |
| Gemma-4 E4B | on | 64.5, 74.7 | 22.7, 24.4 |
| Gemma-4 E4B | off | 43.0, 43.9 | 18.4, 20.4 |
| Qwen3.6-35B-A3B, experts in RAM | on | 43.3, 42.5 | 9.3, 9.1 |
| Qwen3.6-35B-A3B, experts in RAM | off | 25.8, 42.9 | 9.3, 9.1 |

- Switching it off helps the dense Qwen3-14B a lot (reading 6 to 11 times, writing +60%), hurts Gemma-4 E4B (reading about 35% slower) and changes nothing for the MoE with its experts in RAM (round 2; round 1 of "off" was lower for no reason found).
- So a blanket default of "off on Vulkan without cooperative matrices" would cost Gemma 35%: the choice depends on the model. Not made; to say in `docs/platforms.md` for RDNA 2.
- The "off" readings of Qwen3-14B and of the MoE moved a lot between rounds (246 to 470; 26 to 43), unexplained. One card, one build, 3,000 tokens: an order of magnitude, not a figure.
- A first attempt with `--ctx-size 8192` was discarded: the fit put 35 of 40 layers on the card.
