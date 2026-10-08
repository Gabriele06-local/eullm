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
