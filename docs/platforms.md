# Platforms and downloads

Every prebuilt EuLLM binary, what it runs on and who has run it; the ARM and data-centre results; how to report a problem on hardware we cannot reach. The [README](../README.md) has the short version.

## Downloads

**All prebuilt binaries** — pick yours from the [latest release](https://github.com/eullm/eullm/releases/latest):

| Platform | File | Status | Notes |
|----------|------|:------:|-------|
| 🐧 Linux x64 (CPU) | `eullm-linux-x64` | ✅ Tested | Now built on a Rocky Linux 8 base (glibc 2.28) instead of Ubuntu 22.04, so it also runs on RHEL/CentOS/Rocky/Alma/Oracle Linux 8+ — found live on CINECA Leonardo (RHEL 8.7), where it hit the same glibc-symbol gap the CUDA build below already had fixed |
| 🐧 Linux x64 (NVIDIA) | `eullm-linux-x64-cuda-13.1` | ✅ Tested | RTX 3000/4000/5000. Now built on a Rocky Linux 8 base (glibc 2.28) instead of Ubuntu 22.04, so it also runs on RHEL/CentOS/Rocky/Alma/Oracle Linux 8+, not just current Ubuntu/Fedora/Windows — the RTX-card behavior above is what's actually tested, the older-distro compatibility follows from how glibc versioning works but hasn't had its own hardware report yet |
| 🐧 Linux x64 (NVIDIA data-center) | `eullm-linux-x64-cuda-12.4-datacenter` | ✅ Tested | A100 (sm_80), H100 (sm_90) — a different, older compute capability than the consumer build above, which doesn't run on these cards. **CUDA 12.4, not 13.1 like the consumer build**: minimum driver r550 instead of r580. That is deliberate — A100s live in HPC centres and enterprise fleets, which freeze driver versions for years. The first attempt shipped CUDA 13.1 and would not start on CINECA Leonardo at all (`CUDA driver version is insufficient for CUDA runtime version`, then a silent fall back to CPU); see `docs/cineca/leonardo.md`. The 12.4 rebuild was validated on CINECA Leonardo on 6 September 2026: CUDA initialises on an A100-SXM-64GB, every layer goes to the GPU, and a 27B Q8 model decodes at 32.4 tok/s on one GPU |
| 🐧 Linux ARM64 | `eullm-linux-arm64` | ✅ Tested (community) | Generic ARM64 baseline — the one to take unless you are certain otherwise. Validated on Raspberry Pi 400; RPi 4/5, Orange Pi 5+, Jetson, etc. |
| 🐧 Linux ARM64 (CIX P1 tuned) | `eullm-linux-arm64-cix-p1` | ✅ Tested | A CPU profile built for the CIX P1 (Radxa Orion O6), with Armv9.2-A SVE2, BF16, I8MM and DotProd compiled in. **Those instructions are baked into the binary, so it dies with SIGILL on any ARM64 host that lacks them** — a Raspberry Pi among them. It is a separate download precisely so the generic build above stays safe everywhere; see `docs/arm-cix-p1-cpu-profile.md` |
| 🐧 Linux ARM64 (NVIDIA) | `eullm-linux-arm64-cuda-13.1` | ✅ Tested | ARM host + discrete NVIDIA GPU (sm_86/89/120); validated on a Radxa Orion O6 (CIX P1) with an RTX 3060 12GB in its PCIe slot, qwen3-14b Q4 at 33 tok/s — the same board does 3.0 tok/s on the same model CPU-only |
| 🍎 macOS Apple Silicon (Metal) | `eullm-macos-arm64` | ✅ Tested (community) | Validated on M2 Pro (Metal); M1/M2/M3/M4 |
| 🍎 macOS Intel | `eullm-macos-x64` | ✅ Tested (community) | Validated on a 2018 Mac mini (i7-8700B, 54 tok/s) and a 2018 MacBook Pro 15" (i9-8950HK, 41 tok/s), qwen3-0.6b Q4. CPU only: Metal is deliberately not built for Intel Macs, see below |
| 🪟 Windows 11 x64 (CPU) | `eullm-windows-x64.zip` | 🆕 New | eullm.exe + the Visual C++ runtime DLLs — extract, run. Runs on a clean Windows |
| 🪟 Windows 11 x64 (CPU) | `eullm-windows-x64.exe` | ✅ Tested | Standalone binary, CLI/server. Needs the [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) already installed, otherwise it stops with *"MSVCP140.dll was not found"* — take the ZIP above then |
| 🪟 Windows 11 x64 (NVIDIA) | `eullm-windows-x64-cuda-13.1.zip` | ✅ Tested | ZIP bundles the CUDA and Visual C++ runtime DLLs — extract, run |
| 🪟 Windows 11 x64 (Vulkan, AMD/Intel) | `eullm-windows-x64-vulkan.zip` | 🆕 Untested | Any GPU with a Vulkan driver: AMD Radeon cards and Ryzen integrated graphics, Intel Arc, and NVIDIA cards the CUDA build does not cover (GTX 1000, RTX 2000, or a driver older than 580). eullm.exe + the Visual C++ runtime DLLs — extract, run. Nothing GPU-side is bundled: the Vulkan loader (`vulkan-1.dll`) comes with the graphics driver. `install.ps1` picks it for these GPUs |
| 🪟 Windows 11 x64 (AMD Radeon, ROCm) | `eullm-windows-x64-rocm.zip` | 🧪 Experimental, untested | RX 7900 XTX/XT (gfx1100), RX 9070 XT/9070 and Radeon AI PRO R9700 (gfx1201), through ROCm 10.1. Needs the AMD Software: Adrenalin Edition for ROCm 10.1 driver (26.10.41.05) or newer. Bundles the HIP runtime, rocBLAS and hipBLASLt with the kernels of those cards, and the Visual C++ runtime, in ROCm's own layout: `bin\eullm.exe`, with `.kpack\` beside `bin\`. Not installed by `install.ps1` or `eullm update` yet: extract the ZIP. For other Radeons, and Instinct MI50/MI60 (gfx906, which ROCm on Windows does not support), take the Vulkan build |
| 🐧 Linux x64 (Vulkan, AMD/Intel) | `eullm-linux-x64-vulkan` | ✅ Tested (community) | Any GPU with a Vulkan driver — AMD, Intel, and NVIDIA alike. Needs `libvulkan.so.1` and a driver on the machine (mesa RADV, amdvlk, NVIDIA, Intel ANV); nothing is bundled. Validated on a Ryzen AI 9 HX 470 with Radeon 890M integrated graphics (96 GB unified memory, openSUSE Tumbleweed): all 29 layers offloaded, qwen3-0.6b Q4 at ~135 tok/s |
| 🐧 Linux x64 (AMD Radeon, ROCm) | `eullm-linux-x64-rocm-consumer` | 🆕 Untested | RDNA 3 (RX 7900/7800/7700/7600), RDNA 3.5 (Ryzen AI integrated Radeons), RDNA 4 (RX 9000). Needs ROCm 7.2.x on the machine: nothing is bundled, the binary links the system's own HIP and rocBLAS libraries. RX 6000 (RDNA 2) is not included: take the Vulkan build there |
| 🐧 Linux x64 (AMD Instinct, ROCm) | `eullm-linux-x64-rocm-gfx90a` | ✅ Tested | MI250X / MI210 (CDNA 2), for ROCm 6.3.x, the version EuroHPC sites run; check `hipconfig --version` first. Validated on LUMI-G on 12 September 2026: qwen3-8b Q4_K_M at 40.7 tok/s on one GCD of an MI250X, see `docs/lumi/lumi-g.md` |

> **Embedded chat UI — cross-platform.** Every `eullm` binary (Linux, macOS, Windows — CPU, CUDA, Metal) ships with a built-in browser chat. Run `eullm run model.gguf` and open **`http://localhost:11435/`** — same OpenAI/Ollama API on `:11434`, separate chat UI port `:11435` so it never collides with RAG / OpenAI-client routes on `/`. Turn it off with `--no-ui` for headless deployments.
>
> **Interactive picker.** Run `eullm` with no arguments (or `eullm run` with no model) and you get an interactive menu listing your locally installed GGUFs and the [EuLLM model catalog](../catalog/v1/catalog.json) — pick one, the engine takes care of download + launch.
>
> **SmartScreen note (Windows):** the binaries are not yet code-signed, so first launch may show *"Windows protected your PC"*. Click **More info → Run anyway**. CUDA bundles ship the required CUDA DLLs alongside — no separate CUDA toolkit install needed (an up-to-date NVIDIA driver is enough).
>
> **One-click installer paused.** v0.5.6 shipped an Inno Setup `.exe` installer; we pulled it from v0.5.8 onwards because the SmartScreen warning, the launcher script edge cases, and the install-time PATH handling all need a redesign before re-shipping. The standalone binaries above, or `install.ps1`, are the supported Windows distribution. SmartScreen only checks files carrying the "downloaded from the internet" mark; `install.ps1` verifies the checksum and then clears that mark (`Unblock-File`), so the *"Windows protected your PC"* prompt should not appear on first launch.

## Validation and testers

The Linux x64, Windows x64, and Linux ARM64 (CUDA) binaries are validated end-to-end by the maintainer. **macOS Apple Silicon (Metal)**, **Linux ARM64 (CPU)** and, as of v0.6.39, **macOS Intel (x64)** are **community-validated** (see the testers below). Every published binary has been run on real hardware by someone, except the two marked 🆕 Untested, `eullm-linux-x64-rocm-consumer` and `eullm-windows-x64-vulkan.zip`, and the experimental `eullm-windows-x64-rocm.zip`, which has only been built and started on a machine without a Radeon.

> **Intel Macs run on CPU, on purpose.** `eullm-macos-x64` ships without the Metal backend, because Metal produces wrong output on the GPUs those machines carry (Intel UHD 630, AMD Radeon Pro) — a known llama.cpp limitation ([#19563](https://github.com/ggml-org/llama.cpp/issues/19563), [#4004](https://github.com/ggml-org/llama.cpp/issues/4004)), which is why llama.cpp ships its own macOS x64 build the same way. Expect roughly 40-55 tok/s on a 0.6B Q4 model on 2018-era hardware. **If you are on a version before v0.6.39, upgrade**: those builds tried to use the GPU and returned garbage on Intel Macs.

> **How long Intel Macs get a build.** `eullm-macos-x64` is built on GitHub's `macos-15-intel` runner, the last x86_64 macOS image GitHub offers, supported until August 2027; macOS 26 Tahoe is also the last macOS release for Intel Macs. The build stays at least until that runner goes, or until nobody downloads it: across the 29 releases from v0.6.80-rc11 to v0.7.20 it was downloaded 28 times, as often as the Apple Silicon build (31).

If you run local LLMs on a Mac or an ARM64 board (Raspberry Pi 4/5, Orange Pi 5+, Rock 5B, Jetson, …), **your help validating these binaries is hugely appreciated**. See the open testing call:

→ **[Issue #140 — Help wanted: testing on macOS & ARM64 Linux](https://github.com/eullm/eullm/issues/140)** (`help wanted`, `testing`)

The remaining gap is **macOS Intel (x86_64)** — if you run local LLMs on a pre-Apple-Silicon Mac, reports with `eullm --version` output, model used, and what worked/broke are very welcome.

**Community testers — thank you 🙏** Early hands-on reports are already in (see #140):

- **[@PeterHickman](https://github.com/PeterHickman)** — five machines (2018 Mac mini and 2018 MacBook Pro on Intel, Mac mini M1 and M4 Pro, Raspberry Pi 5), re-tested on every release across a week of near-daily ones, with full logs every time. His reports are the reason macOS Intel works at all: they exposed that the published binary was loading all 29 layers onto the machine's GPU through Metal while printing "all inference will run on CPU", and the run he did of llama.cpp's own `llama-cli` on the same model and the same box is what proved the fault was ours rather than his hardware. Two further defects came out of the same thread: the CPU-only binaries were built without an AVX2 baseline, and `eullm serve` was starting with different KV cache defaults than `eullm run`.

- **[@odlg](https://github.com/odlg)** — first report on **Vulkan**, on an AMD Ryzen AI 9 HX 470 with Radeon 890M integrated graphics and 96 GB of unified memory, on openSUSE Tumbleweed: a combination nobody had covered on any axis. His full startup log paid for itself twice over: it showed our KV cache estimate reporting half the memory llama.cpp actually allocated (we derived the head dimension instead of reading the one the model declares), and his editor plugin running out of room against the 4096 default is why the banner now says when the context window is far below what the model can hold. He also found that building from source had been broken for three weeks for anyone following the README, which told people to clone without `--recursive`.

- **[@andreyluiz](https://github.com/andreyluiz)** — macOS (Apple M2 Pro, Metal) and Raspberry Pi 400 (Cortex-A72, ARM64), with full logs. His macOS run surfaced a real packaging bug: the published `eullm-macos-arm64` had been built without Metal and silently ran on CPU — the release now builds the macOS binaries with `--features metal`. Genuinely grateful for the time he's putting into validating hardware the maintainer can't reach.

### Diagnosing garbage output or crashes (`--rust-debug`, new in v0.6.34)

If you're helping test odd hardware and see garbage output (one token repeated over and over, gibberish) or an unexplained crash, add `--rust-debug` to `eullm run` / `eullm serve`. It turns on a NaN/Inf scan of the model's logits right before every sampled token and logs loudly (`tracing::error!`) if it finds corruption — the clearest signal available today for "is this a numerical bug in the compute path" versus something else further downstream.

```bash
eullm run ./model.gguf --rust-debug
eullm serve --rust-debug
```

Off by default: the scan touches every value in the vocabulary (~100-150k floats) on every generated token, so it's real added cost most users shouldn't pay for. Turn it on only when actively chasing a bug like this.

## ARM

### A 35B-parameter model on an ARM board, no GPU

A 35B-parameter hybrid MoE model (`qwen3.6-35b-a3b`, ~3B active params/token)
running entirely on CPU on a **Radxa Orion O6 (CIX P1 SoC, Armv9.2-A,
12-core, ~20W board power)** — no GPU, no NPU, consumer-grade EU-available
hardware:

- **~9-11 tok/s decode**, sustained across real multi-turn conversations
  and multiple topics
- **Multi-turn KV-cache reuse confirmed at 100% exact match** — every turn
  reuses the *entire* prior turn's resident state, verified across 6+
  consecutive turns at both 4096 and 16384-token context, with both F16 and
  Q8_0 KV cache — not a lucky first turn, a sustained, reproducible result

A large, capable open model, genuinely running on sovereign, GPU-free,
low-power EU hardware — not a toy demo on a small distilled model.

The same board with an RTX 3060 12GB in its PCIe slot runs qwen3-14b Q4 at 33 tok/s with `eullm-linux-arm64-cuda-13.1`, against 3.0 tok/s CPU-only. The CPU profile tuned for the CIX P1 is described in [`arm-cix-p1-cpu-profile.md`](arm-cix-p1-cpu-profile.md).

### On ARM CPU: a smaller quantized file can be *slower*, not faster

Counter-intuitive finding from real-hardware testing, worth documenting
because "pick the smaller file" is the natural instinct and is wrong here.
ggml's ARM online-repack fast-matmul path (i8mm/dotprod-accelerated) is
registered for specific tensor types only — confirmed by reading
`ggml-cpu/repack.cpp` directly: `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`,
`MXFP4`, `Q8_0`. Comparing two real quantizations of the same 35B model
side by side (clean, isolated logs — `- type ... tensors` + `file size`
lines from the loader, no cross-run contamination):

| | Q4_K_M | "UD-IQ4_NL" (Unsloth Dynamic) |
|---|---|---|
| Shared tensors | 361 × f32, 251 × q8_0 | 361 × f32, 251 × q8_0 |
| Variable tensors | 80 × Q4_K + 37 × Q5_K + 4 × Q6_K | 37 × IQ4_NL + **80 × IQ3_S** + 4 × Q6_K |
| File size | 20.60 GiB (5.11 BPW) | 16.79 GiB (4.16 BPW) — smaller |
| ARM-accelerated tensors | **121 / 121** (100%) | 41 / 121 (34%) |

The smaller file's size comes from pushing the majority of its "variable"
tensors down to **IQ3_S** — a codebook-based format with **no registered
ARM repack kernel at all** (confirmed absent from `repack.cpp`; `Q5_K`, by
contrast, *is* registered and ARM-accelerated, gated the same way as
`Q4_K`/`Q6_K`). Result: the "smaller, IQ4_NL" file is measurably *slower*
in practice, because two-thirds of its variable tensors run the
unaccelerated generic path. **For CPU-only ARM deployments, prefer a
quantization where every non-shared tensor type is on the accelerated
list above over a smaller file that isn't** — file size and ARM decode
speed are not the same axis.

## Data-centre GPUs and EuroHPC

EuLLM runs where European research computing runs. On **CINECA Leonardo** (NVIDIA A100-SXM-64GB), the `eullm-linux-x64-cuda-12.4-datacenter` build puts every layer on the GPU and decodes a 27B Q8 model at 32.4 tok/s on one GPU; on **LUMI-G** (AMD MI250X), `eullm-linux-x64-rocm-gfx90a` runs qwen3-8b Q4_K_M at 40.7 tok/s on one GCD. How to build and run on each site, and the multi-GPU measurements: [`cineca/leonardo.md`](cineca/leonardo.md) and [`lumi/lumi-g.md`](lumi/lumi-g.md).
