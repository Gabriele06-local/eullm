#!/usr/bin/env bash
# Build llama.cpp's diffusion CLI with the ROCm/HIP backend on a LUMI-G login
# node, from the llama.cpp we actually vendor.
#
#     bash tools/lumi/build_diffusion_cli.sh
#
# WHY THIS EXISTS, and why it is not the engine build. `tools/lumi/build_engine.sh`
# builds EULLM, which links llama.cpp as a static library — the examples are
# never compiled. Diffusion generation is NOT in that library: the whole loop
# lives in `examples/diffusion/diffusion.cpp` (408 lines) and is not exported
# from `llama.h`. So there is nothing in the engine, today, that can run a
# diffusion model, and nothing to measure without building the example.
#
# Building it from the submodule rather than from llama.cpp upstream is the
# point: it measures the exact commit EULLM ships, so a number obtained here is
# a statement about our engine's dependency, not about someone else's tree.
#
# Two binaries come out:
#   llama-diffusion-cli   the thing under test
#   llama-tokenize        needed to count the prompt honestly — see the note on
#                         generated_tokens in sbatch_diffusion.slurm
#
# Overridable:
#   ROCM_PATH             where ROCm lives (default: /opt/rocm)
#   EULLM_AMDGPU_TARGETS  GPU architecture (default: gfx90a = MI250X)
#   EULLM_REPO            repository root (default: inferred from this script)
#   BUILD_DIR             build directory (default: <submodule>/build-hip)

set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
EULLM_AMDGPU_TARGETS="${EULLM_AMDGPU_TARGETS:-gfx90a}"
EULLM_REPO="${EULLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LCPP="$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
BUILD_DIR="${BUILD_DIR:-$LCPP/build-hip}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# ── Preflight ─────────────────────────────────────────────────────────────

[ -d "$ROCM_PATH/lib" ] || err "no ROCm at $ROCM_PATH — check 'module avail rocm' and set ROCM_PATH"
log "ROCm: $("$ROCM_PATH/bin/hipconfig" --version 2>/dev/null || echo unknown) at $ROCM_PATH"

command -v cmake >/dev/null || err "cmake not found — 'module load CMake' or equivalent"
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
printf '3.21\n%s\n' "$CMAKE_VER" | sort -V -C || err "cmake $CMAKE_VER is too old — the HIP language needs 3.21+"
log "cmake: $CMAKE_VER"

# The submodule is a gitlink: a plain `git clone` leaves this directory empty
# and the failure downstream is a confusing CMake error about a missing
# CMakeLists, not a missing checkout.
[ -f "$LCPP/CMakeLists.txt" ] || err "llama.cpp submodule is not checked out at $LCPP
    git -C $EULLM_REPO submodule update --init --depth 1 \\
        engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
log "llama.cpp: $(git -C "$LCPP" rev-parse --short HEAD 2>/dev/null || echo '?') (the commit EULLM ships)"

# The diffusion loop lives under examples/, which the default preset skips.
grep -q "LLM_ARCH_DREAM" "$LCPP/src/llama-arch.h" \
    || err "this llama.cpp has no diffusion architectures — wrong submodule commit?"
[ -f "$LCPP/examples/diffusion/diffusion-cli.cpp" ] \
    || err "no examples/diffusion in this llama.cpp — wrong submodule commit?"

# ── Configure ─────────────────────────────────────────────────────────────
# GPU_TARGETS and CMAKE_HIP_ARCHITECTURES are both named on purpose. Without
# them HIP resolves the architecture from the GPUs of the machine doing the
# compiling, and a login node has none — the same trap that cost us two failed
# CI runs before `build.rs` started naming the target explicitly.

log "configuring for $EULLM_AMDGPU_TARGETS into $BUILD_DIR"
cmake -S "$LCPP" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="$EULLM_AMDGPU_TARGETS" \
    -DCMAKE_HIP_ARCHITECTURES="$EULLM_AMDGPU_TARGETS" \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    >/dev/null || err "cmake configure failed — rerun without >/dev/null to see why"

log "building (this is the long part; run under tmux)"
cmake --build "$BUILD_DIR" --target llama-diffusion-cli llama-tokenize -j "$(nproc)"

# ── Verify ────────────────────────────────────────────────────────────────
# The banner names the backend compiled in, not the device found at runtime, so
# check the object code instead: if gfx90a device code is absent the binary will
# load and silently run on the CPU.

BIN="$BUILD_DIR/bin/llama-diffusion-cli"
[ -x "$BIN" ] || err "expected $BIN, not found"

LDD_OUT=$(ldd "$BIN" 2>&1 || true)
grep -q "amdhip64" <<<"$LDD_OUT" || err "$BIN is not linked against HIP — the ROCm backend did not build in"

STRINGS_OUT=$(strings -a "$BIN" 2>/dev/null || true)
grep -q "amdhsa--$EULLM_AMDGPU_TARGETS" <<<"$STRINGS_OUT" \
    || err "no $EULLM_AMDGPU_TARGETS device code in $BIN — it would fall back to the CPU"

ok "llama-diffusion-cli: $BIN"
ok "llama-tokenize:      $BUILD_DIR/bin/llama-tokenize"
printf '\nNext:\n  export DIFFUSION_BIN=%s\n  export TOKENIZE_BIN=%s\n  sbatch tools/lumi/sbatch_diffusion.slurm\n' \
    "$BIN" "$BUILD_DIR/bin/llama-tokenize"
