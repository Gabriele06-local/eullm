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
# Where llama.cpp's sources are, in order of decreasing certainty. Building
# needs them — unlike the smoke test, which needs only a published binary,
# because the diffusion loop lives in examples/ and is never compiled into the
# library we link. What it does not need is our repository, or four months of
# llama.cpp history.
#
# The middle case is why this is not a one-liner. Run standalone from a
# directory that is not a clone, EULLM_REPO resolves to whatever sits two
# levels up — `/users`, on a LUMI login node — and the script then reports a
# missing submodule in a repository that was never there. So look beside the
# script first: that is where the standalone recipe puts llama.cpp.
if [ -n "${LCPP_DIR:-}" ]; then
    LCPP="$LCPP_DIR"
elif [ -f "$PWD/llama.cpp/CMakeLists.txt" ]; then
    LCPP="$PWD/llama.cpp"
    log_lcpp_found_beside=1
else
    LCPP="$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
fi
BUILD_DIR="${BUILD_DIR:-$LCPP/build-hip}"

# The commit EULLM ships. Kept here so the standalone route above can name it
# without a clone; when the submodule bumps, this bumps with it.
PINNED_COMMIT="${PINNED_COMMIT:-4d9176092d00586775af140581bb0b558ddc4389}"

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
if [ ! -f "$LCPP/CMakeLists.txt" ]; then
    MSG="no llama.cpp sources at $LCPP"
    # Only offer the submodule route when there is actually a repository to run
    # it in. Suggesting `git -C /users submodule update` helps nobody.
    if [ -f "$EULLM_REPO/engine/Cargo.toml" ]; then
        MSG="$MSG
  From this clone:
    git -C $EULLM_REPO submodule update --init --depth 1 \\
        engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
    fi
    err "$MSG
  Standalone, without the repository (one commit, no history):
    git init llama.cpp && git -C llama.cpp fetch --depth 1 \\
        https://github.com/eullm/llama.cpp $PINNED_COMMIT
    git -C llama.cpp checkout FETCH_HEAD
    bash $(basename "${BASH_SOURCE[0]}")        # found automatically from here
  Or point at a checkout elsewhere:
    export LCPP_DIR=/path/to/llama.cpp"
fi
[ -n "${log_lcpp_found_beside:-}" ] && log "llama.cpp found beside this script: $LCPP"
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

# Look for device code wherever ggml-hip actually ended up. llama.cpp builds
# shared libraries by default, and then the executable carries none of it —
# the kernels live in libggml-hip.so. Checking only the executable reports a
# perfectly good build as broken, which is what this check did on its first
# real run.
DEVICE_CODE_IN=""
for obj in "$BIN" "$BUILD_DIR"/bin/libggml*.so "$BUILD_DIR"/lib/libggml*.so; do
    [ -f "$obj" ] || continue
    if strings -a "$obj" 2>/dev/null | grep -q "amdhsa--$EULLM_AMDGPU_TARGETS"; then
        DEVICE_CODE_IN="$obj"
        break
    fi
done
[ -n "$DEVICE_CODE_IN" ] || err "no $EULLM_AMDGPU_TARGETS device code in $BIN or any libggml*.so
    beside it — the build would fall back to the CPU. Objects searched:
$(ls -1 "$BIN" "$BUILD_DIR"/bin/libggml*.so "$BUILD_DIR"/lib/libggml*.so 2>/dev/null | sed 's/^/      /')"
ok "$EULLM_AMDGPU_TARGETS device code in $(basename "$DEVICE_CODE_IN")"

ok "llama-diffusion-cli: $BIN"
ok "llama-tokenize:      $BUILD_DIR/bin/llama-tokenize"
printf '\nNext:\n  export DIFFUSION_BIN=%s\n  export TOKENIZE_BIN=%s\n  sbatch tools/lumi/sbatch_diffusion.slurm\n' \
    "$BIN" "$BUILD_DIR/bin/llama-tokenize"
