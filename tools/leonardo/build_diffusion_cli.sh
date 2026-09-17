#!/usr/bin/env bash
# Build llama.cpp's diffusion CLI with the CUDA backend on a Leonardo login
# node, from the llama.cpp we actually vendor.
#
#     bash tools/leonardo/build_diffusion_cli.sh
#
# The CUDA twin of tools/lumi/build_diffusion_cli.sh. Kept as a separate file
# rather than one script with a backend switch because the two sites differ in
# more than a CMake flag — modules, compute capability, and the failure modes
# below are Leonardo's own, and a script you can copy onto a machine on its own
# is worth more here than one that shares code.
#
# WHY THIS EXISTS. Diffusion generation is not in libllama: the whole loop is
# 408 lines in examples/diffusion, not exported from llama.h, and the engine
# build never compiles the examples. Nothing we ship today can run a diffusion
# model, so there is nothing to measure without this.
#
# THE THREE TRAPS THIS SITE HAS, all of them documented in docs/cineca/leonardo.md
# because each one cost someone an afternoon:
#
#   1. sm_80, and it is not covered by the release binaries. The A100 is
#      data-center Ampere — a different compute capability from the sm_86 of
#      consumer Ampere, and OLDER than the lowest architecture the normal CUDA
#      artifact is built for, so PTX forward-compatibility does not save it.
#      This builds for 80 explicitly.
#   2. No CUDA 13. The newest toolkit module here is 12.6; a binary built
#      against CUDA 13.1 needs driver r580 and Leonardo does not have it. Use
#      the module, never a self-installed newer toolkit.
#   3. The silent CPU fallback. When the driver is too old the binary starts,
#      prints "GPU backend: CUDA", and runs entirely on the CPU. The only sign
#      is one line: "ggml_cuda_init: failed to initialize CUDA". The verify
#      step below looks for device code, and sbatch_diffusion.slurm greps every
#      run for that line, because a benchmark that silently measures the CPU is
#      worse than one that fails.
#
# Overridable:
#   CUDA_ARCH      compute capability (default: 80 = A100)
#   EULLM_REPO     repository root (default: inferred from this script)
#   BUILD_DIR      build directory (default: <submodule>/build-cuda)

set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-80}"
EULLM_REPO="${EULLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# LCPP_DIR lets this run without the EULLM repository at all: fetch llama.cpp
# alone at the pinned commit and point here. Building needs the C++ sources —
# unlike the smoke test, which needs only a published binary — but it does not
# need our repository or four months of llama.cpp history.
LCPP="${LCPP_DIR:-$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LCPP/build-cuda}"

# The commit EULLM ships. Kept here so the standalone route above can name it
# without a clone; when the submodule bumps, this bumps with it.
PINNED_COMMIT="${PINNED_COMMIT:-4d9176092d00586775af140581bb0b558ddc4389}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# ── Preflight ─────────────────────────────────────────────────────────────

command -v nvcc >/dev/null || err "nvcc not found — 'module load cuda/12.2' (12.2, 12.3 or 12.6; there is no CUDA 13 here)"
NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
case "$NVCC_VER" in
    13.*) err "CUDA $NVCC_VER needs driver r580, which Leonardo does not have — load 12.2/12.3/12.6 instead" ;;
esac
log "nvcc: $NVCC_VER"

command -v cmake >/dev/null || err "cmake not found — 'module load cmake' or equivalent"
log "cmake: $(cmake --version | head -1 | awk '{print $3}')"
log "host compiler: $(${CXX:-g++} --version | head -1)  (gcc/12.2.0 is the module that works here)"

[ -f "$LCPP/CMakeLists.txt" ] || err "no llama.cpp sources at $LCPP
  From a clone of this repository:
    git -C $EULLM_REPO submodule update --init --depth 1 \\
        engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp
  Or standalone, without the repository (one commit, no history):
    git init llama.cpp && git -C llama.cpp fetch --depth 1 \\
        https://github.com/eullm/llama.cpp $PINNED_COMMIT
    git -C llama.cpp checkout FETCH_HEAD
    export LCPP_DIR=\$PWD/llama.cpp"
log "llama.cpp: $(git -C "$LCPP" rev-parse --short HEAD 2>/dev/null || echo '?') (the commit EULLM ships)"

grep -q "LLM_ARCH_DREAM" "$LCPP/src/llama-arch.h" \
    || err "this llama.cpp has no diffusion architectures — wrong submodule commit?"
[ -f "$LCPP/examples/diffusion/diffusion-cli.cpp" ] \
    || err "no examples/diffusion in this llama.cpp — wrong submodule commit?"

# ── Configure ─────────────────────────────────────────────────────────────

log "configuring for sm_$CUDA_ARCH into $BUILD_DIR"
cmake -S "$LCPP" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    >/dev/null || err "cmake configure failed — rerun without >/dev/null to see why"

log "building (the long part; run under tmux — a dropped SSH kills it otherwise)"
cmake --build "$BUILD_DIR" --target llama-diffusion-cli llama-tokenize -j "$(nproc)"

# ── Verify ────────────────────────────────────────────────────────────────

BIN="$BUILD_DIR/bin/llama-diffusion-cli"
[ -x "$BIN" ] || err "expected $BIN, not found"

LDD_OUT=$(ldd "$BIN" 2>&1 || true)
grep -qE "libcudart|libcuda" <<<"$LDD_OUT" || err "$BIN is not linked against CUDA — the backend did not build in"

# cuobjdump is the honest check: the banner names the backend compiled in, not
# the device found, so a binary with no sm_80 code still says "GPU backend: CUDA".
if command -v cuobjdump >/dev/null; then
    ARCH_OUT=$(cuobjdump -lelf "$BIN" 2>/dev/null || true)
    grep -q "sm_$CUDA_ARCH" <<<"$ARCH_OUT" \
        || err "no sm_$CUDA_ARCH device code in $BIN — it would fall back to the CPU on an A100"
    ok "sm_$CUDA_ARCH device code present"
else
    log "cuobjdump not on PATH — skipping the device-code check (the job script still greps for the runtime fallback)"
fi

ok "llama-diffusion-cli: $BIN"
ok "llama-tokenize:      $BUILD_DIR/bin/llama-tokenize"
printf '\nNext:\n  export DIFFUSION_BIN=%s\n  export TOKENIZE_BIN=%s\n  sbatch tools/leonardo/sbatch_diffusion.slurm\n' \
    "$BIN" "$BUILD_DIR/bin/llama-tokenize"
