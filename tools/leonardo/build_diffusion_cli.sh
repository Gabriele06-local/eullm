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
BUILD_DIR="${BUILD_DIR:-$LCPP/build-cuda}"

# The commit EULLM ships. Kept here so the standalone route above can name it
# without a clone; when the submodule bumps, this bumps with it.
PINNED_COMMIT="${PINNED_COMMIT:-4d9176092d00586775af140581bb0b558ddc4389}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# ── Preflight ─────────────────────────────────────────────────────────────

command -v nvcc >/dev/null || err "nvcc not found. Load both modules, not just CUDA:
    module load gcc/12.2.0
    module load cuda/12.2        # 12.2, 12.3 or 12.6 — there is no CUDA 13 here
    export CC=gcc CXX=g++"
NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
case "$NVCC_VER" in
    13.*) err "CUDA $NVCC_VER needs driver r580, which Leonardo does not have — load 12.2/12.3/12.6 instead" ;;
esac
log "nvcc: $NVCC_VER"

command -v cmake >/dev/null || err "cmake not found — 'module load cmake' or equivalent"
log "cmake: $(cmake --version | head -1 | awk '{print $3}')"
# The host compiler is checked, not just reported. RHEL 8's own gcc 8.5.0 is
# pre-GCC-9, where std::filesystem lives in a separate libstdc++fs; combined
# with RHEL's patching it produces ABI mismatches on internal classes rather
# than a clean missing-symbol error, halfway through a long build. That is
# blocker #7 in docs/cineca/leonardo.md and it cost an afternoon once already.
HOST_CXX="${CXX:-g++}"
command -v "$HOST_CXX" >/dev/null || err "no C++ compiler ($HOST_CXX) — module load gcc/12.2.0 && export CC=gcc CXX=g++"
HOST_CXX_VER=$("$HOST_CXX" -dumpfullversion -dumpversion 2>/dev/null | head -1)
case "${HOST_CXX_VER%%.*}" in
    ''|*[!0-9]*) log "host compiler: $HOST_CXX $HOST_CXX_VER (version not parsed — continuing)" ;;
    *) if [ "${HOST_CXX_VER%%.*}" -lt 9 ]; then
           err "host compiler is $HOST_CXX $HOST_CXX_VER — pre-GCC-9 puts std::filesystem in a
    separate libstdc++fs and links with ABI mismatches rather than a clean error:
        module load gcc/12.2.0
        export CC=gcc CXX=g++"
       fi
       log "host compiler: $HOST_CXX $HOST_CXX_VER" ;;
esac

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

grep -q "LLM_ARCH_DREAM" "$LCPP/src/llama-arch.h" \
    || err "this llama.cpp has no diffusion architectures — wrong submodule commit?"
[ -f "$LCPP/examples/diffusion/diffusion-cli.cpp" ] \
    || err "no examples/diffusion in this llama.cpp — wrong submodule commit?"

# libcuda.so.1 is the DRIVER library, not the toolkit's. It lives on compute
# nodes and not on login05, so linking anything against ggml-cuda here fails on
# CUDA Driver API symbols — cuMemCreate, cuDeviceGet and friends — with ld
# itself suggesting -rpath or -rpath-link. The toolkit ships a stub for exactly
# this case; the Rust engine build already does the same through RUSTFLAGS.
# See docs/cineca/leonardo.md.
#
# -rpath-link, never -rpath: the first is consulted only while linking, the
# second is recorded in the binary and would make it load the STUB at runtime,
# on a compute node, with a real GPU sitting there. Every CUDA call would fail
# against a library whose entire purpose is to define nothing.
# One more turn of the screw: ld resolves a dependency by FILE NAME, and the
# name recorded in libggml-cuda.so is libcuda.so.1 — while the toolkit's stubs
# directory contains only libcuda.so. Pointing -rpath-link at the stubs is
# therefore necessary and not sufficient: ld looks in the right place and finds
# nothing called what it is looking for, then reports every driver symbol as
# undefined. So build a small directory carrying the versioned name.
CUDA_STUBS="${CUDA_STUBS:-${CUDA_HOME:-}/targets/x86_64-linux/lib/stubs}"
if [ -f "$CUDA_STUBS/libcuda.so" ]; then
    STUB_LINK_DIR="$BUILD_DIR/.cuda-stub"
    mkdir -p "$STUB_LINK_DIR"
    ln -sf "$CUDA_STUBS/libcuda.so" "$STUB_LINK_DIR/libcuda.so"
    ln -sf "$CUDA_STUBS/libcuda.so" "$STUB_LINK_DIR/libcuda.so.1"
    log "CUDA driver stub: $CUDA_STUBS (linked as libcuda.so.1 in $STUB_LINK_DIR)"
    STUB_LDFLAGS="-L$STUB_LINK_DIR -Wl,-rpath-link,$STUB_LINK_DIR"
elif ldconfig -p 2>/dev/null | grep -q "libcuda\.so\.1"; then
    log "libcuda.so.1 present on this node — no stub needed"
    STUB_LDFLAGS=""
else
    err "no libcuda.so.1 on this node and no stub at $CUDA_STUBS
    Linking ggml-cuda needs one or the other. Point CUDA_STUBS at the toolkit's
    stubs directory, or set CUDA_HOME so it can be derived:
        find \${CUDA_HOME:-/usr/local/cuda} -name 'libcuda.so' -path '*stubs*'"
fi

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
    -DCMAKE_EXE_LINKER_FLAGS="$STUB_LDFLAGS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$STUB_LDFLAGS" \
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
    # Same trap as the HIP twin: with shared libraries the kernels are in
    # libggml-cuda.so and the executable carries none, so searching only the
    # executable calls a good build broken.
    DEVICE_CODE_IN=""
    for obj in "$BIN" "$BUILD_DIR"/bin/libggml*.so "$BUILD_DIR"/lib/libggml*.so; do
        [ -f "$obj" ] || continue
        if cuobjdump -lelf "$obj" 2>/dev/null | grep -q "sm_$CUDA_ARCH"; then
            DEVICE_CODE_IN="$obj"
            break
        fi
    done
    [ -n "$DEVICE_CODE_IN" ] || err "no sm_$CUDA_ARCH device code in $BIN or any libggml*.so
    beside it — it would fall back to the CPU on an A100. Objects searched:
$(ls -1 "$BIN" "$BUILD_DIR"/bin/libggml*.so "$BUILD_DIR"/lib/libggml*.so 2>/dev/null | sed 's/^/      /')"
    ok "sm_$CUDA_ARCH device code in $(basename "$DEVICE_CODE_IN")"
else
    log "cuobjdump not on PATH — skipping the device-code check (the job script still greps for the runtime fallback)"
fi

# The failure this must not be allowed to ship: a binary that links here and
# then, on a compute node, resolves libcuda.so.1 to the stub instead of the
# driver. It would start, claim CUDA, and fail every call.
if [ -n "${STUB_LDFLAGS:-}" ] && command -v readelf >/dev/null; then
    for obj in "$BIN" "$BUILD_DIR"/bin/libggml*.so; do
        [ -f "$obj" ] || continue
        if readelf -d "$obj" 2>/dev/null | grep -E "RUNPATH|RPATH" | grep -qE "stubs|\.cuda-stub"; then
            err "$(basename "$obj") records the stub directory in RUNPATH — at runtime it would
    load the stub rather than the driver, on a node where the real one is
    present. Reconfigure in a clean build directory with -rpath-link
    (link-time only), not -rpath."
        fi
    done
    ok "no stub directory baked into RUNPATH"
fi

ok "llama-diffusion-cli: $BIN"
ok "llama-tokenize:      $BUILD_DIR/bin/llama-tokenize"
printf '\nNext:\n  export DIFFUSION_BIN=%s\n  export TOKENIZE_BIN=%s\n  sbatch tools/leonardo/sbatch_diffusion.slurm\n' \
    "$BIN" "$BUILD_DIR/bin/llama-tokenize"
