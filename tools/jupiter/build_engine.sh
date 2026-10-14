#!/usr/bin/env bash
# Build the EULLM engine with CUDA for GH200 (aarch64 Grace + H100, sm_90) on
# a JUPITER login node, as tools/lumi/build_engine.sh does for the MI250X.
#
#     bash tools/jupiter/build_engine.sh [LABEL]
#
# LABEL names the build directory, $PROJECT/$USER/eullm-target/LABEL (default
# main): a running job keeps the binary it started with, so a new engine gets
# a directory of its own (next10, next11, ...). Under tmux: it takes a while.
#
# Why build here: the published eullm-linux-arm64-cuda carries sm_86/89/120,
# the consumer cards, and not Hopper. A binary without device code for the
# GPU it runs on starts, announces CUDA and runs on the CPU (the A100 lesson,
# docs/cineca/leonardo.md), so the last step checks what the binary contains.

set -euo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

LABEL="${1:-main}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$EULLM_TARGETS/$LABEL}"
CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES:-90}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

# ── Preflight: say what is missing before an hour of build does ──────────
[ "$(uname -m)" = aarch64 ] || log "not aarch64 ($(uname -m)): the binary will not run on the GH200 nodes"
[ -n "${CUDA_PATH:-}" ] || err "no nvcc: 'module load CUDA' (the probe listed CUDA/13), or set CUDA_PATH"
export CUDACXX="$CUDA_PATH/bin/nvcc" CMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
export PATH="$CUDA_PATH/bin:$PATH"
log "CUDA: $("$CUDACXX" --version | tail -1) at $CUDA_PATH"
# cudart and cuBLAS are linked statically. llama-cpp-sys-2's build.rs looks
# for them under $CUDA_PATH's lib64, lib and targets/sbsa-linux/lib: say
# here, not after the build, if this toolkit has none.
CUDART_STATIC=""
for d in "$CUDA_PATH/lib64" "$CUDA_PATH/lib" "$CUDA_PATH/targets/sbsa-linux/lib"; do
    if [ -f "$d/libcudart_static.a" ]; then
        CUDART_STATIC="$d/libcudart_static.a"
        break
    fi
done
[ -n "$CUDART_STATIC" ] \
    || err "no libcudart_static.a under $CUDA_PATH (lib64, lib, targets/sbsa-linux/lib): is CUDA_PATH the toolkit's root?"
log "static cudart: $CUDART_STATIC"

command -v gcc >/dev/null && command -v g++ >/dev/null || err "no gcc/g++: 'module load GCC'"
GCC_MAJOR=$(gcc -dumpversion | cut -d. -f1)
[ "$GCC_MAJOR" -ge 9 ] || err "gcc $GCC_MAJOR is too old for llama.cpp (9+): 'module load GCC'"
export CC=gcc CXX=g++
log "gcc: $(gcc -dumpversion)"

if ! command -v cmake >/dev/null; then
    eullm_module load CMake || true
fi
command -v cmake >/dev/null || err "no cmake: 'module avail CMake' and load one (3.21+)"
log "cmake: $(cmake --version | head -1 | awk '{print $3}')"

# bindgen loads libclang. A site module first, then the system's.
if [ -z "${LIBCLANG_PATH:-}" ]; then
    for m in Clang LLVM; do
        eullm_module load "$m" && break
    done
    for d in "${EBROOTCLANG:-}/lib" "${EBROOTLLVM:-}/lib" /usr/lib64 /usr/lib/aarch64-linux-gnu /usr/lib64/llvm*/lib; do
        for f in "$d"/libclang.so* "$d"/libclang-*.so*; do
            if [ -e "$f" ]; then
                export LIBCLANG_PATH="$d"
                break 2
            fi
        done
    done
fi
[ -n "${LIBCLANG_PATH:-}" ] \
    || err "no libclang for bindgen: 'module avail Clang LLVM', load one, or set LIBCLANG_PATH"
log "libclang: $LIBCLANG_PATH"

# Rust from rustup, in the project: login nodes reach the internet.
if ! command -v cargo >/dev/null; then
    log "installing rustup into $CARGO_HOME (once)"
    mkdir -p "$RUSTUP_HOME" "$CARGO_HOME"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --no-modify-path --profile minimal
    export PATH="$CARGO_HOME/bin:$PATH"
fi
command -v cargo >/dev/null || err "cargo still not found after rustup"
log "cargo: $(cargo --version)"

[ -f "$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/CMakeLists.txt" ] \
    || err "the llama.cpp submodule is missing: git -C '$EULLM_REPO' submodule update --init --recursive"
LCPP_STATUS=$(git -C "$EULLM_REPO" submodule status engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp)
[[ "$LCPP_STATUS" != +* ]] \
    || err "llama.cpp is not at the commit this checkout wants: git -C '$EULLM_REPO' submodule update --init --recursive"

# ── Build ─────────────────────────────────────────────────────────────────
# CMAKE_CUDA_ARCHITECTURES is not optional: without it nvcc compiles for what
# it guesses, and the binary's device code is the question checked below.
#
# -no-pie, on the final binary only: rustc links it position-independent,
# and the host code nvcc compiles from ggml-cuda's .cu files is not. Ubuntu's
# gcc makes position-independent code by default, which is why the release
# job for arm64 (the nvidia/cuda Ubuntu image) links without it; the GCC
# module here does not, and the first build on 14-10-2026 stopped at
# "relocation R_AARCH64_ADR_PREL_PG_HI21 ... recompile with -fPIC" in
# ggml-cuda.cu.o. `cargo rustc` scopes the flag to
# the eullm binary (proc-macros are shared objects and must stay PIC), as the
# release job for x64 CUDA and Leonardo do (docs/cineca/leonardo.md, rows 5-6).
log "building for sm_$CUDA_ARCHS into $CARGO_TARGET_DIR (tmux recommended)"
cd "$EULLM_REPO"
CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    cargo rustc --release --features "cuda,multimodal" -p eullm-engine --bin eullm \
    -- -C link-arg=-no-pie

BIN="$CARGO_TARGET_DIR/release/eullm"
[ -x "$BIN" ] || err "the build reported success but $BIN is not there"

# ── What the binary contains ──────────────────────────────────────────────
# Matched against variables, not `| grep -q`: under pipefail grep's early
# exit fails a good binary with SIGPIPE (see tools/lumi/build_engine.sh).
LINKED=$(ldd "$BIN" 2>&1 || true)
grep -q 'libcuda\.so' <<<"$LINKED" || err "$BIN does not link libcuda: not a CUDA build"
ok "links the CUDA driver"
if command -v cuobjdump >/dev/null; then
    ELF=$(cuobjdump --list-elf "$BIN" 2>/dev/null || true)
    grep -q "sm_$CUDA_ARCHS" <<<"$ELF" || err "$BIN has no device code for sm_$CUDA_ARCHS"
    ok "device code present for sm_$CUDA_ARCHS"
fi
"$BIN" --version
ok "binary: $BIN"
echo
echo "Next, on this login node: llama-server at the same llama.cpp commit, then the"
echo "models, the workload sets and the plan (compute nodes may have no network):"
echo
echo "    bash tools/jupiter/install_runtimes.sh"
echo "    EULLM_BIN=$BIN bash tools/jupiter/setup.sh $LABEL"
