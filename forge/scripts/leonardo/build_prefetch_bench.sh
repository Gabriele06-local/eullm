#!/bin/bash
# Build the two llama-server binaries sbatch_prefetch_bench.slurm compares, on a LOGIN node
# (compute nodes have no network, and a build there is billed): the llama.cpp commit the
# engine pins, once as it is and once with the engine's prefetch patches (0003, 0004) and a
# --moe-prefetch flag for llama-server (bench/llama-server-moe-prefetch-flag.patch).
#
#   bash forge/scripts/leonardo/build_prefetch_bench.sh
#
# Everything goes under $BENCH_ROOT (default: the project's fast scratch, /leonardo_scratch/fast/<project>/
# prefetch-bench, NOT $WORK, which is nearly full and holds the pipelines' checkpoints; scratch is purged,
# which suits a bench): src (the checkout), stock/ and patched/ (worktrees, each with
# build/bin/llama-server), models/, results/, modules.txt (what the binaries were built with).
# It builds for the A100 (sm_80) and loads the modules docs/cineca/leonardo.md found to
# work: gcc/12.2.0, because the system gcc 8.5 breaks std::filesystem, and cuda/12.2.
# Idempotent: a checkout and a binary that exist are kept.
#
# Settable: BENCH_REPO (the eullm checkout holding the patches, default $WORK/eullm-prefetch),
# PIN (the llama.cpp commit, default b86d2f0), LCPP_MODULES, JOBS (default 8), BENCH_NO_VMM.
set -euo pipefail

: "${WORK:?WORK is not set: run this on Leonardo}"
BENCH_REPO="${BENCH_REPO:-$WORK/eullm-prefetch}"
PIN="${PIN:-b86d2f0}"
ROOT="${BENCH_ROOT:-/leonardo_scratch/fast/$(basename "$WORK")/prefetch-bench}"
PATCHES="$BENCH_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/patches"
FLAG_PATCH="$BENCH_REPO/bench/llama-server-moe-prefetch-flag.patch"
MODULES="${LCPP_MODULES:-gcc/12.2.0 cuda/12.2}"
JOBS="${JOBS:-8}"

for f in "$PATCHES"/0003-*.patch "$PATCHES"/0004-*.patch "$FLAG_PATCH"; do
    [ -f "$f" ] || { echo "[build] missing $f: is $BENCH_REPO on branch feat/prefetch-reads-moe-cache?" >&2; exit 1; }
done

if command -v module >/dev/null 2>&1; then
    for m in $MODULES; do
        module load "$m" || { echo "[build] cannot load module '$m': see 'module avail ${m%%/*}'" >&2; exit 1; }
    done
fi
command -v nvcc >/dev/null || { echo "[build] no nvcc after loading: $MODULES" >&2; exit 1; }
export CC=gcc CXX=g++
echo "[build] $(g++ --version | head -1); $(nvcc --version | tail -1)"

mkdir -p "$ROOT"

# A login node has no GPU driver, so CMake finds no libcuda and the link of llama-server stops at
# "undefined reference to cuMemGetAllocationGranularity" (8 October). The toolkit ships a stub for
# exactly this; it is named libcuda.so, and ld wants libcuda.so.1 when it follows libggml-cuda's
# own dependencies, so a directory of ours holds the stub under both names. The stub is used at
# link time only (rpath-link): the job loads the driver's own libcuda on the compute node.
# BENCH_NO_VMM=1 builds without the driver's virtual memory API instead, if the stub is not enough.
CUDA_LINK_FLAGS=()
stub=$(ls -d "${CUDA_HOME:-/nonexistent}"/targets/x86_64-linux/lib/stubs "${CUDA_HOME:-/nonexistent}"/lib64/stubs 2>/dev/null | head -1 || true)
if [ -n "$stub" ] && [ -e "$stub/libcuda.so" ]; then
    mkdir -p "$ROOT/stubs"
    ln -sf "$stub/libcuda.so" "$ROOT/stubs/libcuda.so"
    ln -sf "$stub/libcuda.so" "$ROOT/stubs/libcuda.so.1"
    CUDA_LINK_FLAGS+=("-DCUDA_CUDA_LIB=$ROOT/stubs/libcuda.so"
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$ROOT/stubs"
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,$ROOT/stubs")
    echo "[build] libcuda stub: $stub"
else
    echo "[build] no libcuda stub under \$CUDA_HOME: the link may fail without a GPU driver" >&2
fi
[ "${BENCH_NO_VMM:-0}" = 1 ] && CUDA_LINK_FLAGS+=(-DGGML_CUDA_NO_VMM=ON)
if [ ! -d "$ROOT/src/.git" ]; then
    git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp "$ROOT/src"
fi
git -C "$ROOT/src" rev-parse --verify -q "$PIN^{commit}" >/dev/null ||
    { echo "[build] $PIN is not in the checkout: git -C $ROOT/src fetch origin" >&2; exit 1; }

for variant in stock patched; do
    tree="$ROOT/$variant"
    if [ ! -d "$tree" ]; then
        git -C "$ROOT/src" worktree add -q --detach "$tree" "$PIN"
        if [ "$variant" = patched ]; then
            git -C "$tree" apply "$PATCHES"/0003-*.patch
            git -C "$tree" apply "$PATCHES"/0004-*.patch
            git -C "$tree" apply "$FLAG_PATCH"
        fi
    fi
    if [ -x "$tree/build/bin/llama-server" ]; then
        echo "[build] $variant: already built"
        continue
    fi
    echo "[build] $variant: configuring"
    cmake -S "$tree" -B "$tree/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF "${CUDA_LINK_FLAGS[@]}" >"$tree/configure.log" 2>&1 ||
        { tail -20 "$tree/configure.log" >&2; echo "[build] $variant: configure failed" >&2; exit 1; }
    echo "[build] $variant: building (a few minutes)"
    cmake --build "$tree/build" --config Release --target llama-server -j "$JOBS" \
        >"$tree/build.log" 2>&1 || { tail -30 "$tree/build.log" >&2; echo "[build] $variant: failed" >&2; exit 1; }
done

module -t list 2>&1 | grep -E "^(gcc|cuda)(/|$)" >"$ROOT/modules.txt" || echo "$MODULES" | tr ' ' '\n' >"$ROOT/modules.txt"
echo "[build] modules: $(tr '\n' ' ' <"$ROOT/modules.txt")"
"$ROOT/patched/build/bin/llama-server" --help 2>&1 | grep -q -- --moe-prefetch ||
    { echo "[build] the patched binary has no --moe-prefetch" >&2; exit 1; }
"$ROOT/patched/build/bin/llama-server" --help 2>&1 | grep -q -- --moe-cache-mib ||
    { echo "[build] the pin has no --moe-cache-mib: not a commit with the MoE cache" >&2; exit 1; }
echo "[build] ok: $ROOT/stock and $ROOT/patched"
