#!/usr/bin/env bash
# llama-server, llama-batched-bench and llama-bench for GH200, built on a
# JUPITER login node from the engine's own llama.cpp submodule: the same
# commit, without EuLLM's patches, so a difference with EuLLM is EuLLM's
# runtime (tools/lumi/install_runtimes.sh does the same for the MI250X).
#
#     bash tools/jupiter/install_runtimes.sh
#
# Into $CAMPAIGN_DIR/runtimes/; env.sh then exports LLAMA_SERVER_BIN.

set -uo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RT="$CAMPAIGN_DIR/runtimes"
BUILD="$RT/llama-build"
LCPP="$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
mkdir -p "$RT"
[ -n "${CUDA_PATH:-}" ] || { echo "[err] no nvcc: module load CUDA"; exit 1; }
command -v cmake >/dev/null || eullm_module load CMake || true
command -v cmake >/dev/null || { echo "[err] no cmake: module avail CMake"; exit 1; }

echo "=== llama.cpp tools (CUDA, sm_${CMAKE_CUDA_ARCHITECTURES:-90}) from the submodule ==="
if CC=gcc CXX=g++ CUDACXX="$CUDA_PATH/bin/nvcc" \
    cmake -S "$LCPP" -B "$BUILD" -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}" \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >"$RT/llama-cmake.log" 2>&1 &&
   cmake --build "$BUILD" --target llama-server llama-batched-bench llama-bench \
        -j "${BUILD_JOBS:-32}" >"$RT/llama-build.log" 2>&1; then
    for tool in llama-server llama-batched-bench llama-bench; do
        cp -f "$BUILD/bin/$tool" "$RT/$tool"
        echo "[ok] $RT/$tool ($(git -C "$LCPP" describe --tags --always 2>/dev/null))"
    done
else
    echo "[!!] did not build: tail $RT/llama-cmake.log $RT/llama-build.log"
    exit 1
fi
