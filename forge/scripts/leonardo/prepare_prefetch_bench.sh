#!/bin/bash
# Everything the prefetch bench needs, then the job, from ONE command on a login node (run it
# in tmux: the model is 96 GiB):
#
#   bash $WORK/eullm-prefetch/forge/scripts/leonardo/prepare_prefetch_bench.sh
#
# 1. downloads the model (resumable; a finished file is kept), 2. builds the two llama-server
# binaries (build_prefetch_bench.sh, idempotent), 3. submits sbatch_prefetch_bench.slurm unless
# one is already queued or running. Running it twice cannot queue two jobs. Everything lives
# under $BENCH_ROOT, by default the project's fast scratch (see build_prefetch_bench.sh).
set -euo pipefail

: "${WORK:?WORK is not set: run this on Leonardo}"
export BENCH_REPO="${BENCH_REPO:-$WORK/eullm-prefetch}"
export BENCH_ROOT="${BENCH_ROOT:-/leonardo_scratch/fast/$(basename "$WORK")/prefetch-bench}"
HERE="$BENCH_REPO/forge/scripts/leonardo"
# shellcheck disable=SC1091
source "$HERE/env.sh"

mkdir -p "$BENCH_ROOT/models"
echo "[prepare] model into $BENCH_ROOT/models ($(date -Is))"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ggml-org/Qwen3.8-Flash-Next-GGUF', allow_patterns=['*IQ4_NL*'], local_dir='$BENCH_ROOT/models')
"
for part in 00001 00002; do
    [ -s "$BENCH_ROOT/models/Qwen3.8-Flash-Next-IQ4_NL-$part-of-00002.gguf" ] ||
        { echo "[prepare] part $part of the model is missing after the download" >&2; exit 1; }
done

bash "$HERE/build_prefetch_bench.sh"

if squeue --me -h -n eullm-prefetch-bench | grep -q .; then
    echo "[prepare] a bench job is already queued or running:"
    squeue --me -n eullm-prefetch-bench
    exit 0
fi
cd "$BENCH_REPO"
mkdir -p logs
sbatch forge/scripts/leonardo/sbatch_prefetch_bench.slurm
