#!/bin/bash
# Does copying a MoE's experts ahead (EuLLM's prefetch, patches 0003 and 0004 on llama.cpp)
# read a long prompt faster than plain llama-server, on an A100 with 64 GB and a model whose
# experts do not fit it? One GPU does the measuring; the other two are there for the last
# pair of runs, where all three are visible and the prefetch must switch itself off and leave
# the answer unchanged (the case a similar open llama.cpp PR gets wrong).
#
# Model: ggml-org/Qwen3.8-Flash-Next-GGUF IQ4_NL, 96 GiB, the file the maintainers measure.
# Prepare it and the binaries on a login node first: prepare_prefetch_bench.sh does the model, the
# binaries and this submission in one go (docs/leonardo-runbook.md, "Inference-engine bench").
# BENCH_ROOT defaults to the project's fast scratch, not $WORK (nearly full).
#   bash forge/scripts/leonardo/build_prefetch_bench.sh
#   source forge/scripts/leonardo/env.sh && python -c "from huggingface_hub import snapshot_download as d; d('ggml-org/Qwen3.8-Flash-Next-GGUF', allow_patterns=['*IQ4_NL*'], local_dir='$BENCH_ROOT/models')"
#
# Submit from $WORK/eullm-prefetch (the branch with these files), account from env.sh:
#   cd $WORK/eullm-prefetch && sbatch forge/scripts/leonardo/sbatch_prefetch_bench.slurm
#
# Shape (forge/CLAUDE.md): two hours, three GPUs, 24 cores, 340 GB. The experts are read into
# pinned memory (--load-mode none), 96 GiB of them, so the job needs the memory, not the GPUs.
# Elapsed time is what is billed: about 1.5 node hours at most. A variant that cannot finish
# before the limit is skipped, and what ran is in the log, one line per run.
#
# Results: $BENCH_ROOT/results/<job id>/, and the table at the end of the log.
#
#SBATCH --job-name=eullm-prefetch-bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

BENCH_REPO="${BENCH_REPO:-$WORK/eullm-prefetch}"
# shellcheck disable=SC1091
source "$BENCH_REPO/forge/scripts/leonardo/env.sh"

ROOT="${BENCH_ROOT:-/leonardo_scratch/fast/$(basename "$WORK")/prefetch-bench}"
MODEL="${MODEL:-$ROOT/models/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00002.gguf}"
CACHE_MIB="${CACHE_MIB:-16000}"
BUDGET=$((2 * 3600 - 600))   # seconds; the last ten minutes are left for the log
S="$ROOT/stock/build/bin/llama-server"
P="$ROOT/patched/build/bin/llama-server"

for f in "$MODEL" "${MODEL/00001-of-00002/00002-of-00002}" "$S" "$P" "$ROOT/modules.txt" \
         "$BENCH_REPO/bench/llama_server_mmap_compare.sh"; do
    [ -e "$f" ] || { echo "[bench] missing $f: see the header of this script" >&2; exit 1; }
done
while read -r m; do
    module load "$m" || { echo "[bench] cannot load module $m" >&2; exit 1; }
done <"$ROOT/modules.txt"

OUT="$ROOT/results/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
echo "[bench] $(date -Is) node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[bench] pcie: $(nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | head -1)"
echo "[bench] model $MODEL, cache $CACHE_MIB MiB, llama.cpp $(git -C "$ROOT/src" log -1 --format='%h %cd' --date=short)"

one='env CUDA_VISIBLE_DEVICES=0'
three='env CUDA_VISIBLE_DEVICES=0,1,2'
cache="-ngl 99 --cpu-moe --moe-cache-mib $CACHE_MIB"
args=(
    "mmap=$one $S|$cache"
    "pinned=$one $S|$cache --load-mode none"
    "prefetch=$one $P|$cache --load-mode none --moe-prefetch 4"
    "mg-stock=$three $S|-ngl 99 --cpu-moe --load-mode none -sm layer"
    "mg-prefetch=$three $P|-ngl 99 --cpu-moe --load-mode none -sm layer --moe-prefetch 4"
)

# Rounds: the three single-GPU configurations, then the two that matter most in the other
# order, then the multi-GPU pair. Each runs on its own server, and is skipped when fifteen
# minutes of the budget are not left.
round=0
for order in "mmap pinned prefetch" "prefetch pinned" "mg-stock mg-prefetch"; do
    round=$((round + 1))
    for name in $order; do
        if [ "$SECONDS" -gt $((BUDGET - 900)) ]; then
            echo "[bench] r$round $name: skipped, $((BUDGET - SECONDS)) s left"
            continue
        fi
        echo "[bench] r$round $name: start at ${SECONDS}s"
        OUT="$OUT/r$round" ORDER="$name" PORT=$((11600 + round)) \
            CTX=40960 N_UBATCH=2048 PROMPT_TOKENS=33200 \
            bash "$BENCH_REPO/bench/llama_server_mmap_compare.sh" "$MODEL" "${args[@]}" |
            grep -v '^config ' | tee -a "$OUT/table.txt"
    done
done

echo
echo "[bench] done at ${SECONDS}s. config, read tok/s, write tok/s, answer hashes:"
cat "$OUT/table.txt"
grep -h 'moe prefetch' "$OUT"/r*/llama-server-*.log 2>/dev/null | sort | uniq -c
