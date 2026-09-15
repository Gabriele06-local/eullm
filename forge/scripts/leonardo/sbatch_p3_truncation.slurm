#!/bin/bash
# P3 — how much of the teacher's distribution top-K actually keeps.
#
# Runs beside anything else: it needs one model, plain transformers, and the
# venv Phase 1 already uses (read-only, nothing installed). No vLLM, so it is
# not blocked behind P1 — which matters, because P3 answers a question about
# the corpus and the teacher rather than about an inference backend.
#
# It decides the size of the cache. ADR-001 Part 4 puts K=32 at ~143 GB over
# the corpus against ~277 GB at K=64, on a 1 TB quota that also holds the HF
# cache, the corpus and every checkpoint.
#
#   cd "$WORK/eullm_runs/v11_pilot"
#   sbatch "$WORK/eullm-v11/forge/scripts/leonardo/sbatch_p3_truncation.slurm"
#
#SBATCH --job-name=eullm-p3-truncation
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

# Which checkout this job runs is decided by where the file came from, not by
# the submitting shell: `sbatch` exports the submitter's environment and
# `env.sh` sets EULLM_REPO to whatever tree the current run is pinned to.
SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EULLM_PILOT_REPO:-}" ]; then
    EULLM_REPO="$EULLM_PILOT_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    EULLM_REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    EULLM_REPO="${EULLM_REPO:-$WORK/eullm-v11}"
fi
export EULLM_REPO
export EULLM_RUN_DIR="${EULLM_PILOT_RUN_DIR:-$WORK/eullm_runs/v11_pilot}"

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
mkdir -p "$EULLM_RUN_DIR/logs"
cd "$EULLM_RUN_DIR" || exit 1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

echo "[p3] repo   $EULLM_REPO"
echo "[p3] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"

# bf16, deliberately: this measures a property of the teacher's distribution,
# and doing it on the int8 copy would confound truncation with quantization.
# P6 measured the quantization separately, which is what allows this to be
# read on its own.
# The Phase-1 adapter, when it is there. The first P3 run measured the BASE
# model because Phase 1 had not finished, and its numbers are a LOWER BOUND:
# continued pre-training on Italian case law should make the teacher more
# peaked on that text, so the retained mass can only go up. Re-running
# without the adapter reproduces a result we already have and buys nothing,
# which is what this block exists to prevent — the flag was easy to forget
# because the script's default is simply "no adapter".
ADAPTER="${EULLM_P3_ADAPTER:-./checkpoints/qwen3_30b_a3b_legal_it_continued_pt}"
if [ -f "$ADAPTER/adapter_config.json" ]; then
    echo "[p3] teacher adapter: $ADAPTER"
    ADAPTER_ARG=(--adapter "$ADAPTER")
    SUFFIX="_adapted"
else
    echo "[p3] WARNING: no adapter at $ADAPTER — measuring the BASE model." >&2
    echo "[p3] That is the run we already have; its numbers are a lower" >&2
    echo "[p3] bound. Link the Phase-1 adapter into \$EULLM_RUN_DIR first." >&2
    ADAPTER_ARG=()
    SUFFIX="_base"
fi

python "$EULLM_REPO/forge/scripts/pilot/measure_topk_truncation.py" \
    --model Qwen/Qwen3-30B-A3B-Base \
    ${ADAPTER_ARG[@]+"${ADAPTER_ARG[@]}"} \
    --data "$EULLM_DATA_DIR/val.jsonl" \
    --samples 64 \
    --seq-len 512 \
    --k 16 32 64 128 \
    --temperatures 1 2 4 \
    --report "$EULLM_RUN_DIR/p3_topk_truncation${SUFFIX}.json"
STATUS=$?

python3 "$EULLM_REPO/forge/scripts/leonardo/queue_stats.py" 2026-09-02 \
    --json "$WORK/eullm_runs/qstats/queue_stats_$(date +%Y%m%d_%H%M).json" || true

exit "$STATUS"
