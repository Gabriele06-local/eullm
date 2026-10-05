#!/bin/bash
# Consiglio di Stato ruling cards (cds_schede.py): one teacher llama-server
# per GPU, three GPUs, two-hour links of a chain.
#
#   CS_IDS=$WORK/eval/cds/cds-train-ids.txt CS_OUT=$WORK/eullm_runs/cds/schede.jsonl \
#       sbatch forge/scripts/leonardo/sbatch_cds_schede.slurm
#
# Every link carries on from the last card written and stops taking new
# rulings after 105 minutes, so it ends on its own inside its two hours; a
# link that finds nothing left exits at once. The first link's log is the
# teacher throughput step 0 of the plan asks for (rulings/min, tokens/s).
#
# Three GPUs, 24 cores, 340 GB: the shape that gets placed (forge/CLAUDE.md).
# The teacher is Qwen3-30B-A3B as GGUF Q8_0, ~32 GB, which fits one 64 GB
# A100 with its cache; bf16 (61 GB) would not.
#
# Settable:
#   CS_IDS       rulings to card, one id per line (required; cds_split.py)
#   CS_OUT       cards JSONL (required)
#   CS_TEACHER   teacher GGUF (default the judge's Q8_0)
#   CS_CHUNKS    chunk files with the rulings (default $WORK/datasets/legal_it_amm/{train,val}.jsonl)
#   CS_OPENGA    OpenGA CSVs (default $WORK/eval/openga/cds-sentenze-*.csv)
#   CS_ARGS      extra cds_schede.py options, e.g. "--limit 200"
#   CS_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-cds-schede
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${CS_REPO:-}" ]; then
    REPO="$CS_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

TEACHER="${CS_TEACHER:-$WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf}"
# shellcheck disable=SC2206
CHUNKS=(${CS_CHUNKS:-$WORK/datasets/legal_it_amm/train.jsonl $WORK/datasets/legal_it_amm/val.jsonl})
# shellcheck disable=SC2206
OPENGA=(${CS_OPENGA:-$WORK/eval/openga/cds-sentenze-*.csv})
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"

fail=0
[ -s "${CS_IDS:-}" ] || { echo "[cs] no ids at '${CS_IDS:-}'" >&2; fail=1; }
[ -n "${CS_OUT:-}" ] || { echo "[cs] CS_OUT is required" >&2; fail=1; }
[ -s "$TEACHER" ] || { echo "[cs] no teacher GGUF at $TEACHER" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[cs] no llama-server at $SERVER" >&2; fail=1; }
for f in "${CHUNKS[@]}"; do [ -s "$f" ] || { echo "[cs] missing $f" >&2; fail=1; }; done
[ "$GPUS" -ge 1 ] || { echo "[cs] no GPU visible" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1
OG=()
for f in "${OPENGA[@]}"; do [ -s "$f" ] && OG+=("$f"); done
[ "${#OG[@]}" -gt 0 ] || echo "[cs] no OpenGA CSV: cards without appeal number and section" >&2

if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
# The GPUs as Slurm numbered them, one teacher each.
DEVS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
echo "[cs] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')), GPUs $DEVS"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

OPENGA_ARGS=()
[ "${#OG[@]}" -eq 0 ] || OPENGA_ARGS=(--openga "${OG[@]}")
# shellcheck disable=SC2086
python "$REPO/forge/scripts/cds_schede.py" --chunks "${CHUNKS[@]}" "${OPENGA_ARGS[@]}" \
    --ids "$CS_IDS" --out "$CS_OUT" --serve "$TEACHER" --gpus "$DEVS" \
    --llama-server "$SERVER" --stop-after 6300 ${CS_ARGS:-}
