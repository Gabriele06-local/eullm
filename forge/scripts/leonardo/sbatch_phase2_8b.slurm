#!/bin/bash
# Phase 2 — the LARGER STUDENT. Qwen3-8B-Base in the split-node layout:
# teacher Qwen3-30B-A3B in BF16 on GPUs 0-1, student on GPU 2, GPU 3 free.
#
# The variable here is the STUDENT SIZE, so everything else is held identical
# to the split arm — teacher, Phase-1 adapter, corpus, LoRA rank, objective,
# schedule, seed. Against the 4 B at matched steps on the same held-out
# corpus this gives a scaling curve for a vertical domain, which is a better
# result than either model on its own. See the config's header for the rest.
#
# It also raises concurrency, which is the allocation's real problem:
# queue_stats measured a mean of 1.08 nodes while busy on 2026-09-22, against
# a plan that called for stretches at two or three. A second independent
# chain also covers the other's queue gaps — when one waits, the other can
# take a backfill window.
#
# ITS OWN RUN DIRECTORY, or the two chains resume from each other's
# checkpoints:
#
#   export EULLM_RUN_DIR=$WORK/eullm_runs/legal_it_8b
#
# The Phase-1 teacher adapter has to be reachable from there, and it is the
# SAME adapter the other arms use — the teacher must not be a variable. Link
# it rather than copying 61 GB:
#
#   mkdir -p $EULLM_RUN_DIR/checkpoints
#   ln -s $WORK/eullm_runs/legal_it/checkpoints/qwen3_30b_a3b_legal_it_continued_pt \
#         $EULLM_RUN_DIR/checkpoints/
#
# Chain it with afterany, because a 24 h link ends in TIMEOUT by design and
# afterok would never fire:
#   bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" --after-any \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase2_8b.slurm" 4
#
#SBATCH --job-name=eullm-p2-8b
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# THREE GPUs, and only this chain asks for three.
#
# The layout uses three — teacher on 0-1, student on 2 — so asking for four
# asked for a whole node, and a whole node has to be entirely idle before it
# can be given away. On 2026-09-22 `boost_usr_prod` had 301 nodes in `mix`
# and exactly one `idle`, itself unresponsive. Three GPUs make every node with
# a single free card a candidate.
#
# THIS IS NOT WHY THE OTHER CHAINS STALL, and the distinction matters. The
# split arm has been asking for four GPUs and getting them all month: 12-hour
# links completed on the 16th, 18th and 21st, and 3h23 on the morning of the
# 22nd. Its current wait is contention that worsened, not a defect in its
# request. So it keeps four, and its exclusive node, because sharing would
# cost throughput and break comparability with everything already measured on
# it — the +13.5 % design-B figure above all.
#
# This chain has no such history to protect. It is new, its whole purpose is
# to raise concurrency, and a slower run that runs beats a faster one that
# waits. THE COST, for the report rather than a footnote: the node is shared,
# memory bandwidth and PCIe are contended, and throughput may fall by an
# unmeasured amount. Its throughput must not be tabulated beside the split
# arm's.
#
# CUDA_VISIBLE_DEVICES renumbers the three to 0,1,2, so the config's
# `teacher_gpus: [0, 1]` and `student_device: cuda:2` are unchanged.
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Which checkout this job runs is decided by where THIS FILE came from, not by
# the submitting shell's environment.
#
# `sbatch` exports the submitter's environment by default and `env.sh` sets
# EULLM_REPO to whatever tree the current run is pinned to — so submitting
# `$WORK/eullm-v11/.../sbatch_phase2.slurm` from a shell that had sourced
# env.sh for the frozen tree silently ran the frozen tree's code and config.
# Nothing in the log said so. `scontrol` still knows the path the job was
# submitted from, and the repo root is three directories above it.
SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EULLM_PHASE2B_REPO:-}" ]; then
    EULLM_REPO="$EULLM_PHASE2B_REPO"          # explicit wins over everything
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    EULLM_REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"  # interactive runs, no job id
fi
export EULLM_REPO

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"

# Say which tree this is, on the first line, because the wrong answer is
# invisible for hours otherwise.
echo "[p2b] repo   $EULLM_REPO"
echo "[p2b] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"

# Phase 1 must have produced the teacher adapter this YAML points at.
ADAPTER="./checkpoints/qwen3_30b_a3b_legal_it_continued_pt"
[ -f "$ADAPTER/adapter_config.json" ] || {
    echo "[err] Phase-1 adapter not found at $EULLM_RUN_DIR/$ADAPTER" >&2
    echo "[err] This arm runs from its own run dir, so the adapter Phase 1" >&2
    echo "[err] wrote is not under it. Link it — read-only, it is merged" >&2
    echo "[err] into the teacher at load time and never written:" >&2
    echo "[err]   mkdir -p \$EULLM_RUN_DIR/checkpoints" >&2
    echo "[err]   ln -s \$WORK/eullm_runs/legal_it/$ADAPTER \\" >&2
    echo "[err]         \$EULLM_RUN_DIR/checkpoints/" >&2
    exit 1
}

CFG="$EULLM_REPO/forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_8b.yaml"

# Report on ourselves BEFORE the pre-flight, not after. The pre-flight
# imports transformers, which on a cold node cache reads thousands of small
# files off Lustre and takes minutes in complete silence — so starting the
# heartbeat after it left the job mute through the very first phase someone
# would want to watch. Ten minutes of apparent nothing on job 56775410.
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --expect-gpus 3 --data-dir "$EULLM_DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-8B-Base \
    --assert-frozen-base "$CFG"

# Gate: the teacher is quantized, and in distillation its distribution IS
# the target, so a quantization error is trained into the student rather
# than merely degrading the run. Measure it before spending the budget.
#
# Cached deliberately. The check loads both teachers at once and takes
# tens of minutes; running it again in every job of an afterany chain would
# re-pay that for a result that cannot have changed. The report records
# which bit width was validated, so switching 8-bit to 4-bit invalidates it.
TEACHER_BITS="$(grep -qE '^teacher_load_in_8bit:[[:space:]]*true' "$CFG" && echo 8 \
    || { grep -qE '^teacher_load_in_4bit:[[:space:]]*true' "$CFG" && echo 4 || echo 0; })"
REPORT="$EULLM_RUN_DIR/quant_validation_${TEACHER_BITS}bit.json"

if [ "$TEACHER_BITS" = "0" ]; then
    echo "[..] teacher is not quantized — no logit validation needed"
elif [ -f "$REPORT" ]; then
    echo "[ok] teacher already validated at ${TEACHER_BITS}-bit ($REPORT)"
else
    echo "[..] validating the ${TEACHER_BITS}-bit teacher against bf16"
    python "$EULLM_REPO/forge/scripts/validate_quantized_teacher.py" \
        --model "$(grep -E '^teacher_model:' "$CFG" | cut -d: -f2- | xargs)" \
        --bits "$TEACHER_BITS" \
        --adapter "$(grep -E '^teacher_adapter:' "$CFG" | cut -d: -f2- | xargs)" \
        --data "$EULLM_DATA_DIR/val.jsonl" \
        --report "$REPORT"
fi

bash "$EULLM_REPO/forge/scripts/distill.sh" "$CFG" "$EULLM_DATA_DIR"
