#!/bin/bash
# Phase 2 — the ADMINISTRATIVE vertical: legal-it amministrativo, 4B.
# Consiglio di Stato 2017-2024 plus the administrative norms; see the header
# of distill_qwen3_30b_a3b_to_4b_amm.yaml for why it is its own model.
#
# Everything about the node is the split arm's — teacher in BF16 on GPUs 0-1,
# student on GPU 2, three GPUs, 24 cores, 340 GB, two-hour links — because
# that is the geometry measured to be placed (the comments in
# sbatch_phase2_split.slurm carry the measurements). Only the config, the
# data directory and the held-out gate differ.
#
# Its own run dir, with the SAME Phase-1 teacher adapter linked in:
#
#   export EULLM_RUN_DIR=$WORK/eullm_runs/legal_it_amm
#   mkdir -p $EULLM_RUN_DIR/checkpoints $EULLM_RUN_DIR/logs
#   ln -s $WORK/eullm_runs/legal_it/checkpoints/qwen3_30b_a3b_legal_it_continued_pt \
#         $EULLM_RUN_DIR/checkpoints/
#
# The corpus is built on the workstation (the rulings carry personal data
# until anonymised) and copied to EULLM_AMM_DATA_DIR, default
# $WORK/datasets/legal_it_amm — see docs/corpus-acquisition.md.
#
# Chain with afterany; the links end in TIMEOUT by design:
#   cd $EULLM_RUN_DIR && bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase2_amm.slurm" 20
#
#SBATCH --job-name=eullm-p2-amm
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# THREE GPUs, 24 cores, 340 GB — a SHARE of a node, not a whole one, and
# every one of the three numbers matters.
#
# Measured on 2026-09-23 with two probes submitted in the same second,
# identical in walltime, cores and memory, differing only in `--gres`:
#
#   probe-g3   submit 13:46:22   start 13:47:49   → 87 seconds
#   probe-g4   submit 13:46:22   still pending
#
# This arm and the control arm had then been pending for FIFTEEN HOURS at
# two hours of walltime, while the 8 B chain — which asks for three GPUs —
# took five nodes overnight with gaps of 4, 16 and 23 minutes. A node with
# one free card is common; an entirely empty node is not.
#
# The cores and the memory are part of the same request. A Booster node has
# 32 cores, so `--cpus-per-task=32` made the job exclusive by CPU however
# few GPUs it asked for, and 450 GB of ~494 left too little behind to be
# worth anyone's while. Changing only the GRES would have changed nothing.
#
# THIS ARM ALREADY USED ONLY THREE. The config puts the teacher on GPUs 0-1
# and the student on cuda:2, and left GPU 3 idle deliberately so that the
# node was exclusive and the throughput comparable with the control arm.
# That comparison is FINISHED — the +13.5 % design-B figure is measured —
# so the fourth card was protecting a result we already have.
#
# WHAT IT COSTS, for the report rather than a footnote: the node is now
# shared, memory bandwidth and PCIe are contended, and throughput falls by
# an unmeasured amount. Node-hours-per-step before and after this change are
# not comparable and the switch date belongs in the record. What it does NOT
# cost is correctness: sharing changes how fast the arm trains, not what it
# learns, so the loss curve, the perplexity series and the transfer curve
# carry across the boundary untouched.
#
# CUDA_VISIBLE_DEVICES renumbers the three to 0,1,2, so the config's
# `teacher_gpus: [0, 1]` and `student_device: cuda:2` are unchanged.
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
# TWO HOURS, which is a measurement and not a preference.
#
# Probes identical to this job but for their walltime were placed in 50
# seconds at 30 minutes and at 1 hour, and never placed at 3, 4 or 6 — while
# the real 4-hour heads sat pending for 28 hours. Two hours was placed
# twice. The fixed cost of a link is ~22 minutes (14 of startup plus, on
# average, half a save interval of unsaved tail), so 2 h returns ~82 % of the
# node-hours it spends against ~63 % at one hour, and one hour with
# `save_steps: 300` would return nothing at all — 291 steps against a first
# save at 300.
#
# The links end in TIMEOUT by design and resume from the last checkpoint, so
# the length is a scheduling choice and not a training one.
#SBATCH --time=02:00:00
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
if [ -n "${EULLM_AMM_REPO:-}" ]; then
    EULLM_REPO="$EULLM_AMM_REPO"          # explicit wins over everything
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    EULLM_REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"  # interactive runs, no job id
fi
export EULLM_REPO

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"
DATA_DIR="${EULLM_AMM_DATA_DIR:-$WORK/datasets/legal_it_amm}"

# Say which tree this is, on the first line, because the wrong answer is
# invisible for hours otherwise.
echo "[p2amm] repo   $EULLM_REPO"
echo "[p2amm] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[p2amm] data   $DATA_DIR"

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

CFG="$EULLM_REPO/forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_amm.yaml"

# THE HELD-OUT GATE, before anything expensive. Consiglio di Stato 2025-2026
# is the evaluation set of every legal-it model: a corpus that included it
# would train fine and report a number that means nothing. The same call
# asserts the rulings are actually there — pointed at the civil/criminal data
# by mistake, this arm would run to the end answering a different question.
python "$EULLM_REPO/forge/scripts/check_corpus_holdout.py" "$DATA_DIR" \
    --files train.jsonl val.jsonl \
    --holdout-source cds --holdout-from-year 2025 \
    --require-source cds

# Report on ourselves BEFORE the pre-flight, not after. The pre-flight
# imports transformers, which on a cold node cache reads thousands of small
# files off Lustre and takes minutes in complete silence — so starting the
# heartbeat after it left the job mute through the very first phase someone
# would want to watch. Ten minutes of apparent nothing on job 56775410.
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# THREE, and it must match the --gres above or every link of the chain dies
# in the pre-flight after three minutes, having spent a node to do it. That
# is not hypothetical: on 2026-09-22 the 8 B arm waited thirty hours for a
# node and lost it in 174 seconds to a pre-flight it could not pass. The
# resource request and what the job asserts about it are one change, never
# two.
python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --expect-gpus 3 --data-dir "$DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-4B-Base \
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
        --data "$DATA_DIR/val.jsonl" \
        --report "$REPORT"
fi

bash "$EULLM_REPO/forge/scripts/distill.sh" "$CFG" "$DATA_DIR"
