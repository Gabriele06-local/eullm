#!/bin/bash
# Package intermediate checkpoints as testable GGUF models, automatically.
#
# A verticalized model exists at every checkpoint, not only at the end of the
# run. Each `checkpoint-N` is a complete LoRA adapter over the student base;
# merged, it loads, generates and quantizes exactly like the final one, and is
# simply less trained. `distill.py` merges only when the whole run finishes,
# which made the deliverable look gated on an epoch that takes weeks of chained
# 24 h jobs on a busy cluster. It never was.
#
# So this watches the checkpoint directory and exports anything new, on a
# cadence, without being asked.
#
# ON THE SERIAL PARTITION, AND THAT IS THE POINT. Merging an adapter is
# elementwise arithmetic over the weights and GGUF conversion plus
# llama-quantize are CPU-only. Doing either inside the training job would burn
# A100 time on work that needs no GPU *and* stall the training loop while it
# ran. Here it costs no GPU allocation at all and cannot touch the chain.
#
# Self-resubmitting, like the queue-stats job: it re-arms itself so the
# exports keep appearing for as long as the run does.
#
# EVERYTHING IS SETTABLE AT SUBMISSION. Export each of these before `sbatch`
# and it is inherited by every re-arm, so the cadence and the disk policy are
# decided once rather than edited into the file:
#
#   EULLM_EXPORT_EVERY     minimum TRAINING STEPS between exports (default
#                          4000). A floor on the gap, not a set of step
#                          numbers: the newest surviving checkpoint is what
#                          gets packaged.
#   EULLM_EXPORT_KEEP_F16  keep the intermediate F16 GGUF (default 0, delete)
#   EULLM_EXPORT_KEEP      merged HF dirs to keep (default 2)
#   EULLM_EXPORT_MAX_GGUF  Q4 models to keep, oldest deleted first; 0 = all
#   EULLM_EXPORT_MIN_FREE_GB  skip a round below this much free space (default 50)
#   EULLM_EXPORT_HOURS     hours between rounds (default 4)
#   EULLM_EXPORT_CKPT_DIR  where to look (default: the split arm's output_dir)
#   EULLM_EXPORT_DIR       where merged models and GGUFs go
#
# WHY STEPS AND NOT HOURS, since a time-based cadence is the obvious instinct:
# what makes two exports worth having is training progress between them, not
# wall-clock. Links queue for hours on a busy partition, so two exports a day
# apart during a drought can be the same model — paid for twice in disk. Set
# EULLM_EXPORT_EVERY to the progress you want to see; at the duty cycle
# measured so far (~20 %) 4000 steps is about two exports a week.
#
# DISK, because the numbers are not small. Each export writes a ~8 GB
# intermediate F16 GGUF and a ~2.5 GB Q4_K_M for a 4 B model. Keeping both for
# the ~15 exports left in an epoch is ~160 GB of a 1 TB $WORK shared with a
# 61 GB teacher cache and two runs' checkpoints. The F16 is a build artefact
# that regenerates from the merged directory in minutes, so it goes by
# default and the Q4 — the thing anyone actually wants — stays.
#
# Start it once:
#   sbatch "$EULLM_REPO/forge/scripts/leonardo/sbatch_export_gguf.slurm"
#
#   EULLM_EXPORT_EVERY=10000 \
#       sbatch "$EULLM_REPO/forge/scripts/leonardo/sbatch_export_gguf.slurm"
#
#SBATCH --job-name=eullm-export-gguf
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Matched to sbatch_quantize.slurm, which does the same GGUF conversion on
# this partition and is known to pass. An earlier 8 cores / 64 G was rejected
# outright with QOSMaxMemoryPerUser: lrd_all_serial caps per-user memory well
# below a compute node's, and the merge only needs room for a 4 B model in
# bf16 (~8 GB of weights) plus the copy save_pretrained writes.
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EULLM_EXPORT_REPO:-}" ]; then
    EULLM_REPO="$EULLM_EXPORT_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    EULLM_REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
fi
export EULLM_REPO

# The run dir is the directory this was SUBMITTED from, not whatever the
# submitting shell happened to export.
#
# This bit on the first real round. `env.sh` defaults EULLM_RUN_DIR to the
# control arm's directory, so a submission from a shell that had not sourced
# it sent the job looking for the split arm's checkpoints under the control
# arm's run dir. It found none, printed "no checkpoint directory yet", and
# re-armed — correct behaviour for a wrong directory, and indistinguishable
# from "training has not written anything yet". Meanwhile the log landed in
# the right place, because --output IS relative to the submit directory, so
# the two halves of the job disagreed about where they were.
#
# Fourth time this family of bug has cost us something. Same fix as
# sbatch_phase2.slurm: decide from where the job came, not from the
# environment it inherited.
if [ -z "${EULLM_RUN_DIR:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    export EULLM_RUN_DIR="$SLURM_SUBMIT_DIR"
fi
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"
echo "[export] run dir $EULLM_RUN_DIR"

echo "[export] repo   $EULLM_REPO"
echo "[export] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"

EVERY="${EULLM_EXPORT_EVERY:-4000}"
KEEP_F16="${EULLM_EXPORT_KEEP_F16:-0}"
MAX_GGUF="${EULLM_EXPORT_MAX_GGUF:-0}"
MIN_FREE_GB="${EULLM_EXPORT_MIN_FREE_GB:-50}"
HOURS="${EULLM_EXPORT_HOURS:-4}"
# Re-armed rounds inherit these, so a cadence chosen once at submission holds
# for the life of the chain instead of reverting to the defaults tomorrow.
export EULLM_EXPORT_EVERY="$EVERY" EULLM_EXPORT_KEEP_F16="$KEEP_F16"
export EULLM_EXPORT_MAX_GGUF="$MAX_GGUF" EULLM_EXPORT_MIN_FREE_GB="$MIN_FREE_GB"
export EULLM_EXPORT_HOURS="$HOURS"
CKPT_DIR="${EULLM_EXPORT_CKPT_DIR:-$EULLM_RUN_DIR/checkpoints/qwen3_4b_legal_it_distilled_split}"
OUT_DIR="${EULLM_EXPORT_DIR:-$EULLM_RUN_DIR/exports}"
KEEP="${EULLM_EXPORT_KEEP:-2}"
mkdir -p "$OUT_DIR"

echo "[export] watching $CKPT_DIR every $EVERY steps → $OUT_DIR"
keep_gguf_label="$MAX_GGUF"
[ "$MAX_GGUF" -eq 0 ] && keep_gguf_label="all"
echo "[export] keep F16=$KEEP_F16  merged=$KEEP  gguf=$keep_gguf_label  every ${HOURS}h"

# Refuse rather than half-write. A conversion that runs out of disk leaves a
# truncated .gguf that llama.cpp loads far enough to look plausible.
free_gb=$(df -BG --output=avail "$OUT_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "[export] only ${free_gb}G free under $OUT_DIR, need ${MIN_FREE_GB}G —" \
         "skipping this round" >&2
    SKIP_ROUND=1
else
    SKIP_ROUND=0
    [ -n "$free_gb" ] && echo "[export] ${free_gb}G free"
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "[export] WARNING: no such directory: $CKPT_DIR" >&2
    echo "[export] This is more often a wrong path than a run that has" >&2
    echo "[export] written nothing. Check that the run dir above is the one" >&2
    echo "[export] the training chain writes to, or set" >&2
    echo "[export] EULLM_EXPORT_CKPT_DIR explicitly. Directories present:" >&2
    ls -1d "$EULLM_RUN_DIR"/checkpoints/*/ 2>/dev/null | sed 's/^/[export]   /' >&2 ||
        echo "[export]   (no checkpoints/ at all under the run dir)" >&2
fi

# Pick the NEWEST checkpoint and export it if enough steps have passed since
# the last export — rather than looking for exact multiples of EVERY.
#
# The exact-multiple version could not work, and would have failed silently.
# `save_total_limit` keeps only the newest few checkpoints, so at save_steps
# 300 and ~360 steps/hour a checkpoint-8000 exists for about 2.7 hours while
# this job wakes every 4. It would have found nothing on almost every round
# and reported "nothing new to package" while doing nothing, indefinitely.
#
# Newest-since-last is immune to that: whatever survives pruning is what gets
# packaged, and the cadence is a floor on the gap between exports rather than
# a set of step numbers that have to be caught in flight.
newest=""
newest_step=-1
for ckpt in "$CKPT_DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    step="${ckpt##*-}"
    case "$step" in ''|*[!0-9]*) continue;; esac
    if [ "$step" -gt "$newest_step" ]; then
        newest_step="$step"
        newest="$ckpt"
    fi
done

last_exported=-1
for done_gguf in "$OUT_DIR"/gguf/legal-it-4b-step*; do
    [ -d "$done_gguf" ] || continue
    compgen -G "$done_gguf/*.gguf" >/dev/null || continue   # half-finished round
    step="${done_gguf##*step}"
    case "$step" in ''|*[!0-9]*) continue;; esac
    [ "$step" -gt "$last_exported" ] && last_exported="$step"
done

exported_any=0
if [ "$SKIP_ROUND" = "1" ]; then
    echo "[export] round skipped on free space"
elif [ "$newest_step" -lt 0 ]; then
    echo "[export] no checkpoints yet"
elif [ "$last_exported" -ge 0 ] && [ $((newest_step - last_exported)) -lt "$EVERY" ]; then
    echo "[export] newest is step $newest_step, last exported $last_exported —" \
         "less than $EVERY apart, waiting"
else
    step="$newest_step"
    name="legal-it-4b-step${step}"
    gguf="$OUT_DIR/gguf/$name"
    merged="$OUT_DIR/merged/$name"
    echo "[export] === step $step (last exported: $last_exported) ==="

    if ! python "$EULLM_REPO/forge/scripts/export_checkpoint.py" \
            --checkpoint "$newest" --output "$merged" --force; then
        echo "[export] merge FAILED for $newest — leaving it for the next round" >&2
    # LCPP_SKIP_UPDATE: the llama.cpp checkout is shared and pinned; a git pull
    # here would change the quantizer under a series of exports that are
    # supposed to be comparable with each other.
    elif ! LCPP_SKIP_UPDATE=1 bash "$EULLM_REPO/forge/scripts/quantize_to_gguf.sh" \
            "$merged" "$gguf"; then
        echo "[export] GGUF conversion FAILED for step $step" >&2
    else
        exported_any=1
        echo "[export] packaged $gguf"
        if [ "$KEEP_F16" != "1" ]; then
            # A build intermediate, ~8 GB and four times the Q4 it produced.
            # Regenerates from the merged directory in minutes.
            rm -f "$gguf"/*-f16.gguf && echo "[export] dropped the F16 intermediate"
        fi
    fi

    # The merged HF directory is ~8 GB and exists only to be converted; the
    # GGUF is ~2.5 GB and is the artefact anyone actually wants. Keep a couple
    # of the former, never delete the latter.
    if [ "$KEEP" -gt 0 ]; then
        ls -dt "$OUT_DIR/merged"/*/ 2>/dev/null | tail -n +$((KEEP + 1)) |
            while read -r old; do
                echo "[export] pruning merged dir $old (GGUF kept)"
                rm -rf "$old"
            done
    fi

    # Q4 models, oldest first. Off by default: at ~2.5 GB each they are the
    # cheap half, and a series of them across training is the point.
    if [ "$MAX_GGUF" -gt 0 ]; then
        ls -dt "$OUT_DIR/gguf"/*/ 2>/dev/null | tail -n +$((MAX_GGUF + 1)) |
            while read -r old; do
                echo "[export] pruning old GGUF $old (EULLM_EXPORT_MAX_GGUF=$MAX_GGUF)"
                rm -rf "$old"
            done
    fi
fi

[ "$exported_any" -eq 1 ] || echo "[export] nothing new to package this round"

echo "[export] available models:"
ls -1 "$OUT_DIR/gguf" 2>/dev/null || echo "  (none yet)"

# Re-arm. The wake interval is deliberately well under the export cadence:
# waking often and deciding to do nothing is free, while waking rarely risks
# trailing the run. A missed round costs nothing and a crashed round self-heals.
if [ "${EULLM_EXPORT_ONCE:-0}" = "1" ]; then
    echo "[export] EULLM_EXPORT_ONCE=1 — not re-arming"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    if sbatch --begin=now+"${HOURS}"hours "$SELF"; then
        echo "[export] re-armed for ${HOURS}h from now"
    else
        echo "[export] FATAL: re-submission failed — the export chain stops here." >&2
        exit 1
    fi
else
    echo "[export] cannot find myself to re-arm — run again by hand" >&2
    exit 1
fi
