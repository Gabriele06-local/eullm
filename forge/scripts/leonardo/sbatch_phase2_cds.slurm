#!/bin/bash
# Phase 2 — the LARGER CORPUS. Cassazione + Consiglio di Stato 2017-2024,
# in the same split-node layout as the arm it is compared against: teacher
# Qwen3-30B-A3B in BF16 on GPUs 0-1, student Qwen3-4B on GPU 2.
#
# The variable is THE CORPUS, so everything else is held identical to the
# split arm — teacher, Phase-1 adapter, student, LoRA rank, objective,
# schedule, seed, node layout. At matched steps on the same held-out corpus,
# the difference between the two arms is the corpus and nothing else.
#
# It is also the fourth independent chain, which is the allocation's real
# problem. A chain has exactly ONE eligible job at a time — the rest are held
# on `afterany` — so three chains compete for three slots however many links
# are queued behind them. Adding a chain adds an eligible job; nothing else we
# control does. See docs/cineca/allocation-recovery.md for the measurements.
#
# ITS OWN RUN DIRECTORY, or the chains resume from each other's checkpoints:
#
#   export EULLM_RUN_DIR=$WORK/eullm_runs/legal_it_cds
#
# The Phase-1 teacher adapter has to be reachable from there, and it is the
# SAME adapter every arm uses — the teacher must not be a variable. Link it
# rather than copying 61 GB:
#
#   mkdir -p $EULLM_RUN_DIR/checkpoints
#   ln -s $WORK/eullm_runs/legal_it/checkpoints/qwen3_30b_a3b_legal_it_continued_pt \
#         $EULLM_RUN_DIR/checkpoints/
#
# Chain it with afterany, because a link ends in TIMEOUT by design and
# afterok would never fire:
#   bash "$EULLM_REPO/forge/scripts/leonardo/submit_chain.sh" \
#       "$EULLM_REPO/forge/scripts/leonardo/sbatch_phase2_cds.slurm" 4
#
#SBATCH --job-name=eullm-p2-cds
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Which checkout this job runs is decided by where THIS FILE came from, not by
# the submitting shell's environment. `sbatch` exports the submitter's
# environment and `env.sh` sets EULLM_REPO to whatever tree the current run is
# pinned to, so a job submitted from a shell that had sourced env.sh for a
# frozen tree silently ran the frozen tree's code and config, with nothing in
# the log saying so.
SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EULLM_CDS_REPO:-}" ]; then
    EULLM_REPO="$EULLM_CDS_REPO"              # explicit wins over everything
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    EULLM_REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    EULLM_REPO="${EULLM_REPO:-$WORK/eullm}"   # interactive runs, no job id
fi
export EULLM_REPO

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"

# The combined corpus, which is the whole point of this arm. It is NOT the
# default data dir — env.sh points that at the Cassazione-only corpus every
# other arm uses, and inheriting it here would produce a run that looks
# correct, trains happily, and answers a different question.
DATA_DIR="${EULLM_CDS_DATA_DIR:-$WORK/datasets/legal_it_cds}"

cd "$EULLM_RUN_DIR"

echo "[p2cds] repo   $EULLM_REPO"
echo "[p2cds] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[p2cds] data   $DATA_DIR"

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

CFG="$EULLM_REPO/forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_cds.yaml"

# THE HELD-OUT GATE, and it runs before anything expensive.
#
# Consiglio di Stato 2025-2026 is the evaluation set. A corpus that included
# it would not fail — it would train fine and report a better number that
# means nothing, and it would invalidate every transfer-curve point already
# measured, because those would no longer be comparable with the ones after.
# There is no later stage that can notice this, so it is checked here.
#
# The same call asserts the new material is actually present: this arm exists
# to train on the larger corpus, and pointed at the old data directory it
# would run to completion answering a different question.
#
# EULLM_CDS_MARKER is the substring that identifies Consiglio di Stato
# records — it matches `source_id`, `source` or `kind`, and `kind` is what
# format_pretraining.py infers from the corpus filename. Name the CdS corpus
# files so the marker matches, or set it to whatever they are named.
MARKER="${EULLM_CDS_MARKER:-cds}"
python "$EULLM_REPO/forge/scripts/check_corpus_holdout.py" "$DATA_DIR" \
    --files train.jsonl val.jsonl \
    --holdout-source "$MARKER" --holdout-from-year 2025 \
    --require-source "$MARKER"

# Report on ourselves BEFORE the pre-flight, not after. The pre-flight
# imports transformers, which on a cold node cache reads thousands of small
# files off Lustre and takes minutes in complete silence.
# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --expect-gpus 4 --data-dir "$DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-4B-Base \
    --assert-frozen-base "$CFG"

# The teacher is BF16 here, so there is no quantization to validate — the
# split arm's cached report covers the arms that use an 8-bit teacher. Kept
# as an explicit line rather than an omission, so the next person does not
# wonder whether it was forgotten.
echo "[p2cds] teacher is BF16 — no quantized-logit validation needed"

bash "$EULLM_REPO/forge/scripts/distill.sh" "$CFG" "$DATA_DIR"
