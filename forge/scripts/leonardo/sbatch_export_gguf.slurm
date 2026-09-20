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
#   EULLM_EXPORT_EVERY     export checkpoints at multiples of this (default 2000)
#   EULLM_EXPORT_CKPT_DIR  where to look (default: the split arm's output_dir)
#   EULLM_EXPORT_DIR       where merged models and GGUFs go
#   EULLM_EXPORT_KEEP      merged HF dirs to keep; GGUFs are never deleted
#
# Start it once:
#   sbatch "$EULLM_REPO/forge/scripts/leonardo/sbatch_export_gguf.slurm"
#
#SBATCH --job-name=eullm-export-gguf
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR"

echo "[export] repo   $EULLM_REPO"
echo "[export] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"

EVERY="${EULLM_EXPORT_EVERY:-2000}"
CKPT_DIR="${EULLM_EXPORT_CKPT_DIR:-$EULLM_RUN_DIR/checkpoints/qwen3_4b_legal_it_distilled_split}"
OUT_DIR="${EULLM_EXPORT_DIR:-$EULLM_RUN_DIR/exports}"
KEEP="${EULLM_EXPORT_KEEP:-2}"
mkdir -p "$OUT_DIR"

echo "[export] watching $CKPT_DIR every $EVERY steps → $OUT_DIR"

if [ ! -d "$CKPT_DIR" ]; then
    echo "[export] no checkpoint directory yet — nothing to do this round"
fi

exported_any=0
for ckpt in $(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    step="${ckpt##*-}"
    case "$step" in ''|*[!0-9]*) continue;; esac
    # Only the cadence, and only once. `save_steps` is far finer than anything
    # worth packaging — exporting all of them would fill $WORK with models
    # nobody asked for.
    [ $((step % EVERY)) -eq 0 ] || continue

    name="legal-it-4b-step${step}"
    gguf="$OUT_DIR/gguf/$name"
    if [ -d "$gguf" ] && compgen -G "$gguf/*.gguf" >/dev/null; then
        continue                       # already packaged on an earlier round
    fi

    echo "[export] === step $step ==="
    merged="$OUT_DIR/merged/$name"
    if ! python "$EULLM_REPO/forge/scripts/export_checkpoint.py" \
            --checkpoint "$ckpt" --output "$merged" --force; then
        echo "[export] merge FAILED for $ckpt — leaving it for the next round" >&2
        continue
    fi

    # LCPP_SKIP_UPDATE: the llama.cpp checkout is shared and pinned; a git
    # pull here would change the quantizer under a series of exports that are
    # supposed to be comparable with each other.
    if ! LCPP_SKIP_UPDATE=1 bash "$EULLM_REPO/forge/scripts/quantize_to_gguf.sh" \
            "$merged" "$gguf"; then
        echo "[export] GGUF conversion FAILED for step $step" >&2
        continue
    fi
    exported_any=1
    echo "[export] packaged $gguf"

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
done

[ "$exported_any" -eq 1 ] || echo "[export] nothing new to package this round"

echo "[export] available models:"
ls -1 "$OUT_DIR/gguf" 2>/dev/null || echo "  (none yet)"

# Re-arm. Four hours is well under the cadence at which new checkpoints
# appear, so a missed round costs nothing and a crashed round self-heals.
if [ "${EULLM_EXPORT_ONCE:-0}" = "1" ]; then
    echo "[export] EULLM_EXPORT_ONCE=1 — not re-arming"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    if sbatch --begin=now+4hours "$SELF"; then
        echo "[export] re-armed for four hours from now"
    else
        echo "[export] FATAL: re-submission failed — the export chain stops here." >&2
        exit 1
    fi
else
    echo "[export] cannot find myself to re-arm — run again by hand" >&2
    exit 1
fi
