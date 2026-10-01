#!/usr/bin/env bash
# Wrapper around forge/scripts/distill.py: pass the YAML and, optionally, a
# data dir. A re-run picks up where the last one stopped.
#
# The resume decision belongs to distill.py and this wrapper deliberately does
# not make one. It used to: it looked for checkpoint-* in the YAML's
# output_dir, took the one with the newest mtime, and passed it as
# --resume-from, which is how every phase-2 job launches. Two things were
# wrong with that. mtime is not the step number -- anything that touches a
# checkpoint directory without writing one reorders the list, and the
# copy of $WORK that runs between links is exactly that -- and a resume
# reloads a checkpoint half the run old and redoes thousands of steps.
# distill.py already sorts by step number (_checkpoint_step, whose comment
# says exactly this) and logs the directory it loads, so passing nothing says
# the same thing correctly.
#
# Usage:
#   bash forge/scripts/distill.sh <config.yaml> [data-dir]
#
# Example:
#   bash forge/scripts/distill.sh \
#       forge/training/configs/distill_qwen3_32b_to_7b.yaml \
#       ~/datasets/legal_it

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [data-dir]}"
DATA_DIR="${2:-${TRAINING_DATA_DIR:-$HOME/datasets/legal_it}}"

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

[ -f "$CONFIG" ]    || err "config not found: $CONFIG"
[ -d "$DATA_DIR" ]  || err "data dir not found: $DATA_DIR"
[ -f "$DATA_DIR/train.jsonl" ] || err "missing $DATA_DIR/train.jsonl"
[ -f "$DATA_DIR/val.jsonl" ]   || err "missing $DATA_DIR/val.jsonl"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/forge/scripts/distill.py"
[ -f "$SCRIPT" ] || err "missing $SCRIPT"

# No --resume-from: distill.py reads output_dir from the YAML it already has to
# load, and picks the highest step among the checkpoints it finds.
CMD=(python "$SCRIPT"
     --config "$CONFIG"
     --dataset-dir "$DATA_DIR")

echo
log "Launching:"
printf '   %s\n' "${CMD[*]}"
echo

exec "${CMD[@]}"