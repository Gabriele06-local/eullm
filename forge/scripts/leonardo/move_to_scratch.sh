#!/usr/bin/env bash
# Move downloaded base models from $WORK to the project's fast scratch,
# leaving a symlink where they were, without stopping a job that uses them.
#
#   sbatch -J eullm-move -p lrd_all_serial -c 2 --mem=4G -t 04:00:00 \
#       --wrap "bash forge/scripts/leonardo/move_to_scratch.sh \
#               /leonardo_scratch/fast/$(basename $WORK)/models \
#               $HF_HOME/hub $WORK/models/Qwen3-14B"
#
# $WORK is the project's 1 TB, and on 2026-10-10 the base models alone took a
# large part of it: the Hugging Face cache (teachers, judges, retrieval
# models), Qwen3-14B, Qwen3.8-27B, Mistral-Small-24B. They are the one kind
# of file that can be had again for free, from Hugging Face, so they go to the
# fast scratch, which is separate from that quota and purged when unused.
# What only we can produce -- adapters, merged models, GGUFs, answers, grades,
# corpora, rulings -- stays on $WORK. See docs/leonardo-runbook.md, "Storage".
#
# Copy, then swap. The original stays in place, and in use, while it is
# copied (rsync, then a second rsync for anything that changed meanwhile); the
# swap is a rename and a symlink, a moment in which a job opening the path
# finds one or the other. A job that already loaded the model keeps it in
# memory; the next one follows the symlink. `mv` across filesystems instead
# would leave the path half-copied for minutes, under a chain whose next link
# could start loading at any time.
#
# Each path is skipped, never forced, when it is already a symlink, does not
# exist, or would leave less than MOVE_MARGIN_GB (default 50) free on the
# scratch. Prints one line per path and $WORK's use at the end.

set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <scratch dir> <path on \$WORK>..." >&2
    exit 2
fi
dest="$1"; shift
mkdir -p "$dest"
margin_kb=$(( ${MOVE_MARGIN_GB:-50} * 1024 * 1024 ))

for src in "$@"; do
    src="${src%/}"
    if [ -L "$src" ]; then echo "[move] $src: already a symlink"; continue; fi
    if [ ! -d "$src" ]; then echo "[move] $src: not there"; continue; fi
    need=$(du -sk "$src" | cut -f1)
    free=$(df -Pk "$dest" | awk 'NR == 2 {print $4}')
    if [ "$need" -gt $(( free - margin_kb )) ]; then
        echo "[move] $src: $(( need / 1048576 )) GB would leave less than ${MOVE_MARGIN_GB:-50} GB free on the scratch ($(( free / 1048576 )) GB free), skipped"
        continue
    fi
    dst="$dest/$(basename "$src")"
    if ! rsync -a "$src"/ "$dst"/ || ! rsync -a --delete "$src"/ "$dst"/; then
        echo "[move] $src: copy failed, left in place"
        continue
    fi
    mv "$src" "$src.old" && ln -s "$dst" "$src" && rm -rf "$src.old" &&
        echo "[move] $src -> $dst ($(( need / 1048576 )) GB)"
done
[ -n "${WORK:-}" ] && df -h "$WORK" | tail -1
exit 0
