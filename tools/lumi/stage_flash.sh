#!/usr/bin/env bash
# Put models on LUMI-F, the flash file system, behind links of the same
# names in the EuLLM store, or bring them back:
#
#   export SBATCH_ACCOUNT=project_465003366
#   bash tools/lumi/stage_flash.sh MODEL_ID ...          # copy to flash, link
#   bash tools/lumi/stage_flash.sh --undo MODEL_ID ...   # back to scratch only
#
# On a login node, under tmux: copying runs at Lustre's speed (about 250 MB/s
# from a login node on 06-10-2026, so roughly an hour for 700 GB).
#
# WHY. The Coder-480B (270 GiB) did not finish loading from scratch in the
# hour the runner allows (c02, 05-10-2026). Scratch is LUMI-P, disks: one
# stream read 116 MB/s and four at once 245 MB/s in all from a login node
# (06-10-2026). LUMI-F is SSDs, 8 PB and 1,740 GB/s over 58 OSTs, for
# "very fast disk I/O"; whether it loads a model faster is measured, not
# assumed: sbatch_lustre_probe.slurm on both copies, and every result records
# which one its model was read from (`load.storage` in bench/campaign).
#
# COST. LUMI bills storage by the volume stored over time, flash at 3x
# scratch: 1 TB on flash for a day is 72 TB-hours. The project's flash quota
# is 2 TB unless support raised it (`lumi-quota` shows it).
#
# Nothing on LUMI is backed up, and automatic cleaning of scratch and flash
# may be enabled with three months' notice. So the scratch original of each
# part stays, as a hard link in $EULLM_MODELS_DIR.on-flash/<id> (no copy, the
# same file system), and --undo puts it back.
#
# The model directory and its manifest stay on scratch, because the store
# lists real directories only; each GGUF part in it becomes a symlink to its
# flash copy, swapped in by a rename, so a server opening the model at that
# moment finds either file and never a gap. A run cut short is finished by
# running it again.
#
#   FLASH_MODELS   where the copies go (default /flash/$SBATCH_ACCOUNT/$USER/eullm-models)
#   COPY_JOBS      parts copied at once (default 4)

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"

ACCOUNT="${SBATCH_ACCOUNT:-project_465003366}"
FLASH_MODELS="${FLASH_MODELS:-/flash/$ACCOUNT/$USER/eullm-models}"
ASIDE="$EULLM_MODELS_DIR.on-flash"
UNDO=0
if [ "${1:-}" = "--undo" ]; then
    UNDO=1
    shift
fi
if [ $# -eq 0 ]; then
    echo "usage: $0 [--undo] MODEL_ID ...   (ids as in $EULLM_MODELS_DIR)" >&2
    exit 2
fi

stage() {
    local id=$1
    local src="$EULLM_MODELS_DIR/$id" dst="$FLASH_MODELS/$id" keep="$ASIDE/$id"
    local f n parts=() linked=0
    if [ ! -f "$src/manifest.json" ]; then
        echo "[!!] $id: not in the store ($src)"
        return 1
    fi
    for f in "$src"/*.gguf; do
        if [ -L "$f" ]; then
            linked=$((linked + 1))
        elif [ -f "$f" ]; then
            parts+=("$f")
        fi
    done
    if [ ${#parts[@]} -eq 0 ]; then
        if [ "$linked" -gt 0 ]; then
            echo "  $id: already on flash ($linked parts)"
            return 0
        fi
        echo "[!!] $id: no GGUF in $src"
        return 1
    fi
    mkdir -p "$dst" "$keep" || return 1
    echo "=== $id: ${#parts[@]} parts, $(du -ch "${parts[@]}" | tail -1 | cut -f1) → $dst ==="

    # Each part to a .partial renamed when whole: an interrupted copy leaves
    # nothing a link could point at. A part already there at full size is
    # not copied again.
    # shellcheck disable=SC2016  # expanded by the inner shell
    if ! printf '%s\0' "${parts[@]}" | xargs -0 -P "${COPY_JOBS:-4}" -I{} sh -c '
        n=$(basename "$1")
        if [ -f "$2/$n" ] && [ "$(stat -c %s "$1")" = "$(stat -c %s "$2/$n")" ]; then
            echo "  $n: already copied"
            exit 0
        fi
        cp "$1" "$2/$n.partial" && mv -f "$2/$n.partial" "$2/$n" && echo "  [ok] $n"
    ' _ {} "$dst"; then
        rm -f "$dst"/*.partial
        echo "[!!] $id: a copy failed (the flash quota? lumi-quota); scratch is untouched"
        return 1
    fi
    for f in "${parts[@]}"; do
        n=$(basename "$f")
        if [ "$(stat -c %s "$f")" != "$(stat -c %s "$dst/$n" 2>/dev/null)" ]; then
            echo "[!!] $id: $n has another size on flash; scratch is untouched"
            return 1
        fi
    done

    for f in "${parts[@]}"; do
        n=$(basename "$f")
        if ! { ln -f "$f" "$keep/$n" &&
               ln -sfn "$dst/$n" "$src/.$n.link" &&
               mv -Tf "$src/.$n.link" "$f"; }; then
            rm -f "$src/.$n.link"
            echo "[!!] $id: could not link $n; the store is as it was for the parts not yet linked"
            return 1
        fi
    done
    echo "[ok] $id reads from flash; scratch originals in $keep"
}

undo() {
    local id=$1
    local src="$EULLM_MODELS_DIR/$id" keep="$ASIDE/$id"
    local f n target back=0
    for f in "$src"/*.gguf; do
        [ -L "$f" ] || continue
        n=$(basename "$f")
        if [ -f "$keep/$n" ]; then
            mv -Tf "$keep/$n" "$f" || return 1
        else
            # No original kept: copy the flash file back before dropping the link.
            target=$(readlink -f "$f")
            { cp "$target" "$src/.$n.partial" && mv -Tf "$src/.$n.partial" "$f"; } || {
                rm -f "$src/.$n.partial"
                echo "[!!] $id: could not bring $n back"
                return 1
            }
        fi
        back=$((back + 1))
    done
    rmdir "$keep" 2>/dev/null
    rm -rf "${FLASH_MODELS:?}/$id"
    echo "[ok] $id: $back parts back on scratch, flash copy removed"
}

FAILED=()
for id in "$@"; do
    if [ "$UNDO" = 1 ]; then
        undo "$id" || FAILED+=("$id")
    else
        stage "$id" || FAILED+=("$id")
    fi
done

if [ -d "$FLASH_MODELS" ]; then
    used=$(du -sb "$FLASH_MODELS" 2>/dev/null | cut -f1)
    awk -v b="${used:-0}" 'BEGIN { printf "\non flash now: %.2f TB, %.0f TB-hours a day at 3x\n",
                                   b / 1e12, b / 1e12 * 24 * 3 }'
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[!!] not done: ${FAILED[*]}"
    exit 1
fi
