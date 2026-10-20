# A model copied once into the node's memory (/dev/shm), for the jobs that
# load it from there: source this file, then
#
#     path=$(stage_model <hub id or dir>)    # prints where to load it from
#
# Loading a 61 GB model straight from $PROJECT took 3.5 minutes on 14-10-2026
# and 50-70 on 17-10 (distillation links 2362188-2362191, 2371817), and the
# judge's four graders ran out of their hour on 19-10 (jobs 2403554,
# 2403558): a load is many small reads at once, as fast as the file system
# is free. A copy is a few large reads, made once per node. Without room in
# /dev/shm the caller loads from $PROJECT, as before. The copy is removed
# when the job's shell exits (STAGE_DIR, default /dev/shm/forge-<job>).

STAGE_DIR="${STAGE_DIR:-/dev/shm/forge-${SLURM_JOB_ID:-local}}"
trap 'rm -rf "$STAGE_DIR"' EXIT
trap 'exit 143' TERM INT

stage_source() {    # stage_source <hub id or dir>: the directory that holds the model
    local id="$1" err g dir
    [ -d "$id" ] && { echo "$id"; return; }
    err=$(mktemp)
    if dir=$(python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], local_files_only=True))" "$id" 2>"$err"); then
        rm -f "$err"; echo "$dir"; return
    fi
    # On 19-10-2026 snapshot_download refused a snapshot that was complete
    # on disk (jobs 2414120, 2414122): the snapshot directory itself will do.
    g=$(ls -d "${HF_HUB_CACHE:-$HF_HOME/hub}/models--${id//\//--}"/snapshots/*/ 2>/dev/null | head -1)
    if [ -n "$g" ]; then
        echo "[stage] snapshot_download: $(tail -n 1 "$err"); using ${g%/}" >&2
        rm -f "$err"; echo "${g%/}"; return
    fi
    echo "[stage] $id is not in the cache: $(tail -n 1 "$err")" >&2
    rm -f "$err"; return 1
}

stage_model() {     # stage_model <hub id or dir>: prints where to load it from
    local id="$1" snap dst need avail t0
    dst="$STAGE_DIR/${id//\//--}"
    if [ -f "$dst/config.json" ]; then echo "$dst"; return; fi
    snap=$(stage_source "$id") || { echo "$id"; return; }
    need=$(du -sLk "$snap" | cut -f1)
    avail=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 { print $4 }')
    if [ -z "$avail" ] || [ "$avail" -lt $((need + need / 10)) ]; then
        echo "[stage] /dev/shm has ${avail:-no} kB free for $((need / 1048576)) GB: loading $id from $snap" >&2
        echo "$id"; return
    fi
    t0=$(date +%s)
    mkdir -p "$dst.tmp"
    find "$snap/" -maxdepth 1 ! -type d -printf '%f\n' |
        xargs -P 16 -I{} cp -L "$snap/{}" "$dst.tmp/" &&
        mv "$dst.tmp" "$dst" || { echo "[stage] copy of $id into /dev/shm failed: loading it from $snap" >&2; rm -rf "$dst.tmp"; echo "$id"; return; }
    echo "[stage] $id copied into $dst: $((need / 1048576)) GB in $(( $(date +%s) - t0 )) s" >&2
    echo "$dst"
}
