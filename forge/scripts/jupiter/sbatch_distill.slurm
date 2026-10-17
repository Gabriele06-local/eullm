#!/bin/bash
# One link of the distillation chain on JUPITER Booster: two runs side by
# side on one node (deliverable 3), each distill.py with the teacher in BF16
# alone on one GPU and the student on the next (deliverable 2):
#
#   arm 1: GPUs 0-1, cores 0-143    (probe, 14-10-2026: GPU i sits by
#   arm 2: GPUs 2-3, cores 144-287   cores 72i-72i+71)
#
# By default the arms are the 4B and the 8B student of
# forge/training/configs/jupiter/ on the legislation corpus
# ($FORGE_DATA_DIR/norms, make_norms_split.py); FORGE_DISTILL_ARMS names
# others as "name:config name:config". Each arm has its own run directory,
# $FORGE_RUN_DIR/<name>, where its checkpoints are; distill.py resumes from
# the newest one, and an arm whose run is complete (checkpoints/*/merged) is
# not started again.
#
# Submit through submit_distill.sh, which checks the inputs and chains the
# links (--dependency=singleton). Whole node: 4 GPU-hours an hour. Two hours
# a link: each run trains until FORGE_TRAIN_S (110 minutes by default) after
# the link began and is then stopped. The teacher is first copied into the
# node's memory and both runs load it from there (below); each run saves
# every 100 steps, so a link loses little to its end.
# The GPUs' power is sampled every 5 s into $FORGE_RUN_DIR/power/ (Slurm
# keeps no energy here): deliverable 9.

#SBATCH --job-name=forge-distill
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

set -uo pipefail
# sbatch runs a copy of this file from Slurm's spool: the checkout is where
# the job was submitted from (submit_distill.sh submits from the repo).
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
CFG_DIR="$EULLM_REPO/forge/training/configs/jupiter"
ARMS="${FORGE_DISTILL_ARMS:-norms-4b:$CFG_DIR/distill_qwen3_30b_a3b_to_4b_norms.yaml norms-8b:$CFG_DIR/distill_qwen3_30b_a3b_to_8b_norms.yaml}"
DATA="${FORGE_DISTILL_DATA:-$FORGE_DATA_DIR/norms}"
TRAIN_S="${FORGE_TRAIN_S:-6600}"
JOB="${SLURM_JOB_ID:-local}"
T_JOB=$(date +%s)
GPUS=("0,1" "2,3")
CORES=("0-143" "144-287")

echo "=== Forge distillation: job $JOB on $(hostname), $(date -Is) ==="
echo "  repo $EULLM_REPO ($(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo 'no git here'))"
echo "  data $DATA, training ${TRAIN_S}s per run"
[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/val.jsonl" ] ||
    { echo "[!!] no $DATA/train.jsonl and val.jsonl: make_norms_split.py on a login node"; exit 1; }
python -c "import transformers, peft, accelerate, datasets; print('  imports: transformers', transformers.__version__, 'peft', peft.__version__)" ||
    { echo "[!!] the venv does not import on this node"; tr ':' '\n' <<<"${PYTHONPATH:-}" | sed 's/^/    /'; exit 1; }

mkdir -p "$FORGE_RUN_DIR/power"
POWER="$FORGE_RUN_DIR/power/distill-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!

# The teacher copied once into the node's memory (/dev/shm), both runs loading
# it from there. Each run reading its 61 GB from $PROJECT took 3.5 minutes on
# 14-10-2026 and 50-70 on 17-10 (links 2362188-2362191, 2371817; 12 for
# 2371816): a load of many small reads at once, as fast as the file system
# is free. A copy is a few large reads, made once per node instead of twice.
# Without room in /dev/shm the run reads from $PROJECT, as before.
SHM="/dev/shm/forge-$JOB"
trap 'rm -rf "$SHM"' EXIT
trap 'exit 143' TERM INT
stage_teacher() {   # stage_teacher <hub id or dir>: prints where to load it from
    local id="$1" snap dst need avail t0
    dst="$SHM/${id//\//--}"
    if [ -f "$dst/config.json" ]; then echo "$dst"; return; fi
    snap=$(python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], local_files_only=True))" "$id" 2>/dev/null) ||
        { [ -d "$id" ] && snap="$id"; } || { echo "$id"; return; }
    need=$(du -sLk "$snap" | cut -f1)
    avail=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 { print $4 }')
    if [ -z "$avail" ] || [ "$avail" -lt $((need + need / 10)) ]; then
        echo "[teacher] /dev/shm has ${avail:-no} kB free for $((need / 1048576)) GB: loading $id from $snap" >&2
        echo "$id"; return
    fi
    t0=$(date +%s)
    mkdir -p "$dst.tmp"
    find "$snap/" -maxdepth 1 ! -type d -printf '%f\n' |
        xargs -P 16 -I{} cp -L "$snap/{}" "$dst.tmp/" &&
        mv "$dst.tmp" "$dst" || { echo "[teacher] copy of $id into /dev/shm failed: loading it from $snap" >&2; rm -rf "$dst.tmp"; echo "$id"; return; }
    echo "[teacher] $id copied into $dst: $((need / 1048576)) GB in $(( $(date +%s) - t0 )) s" >&2
    echo "$dst"
}

PIDS=() NAMES=() LOGS=()
i=0
for arm in $ARMS; do
    name="${arm%%:*}" cfg="${arm#*:}"
    if [ "$i" -ge 2 ]; then
        echo "[!!] $name: two runs a node, the rest wait"; continue
    fi
    [ -f "$cfg" ] || { echo "[!!] $name: no config $cfg"; i=$((i + 1)); continue; }
    run="$FORGE_RUN_DIR/$name"
    mkdir -p "$run/logs"
    out=$(sed -n 's/^output_dir:[[:space:]]*//p' "$cfg" | sed 's/[[:space:]]*#.*//')
    if [ -f "$run/$out/merged/config.json" ]; then
        echo "--- $name: complete ($run/$out/merged), not started"
        i=$((i + 1)); continue
    fi
    log="$run/logs/distill-$JOB.log"
    teacher=$(stage_teacher "$(sed -n 's/^teacher_model:[[:space:]]*//p' "$cfg" | sed 's/[[:space:]]*#.*//')")
    # The copy's minutes come out of the training time, not the link's margin.
    train_s=$(( TRAIN_S - ($(date +%s) - T_JOB) ))
    echo "--- $name: GPUs ${GPUS[$i]}, cores ${CORES[$i]}, $cfg -> $run/$out, teacher from $teacher, ${train_s}s"
    (cd "$run" && CUDA_VISIBLE_DEVICES=${GPUS[$i]} timeout --signal=TERM "$train_s" \
        taskset -c "${CORES[$i]}" python "$EULLM_REPO/forge/scripts/distill.py" \
        --config "$cfg" --dataset-dir "$DATA" --teacher-model "$teacher") >"$log" 2>&1 &
    PIDS+=($!) NAMES+=("$name") LOGS+=("$log")
    i=$((i + 1))
done

for k in "${!PIDS[@]}"; do
    wait "${PIDS[$k]}"; rc=$?
    case $rc in
        0) what="finished its run" ;;
        124) what="stopped at its time limit, as planned" ;;
        *) what="FAILED (exit $rc)" ;;
    esac
    echo
    echo "=== ${NAMES[$k]}: $what ==="
    grep -E '^\[(resume|step|eval|ckpt|done|merge|err|teacher|student)' "${LOGS[$k]}" | tail -12 | sed 's/^/  /'
    [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && tail -20 "${LOGS[$k]}" | sed 's/^/    /'
done

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
echo
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) ==="
