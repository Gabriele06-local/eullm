#!/bin/bash
# How fast a compute node reads a large model from Lustre: one stream against
# several at once, bypassing the page cache and through it.
#
#     export SBATCH_ACCOUNT=project_465003366
#     sbatch tools/lumi/sbatch_lustre_probe.slurm [MODEL_DIR]
#
# MODEL_DIR is a directory of GGUF parts in the EuLLM store (default: the
# Coder-480B of c02). Nothing is written: every read goes to /dev/null.
#
# WHY. The 480B (270 GiB) had not finished loading after the hour the runner
# allows (c02, 05-10-2026), so the engine read it at under 80 MB/s. Its parts
# are already on 8 OSTs beyond their first 16 GiB (the scratch directory's
# default layout), so striping is not what limits it. From a login node on
# 06-10-2026, with direct I/O: one stream 116 MB/s, four streams at once
# 245 MB/s in total. A login node is shared and says little about a compute
# node, which is where models load: this job measures there, and the numbers
# decide whether reading the parts in parallel before a load is worth
# building.
#
# Each read is 2 GiB at its own offset, from 16 GiB into a part onwards (the
# bytes on 8 OSTs, where most of a large part is), never the same bytes twice
# in a job, so no cache on either side of the network answers for Lustre.
#
# Costs ~0.25 GPU-hours (small-g bills 0.5 per GCD-hour; the GCD is idle, the
# partition has no GPU-less nodes).

#SBATCH --job-name=eullm-lustre-probe
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=eullm-lustre-probe-%j.out

set -uo pipefail

ACCOUNT="${SBATCH_ACCOUNT:-project_465003366}"
DIR="${1:-/scratch/$ACCOUNT/$USER/eullm-models/qwen3-coder-480b-a35b-instruct-gguf-q4_k_m}"
CHUNK_MIB=2048
START_MIB=16384

PARTS=("$DIR"/*.gguf)
[ -f "${PARTS[0]}" ] || { echo "[err] no GGUF in $DIR" >&2; exit 1; }
SIZES=()
for f in "${PARTS[@]}"; do
    SIZES+=($(($(stat -c %s "$f") / 1048576)))
done

echo "=== Lustre read probe, job ${SLURM_JOB_ID:-?} on $(hostname), $(date -Is) ==="
echo "  $DIR: ${#PARTS[@]} parts, stripe count $(lfs getstripe -c "${PARTS[0]}" 2>/dev/null | head -1)"
lctl get_param llite.*.max_read_ahead_mb llite.*.max_read_ahead_per_file_mb 2>/dev/null |
    sed 's/^/  /'
echo

# Chunk k is part k mod n, at START + CHUNK * (k div n): consecutive reads
# fall on different parts, as a parallel load of a split model would.
NEXT=0
run() {
    local label=$1 streams=$2 flags=$3
    local j k part off t0 t1 failed=0 pids=()
    t0=$(date +%s.%N)
    for ((j = 0; j < streams; j++)); do
        k=$NEXT
        NEXT=$((NEXT + 1))
        part=$((k % ${#PARTS[@]}))
        off=$((START_MIB + CHUNK_MIB * (k / ${#PARTS[@]})))
        if [ $((off + CHUNK_MIB)) -gt "${SIZES[part]}" ]; then
            echo "  [!!] $label x$streams: the parts are too small for this many reads"
            return 1
        fi
        # shellcheck disable=SC2086  # flags is empty or one dd operand
        dd if="${PARTS[part]}" of=/dev/null bs=16M skip=$((off / 16)) \
            count=$((CHUNK_MIB / 16)) $flags status=none &
        pids+=($!)
    done
    for p in "${pids[@]}"; do
        wait "$p" || failed=$((failed + 1))
    done
    t1=$(date +%s.%N)
    awk -v l="$label" -v n="$streams" -v c="$CHUNK_MIB" -v t0="$t0" -v t1="$t1" -v f="$failed" \
        'BEGIN {
            s = t1 - t0; mb = n * c * 1048576 / 1e6
            printf "  %-8s %2d at once  %5.1f GiB in %6.1f s  %6.0f MB/s total  %5.0f MB/s each%s\n",
                   l, n, n * c / 1024, s, mb / s, mb / s / n, (f ? "  (" f " reads FAILED)" : "")
        }'
}

for n in 1 2 4 8 16; do
    run direct "$n" iflag=direct
done
# Through the page cache, as the engine's mapped load reads (Lustre's
# readahead applies here, not above).
for n in 1 4 16; do
    run cached "$n" ""
done
