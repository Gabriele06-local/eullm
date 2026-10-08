#!/bin/bash
# Watch one large model load on a whole LUMI-G node, every 30 seconds: how
# much of the file has reached memory, how much has reached the GPUs, and
# what the server's threads are doing.
#
#     export SBATCH_ACCOUNT=project_465003366
#     export EULLM_BIN=$PWD/target-next2/release/eullm
#     sbatch tools/lumi/sbatch_load_watch.slurm [MODEL_ID] [LOAD_THREADS] [SERVE_FLAG ...]
#
# Default: the Coder-480B, --load-threads 0; flags after the third argument go
# to `eullm serve` as they are (--no-mmap, for one). The output is
# eullm-load-watch-<job>.out beside where it was submitted, the server's own
# log next to it.
#
# WHY. The 480B (270 GiB, Q4_K_M, six parts) did not load within an hour on
# 8 GCDs, in c02 (05-10-2026) and in c09 with 16 and 32 readers (07-10-2026),
# while the 235B (132 GiB) loaded cold in 110 s. The log stops at "Loading
# model": a slow read, a stall on the GPUs and a stall in Lustre all look the
# same there. Each line below tells them apart:
#   cache     the node's page cache: grows while the file is read
#   file_rss  the server's resident pages of the mapped file
#   vram      VRAM used on all GCDs together: grows as tensors are uploaded
#   threads   the server's threads by state: D waits on I/O, R runs, S sleeps
# and every five minutes the threads that are not asleep, with the kernel
# function they wait in (wchan).

#SBATCH --job-name=eullm-load-watch
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:10:00
#SBATCH --output=eullm-load-watch-%j.out

set -uo pipefail
# shellcheck source=campaign_env.sh
source "${EULLM_REPO:-$PWD}/tools/lumi/campaign_env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the ROCm eullm binary}"

MODEL="${1:-qwen3-coder-480b-a35b-instruct-gguf-q4_k_m}"
LOAD_THREADS="${2:-0}"
EXTRA=("${@:3}")
PORT=$((18500 + ${SLURM_JOB_ID:-0} % 400))
OUT="eullm-load-watch-${SLURM_JOB_ID:-local}"

echo "=== load watch: $MODEL, --load-threads $LOAD_THREADS ${EXTRA[*]}, on $(hostname), $(date -Is) ==="
echo "  engine $("$EULLM_BIN" --version 2>&1 | head -1)"
grep -E "MemTotal|MemAvailable" /proc/meminfo | sed 's/^/  /'
cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)/memory.max" 2>/dev/null |
    sed 's/^/  cgroup memory.max /'
# The Lustre client's own cap on cached file data: a model larger than what is
# left of it is evicted by Lustre while it is still being loaded.
# lctl is not on a compute node's PATH; the same file is readable under /proc.
{ lctl get_param -n llite.*.max_cached_mb 2>/dev/null || /usr/sbin/lctl get_param -n llite.*.max_cached_mb 2>/dev/null ||
    cat /proc/fs/lustre/llite/*/max_cached_mb /sys/fs/lustre/llite/*/max_cached_mb 2>/dev/null; } |
    grep -E "max_cached_mb|used_mb" | sed 's/^/  lustre /'
echo

"$EULLM_BIN" serve --port "$PORT" --batch-size 1 --ctx-size 4096 --fit-strict \
    --load-threads "$LOAD_THREADS" "${EXTRA[@]}" >"$OUT.server.log" 2>&1 &
server=$!
for _ in $(seq 1 60); do
    curl -s "localhost:$PORT/api/version" >/dev/null 2>&1 && break
    sleep 1
done

# One request names the model, so the server loads it; it answers after the load.
curl -s -m 3900 "localhost:$PORT/api/generate" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
    >"$OUT.reply.json" 2>&1 &
request=$!

# GiB from a count of units of $2 bytes: kB (/proc's 1024, the default) or bytes (1).
gib() { awk -v v="${1:-0}" -v unit="${2:-1024}" 'BEGIN { printf "%7.1f", v * unit / 1073741824 }'; }
t0=$(date +%s)
n=0
while kill -0 "$request" 2>/dev/null && kill -0 "$server" 2>/dev/null; do
    t=$(($(date +%s) - t0))
    cache=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
    file_rss=$(awk '/^RssFile:/ {print $2}' "/proc/$server/status" 2>/dev/null)
    vram=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/ {s += $NF} END {print s + 0}')
    states=$(ps -L -o stat= -p "$server" 2>/dev/null | cut -c1 | sort | uniq -c | awk '{printf "%s%s ", $1, $2}')
    printf '%5d s  cache %s GiB  file_rss %s GiB  vram %s GiB  threads %s\n' \
        "$t" "$(gib "$cache")" "$(gib "$file_rss")" "$(gib "$vram" 1)" "$states"
    if [ $((n % 10)) -eq 0 ]; then
        ps -L -o tid=,stat=,pcpu=,wchan:32=,comm= -p "$server" 2>/dev/null |
            awk '$2 !~ /^S/' | head -20 | sed 's/^/          /'
    fi
    n=$((n + 1))
    sleep 30
done

echo
echo "=== after $(($(date +%s) - t0)) s: request $(kill -0 "$request" 2>/dev/null && echo running || echo ended), server $(kill -0 "$server" 2>/dev/null && echo alive || echo gone) ==="
head -c 600 "$OUT.reply.json"
echo
echo "--- server log, last lines ---"
tail -25 "$OUT.server.log"
kill "$server" 2>/dev/null
wait "$server" 2>/dev/null
