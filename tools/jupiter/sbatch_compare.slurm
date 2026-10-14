#!/bin/bash
# The engine against llama-server on one GH200 node of JUPITER Booster: the
# c07 points setup.sh planned, four at a time (one per GPU), then the same
# five-turn conversations measured on LUMI (multiturn_probe.py), so that the
# MI250X and the H100 compare row by row. Submit through submit.sh.
#
# A booster node is billed whole (OverSubscribe=EXCLUSIVE): 4 GPU-hours an
# hour, and only for the time used. The first full run (job 2326571,
# 14-10-2026: 22 points, then the conversations) took 17 minutes; it asks
# for an hour, which waits less in the queue than three, and stops starting
# points 15 minutes before the end (--margin-s 900) to leave the
# conversations theirs (two model loads each side).

#SBATCH --job-name=eullm-gh200
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

set -uo pipefail
# shellcheck source=env.sh
source "${EULLM_REPO:?submit through submit.sh, which exports EULLM_REPO}/tools/jupiter/env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the CUDA eullm binary}"
GPUS="${SLURM_GPUS_ON_NODE:-4}"

echo "=== eullm on GH200: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
printf '  %-12s %s\n' gpus "$GPUS" engine "$("$EULLM_BIN" --version 2>&1 | head -1)" \
    label "${EULLM_ENGINE_LABEL:-(none)}" repo "${EULLM_BENCH_REV:-?}" \
    llama-server "${LLAMA_SERVER_BIN:-none}" queue "$CAMPAIGN_DIR"
nvidia-smi --query-gpu=index,name,memory.total,power.limit,driver_version --format=csv,noheader |
    sed 's/^/  /'
# Whether a compute node reaches the internet: on 14-10-2026 it did not.
HF=$(curl -s -o /dev/null -m 10 -w '%{http_code}' https://huggingface.co || true)
[ "${HF:-000}" = 000 ] && HF="unreachable" || HF="HTTP $HF"
echo "  huggingface.co from a compute node: $HF"
echo

echo "=== 1. the c07 points ==="
$CAMPAIGN run --queue "$CAMPAIGN_DIR" --devices "0-$((GPUS - 1))" --backend cuda \
    --bind none --site jupiter-booster --engine "$EULLM_BIN" --margin-s 900

echo
echo "=== 2. five turns over a 6,000-token document, twice: prompt_ms per turn ==="
PORT=$((18700 + ${SLURM_JOB_ID:-0} % 300))
OUT="$CAMPAIGN_DIR/logs/eullm-gh200-${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0

wait_ready() {  # PATH PID: up to five minutes for the server to answer, while it lives
    for _ in $(seq 1 300); do
        curl -sf "localhost:$PORT$1" >/dev/null 2>&1 && return 0
        kill -0 "$2" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

conversation() {  # MODEL LABEL: two conversations of five turns on one server
    local m=$1 label=$2 api=eullm ready=/api/version server gguf
    if [ "$label" = llama-server ]; then
        gguf=$("$PY" -c "import sys; sys.path.insert(0, '$EULLM_REPO/bench/campaign')
from point import store_gguf; print(store_gguf('$m') or '')")
        [ -n "$gguf" ] || { echo "[!!] $m is not in $EULLM_MODELS_DIR"; return; }
        api=llama ready=/health
        "$LLAMA_SERVER_BIN" --host 127.0.0.1 --port "$PORT" -m "$gguf" --alias "$m" \
            -c 16384 -np 1 -ngl 999 >"$OUT/$m.$label.log" 2>&1 &
    else
        RUST_LOG="eullm=info,eullm::inference::scheduler=debug" \
            "$EULLM_BIN" serve --port "$PORT" --batch-size 1 --ctx-size 16384 \
            >"$OUT/$m.$label.log" 2>&1 &
    fi
    server=$!
    if wait_ready "$ready" "$server"; then
        "$PY" "$EULLM_REPO/tools/lumi/multiturn_probe.py" --url "http://127.0.0.1:$PORT" \
            --api "$api" --model "$m" --label "$m $label" --turns 5 --conversations 2
    else
        echo "[!!] $label did not start with $m: tail $OUT/$m.$label.log"
    fi
    kill "$server" 2>/dev/null
    wait "$server" 2>/dev/null
    sleep 2
}

for m in qwen3.6-27b qwen3.6-35b-a3b; do
    conversation "$m" eullm
    if [ -x "${LLAMA_SERVER_BIN:-}" ]; then
        conversation "$m" llama-server
    fi
done
echo
echo "=== done $(date -Is) ==="
