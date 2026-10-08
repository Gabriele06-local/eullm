#!/bin/bash
# Why EuLLM falls behind llama-server with many requests at once (c07), on one
# GCD: both servers, one model, 1, 4 and 16 requests at once, each with the
# same sampling options (concurrency_probe.py), and EuLLM's own account of
# where each step's time goes (the `eullm::steps` log lines).
#
#     export SBATCH_ACCOUNT=project_465003366
#     export EULLM_BIN=$PWD/target-next3/release/eullm
#     sbatch tools/lumi/sbatch_concurrency_diag.slurm [MODEL_ID] [CONCURRENCY ...]
#
# Default: qwen3-8b, 1 4 16. Output: eullm-concurrency-<job>.out beside where
# it was submitted, every server's log next to it (*.eullm-<n>.log,
# *.llama-<n>.log).
#
# Reading it. One JSON line per round: `agg_tok_s` is every token of the
# round over its wall time. Profile `none` is what c07 measured (each server
# its own defaults), `ollama` and `llama` give both servers the same chain.
# In EuLLM's log, every `steps:` line splits ten seconds of the scheduler
# into decode (the GPU), sample (choosing each sequence's token, on one CPU
# thread), emit (text and the reply channel) and prefill.

#SBATCH --job-name=eullm-concurrency
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=eullm-concurrency-%j.out

set -uo pipefail
# shellcheck source=campaign_env.sh
source "${EULLM_REPO:-$PWD}/tools/lumi/campaign_env.sh"
: "${EULLM_BIN:?set EULLM_BIN to the ROCm eullm binary}"
: "${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN (install_runtimes.sh builds it)}"
for bin in "$EULLM_BIN" "$LLAMA_SERVER_BIN"; do
    [ -x "$bin" ] || { echo "[err] $bin is not there: build it first"; exit 1; }
done

MODEL="${1:-qwen3-8b}"
shift || true
if [ $# -gt 0 ]; then LEVELS=("$@"); else LEVELS=(1 4 16); fi
SLOT_CTX=4096
PORT=$((18600 + ${SLURM_JOB_ID:-0} % 300))
OUT="eullm-concurrency-${SLURM_JOB_ID:-local}"
PROBE="$PY $EULLM_REPO/tools/lumi/concurrency_probe.py"
GGUF=$("$PY" -c "import sys; sys.path.insert(0, '$EULLM_REPO/bench/campaign')
from point import store_gguf; print(store_gguf('$MODEL') or '')")
[ -n "$GGUF" ] || { echo "[err] $MODEL is not in $EULLM_MODELS_DIR"; exit 1; }

echo "=== concurrency diagnosis: $MODEL at ${LEVELS[*]} requests, on $(hostname), $(date -Is) ==="
echo "  eullm        $("$EULLM_BIN" --version 2>&1 | head -1)"
echo "  llama-server $("$LLAMA_SERVER_BIN" --version 2>&1 | grep -m1 version)"
echo "  gguf         $GGUF"
echo

wait_ready() {  # PATH PID: up to five minutes for the server to answer, while it lives
    for _ in $(seq 1 300); do
        curl -sf "localhost:$PORT$1" >/dev/null 2>&1 && return 0
        kill -0 "$2" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

stop() {
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
    sleep 2
}

for n in "${LEVELS[@]}"; do
    ctx=$((SLOT_CTX * n))

    # The scheduler's step lines, every ten seconds while it works.
    RUST_LOG="eullm=info,eullm::steps=debug" \
        "$EULLM_BIN" serve --port "$PORT" --batch-size "$n" --ctx-size "$ctx" \
        --cache-type-k f16 --cache-type-v f16 >"$OUT.eullm-$n.log" 2>&1 &
    server=$!
    if wait_ready /api/version "$server"; then
        $PROBE --url "http://127.0.0.1:$PORT" --api eullm --model "$MODEL" --label eullm \
            --concurrency "$n" --profiles none,ollama,llama,greedy
    else
        echo "[!!] eullm did not start at $n: tail $OUT.eullm-$n.log"
    fi
    stop "$server"
    echo "--- eullm at $n: its step lines ---"
    grep "steps:" "$OUT.eullm-$n.log" | tail -12

    "$LLAMA_SERVER_BIN" --host 127.0.0.1 --port "$PORT" -m "$GGUF" --alias "$MODEL" \
        -c "$ctx" -np "$n" -ngl 999 -ctk f16 -ctv f16 >"$OUT.llama-$n.log" 2>&1 &
    server=$!
    if wait_ready /health "$server"; then
        $PROBE --url "http://127.0.0.1:$PORT" --api llama --model "$MODEL" --label llama-server \
            --concurrency "$n" --profiles none,ollama,llama,greedy
    else
        echo "[!!] llama-server did not start at $n: tail $OUT.llama-$n.log"
    fi
    stop "$server"
    echo
done
echo "=== done $(date -Is) ==="
