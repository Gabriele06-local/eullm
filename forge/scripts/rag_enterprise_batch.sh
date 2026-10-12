#!/usr/bin/env bash
# Asks the exam to RAG Enterprise once per model, one model after the other,
# unattended: for each, the model goes into .env, RAG Enterprise starts, the
# questions are asked (rag_enterprise_eval.py ask) and RAG Enterprise stops.
#
#   export RAG_PASSWORD   # once, by hand: read -s RAG_PASSWORD; export RAG_PASSWORD
#   bash forge/scripts/rag_enterprise_batch.sh ~/rag-test/engine \
#       ~/rag-test/norm-exam-devbig.jsonl ~/rag-test \
#       "mistral-small-24b=$HOME/rag-test/models/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf" \
#       "qwen36-35b-a3b=$HOME/rag-test/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|--moe-cache auto"
#
# Each model is "label=file.gguf", optionally "|eullm options" for
# EULLM__EXTRA_ARGS (RAG Enterprise 0.2+). Answers go to
# OUT_DIR/answers-<label>-<RAG_TAG>.jsonl (RAG_TAG: the RAG Enterprise version,
# default v022), RAG Enterprise's log to
# OUT_DIR/engine-<label>.log. A model whose answers file is complete is
# skipped; one that fails to start is logged and the next one goes on, so a
# night is never lost to the first problem. The .env is copied before the
# first model and put back at the end, whatever happens.
set -u

ENGINE_DIR=$1; ITEMS=$2; OUT_DIR=$3; shift 3
EVAL="$(cd "$(dirname "$0")" && pwd)/rag_enterprise_eval.py"
URL=${RAG_URL:-http://localhost:8000}
TAG=${RAG_TAG:-v022}
START_TIMEOUT=${START_TIMEOUT:-1200}   # seconds: a first start downloads eullm, a big model loads slowly

[ -n "${RAG_PASSWORD:-}" ] || { echo "[batch] export RAG_PASSWORD first" >&2; exit 2; }
[ -x "$ENGINE_DIR/i3k-rag-engine" ] || { echo "[batch] no i3k-rag-engine in $ENGINE_DIR" >&2; exit 2; }
if curl -fs "$URL/health" >/dev/null 2>&1; then
    echo "[batch] RAG Enterprise is already running at $URL: stop it first (Ctrl+C)" >&2; exit 2
fi
mkdir -p "$OUT_DIR"
ENV_FILE="$ENGINE_DIR/.env"
BACKUP="$OUT_DIR/env-before-batch-$(date +%F-%H%M)"
cp "$ENV_FILE" "$BACKUP"
PID=""
cleanup() {
    [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null && wait "$PID" 2>/dev/null
    cp "$BACKUP" "$ENV_FILE"
    echo "[batch] .env put back from $BACKUP"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

total=$(grep -c . "$ITEMS")
for spec in "$@"; do
    label=${spec%%=*}; rest=${spec#*=}; gguf=${rest%%|*}; extra=""
    [ "$rest" != "$gguf" ] && extra=${rest#*|}
    out="$OUT_DIR/answers-$label-$TAG.jsonl"
    if [ -s "$out" ] && [ "$(grep -c . "$out")" -ge "$total" ]; then
        echo "[batch] $label: already answered, skipped"; continue
    fi
    [ -s "$gguf" ] || { echo "[batch] $label: $gguf not found, skipped"; continue; }

    cp "$BACKUP" "$ENV_FILE"
    sed -i '/^EULLM__MODEL_OVERRIDE=/d;/^EULLM__EXTRA_ARGS=/d' "$ENV_FILE"
    echo "EULLM__MODEL_OVERRIDE=$gguf" >> "$ENV_FILE"
    [ -n "$extra" ] && echo "EULLM__EXTRA_ARGS=$extra" >> "$ENV_FILE"

    echo "[batch] $(date +%H:%M) $label: starting RAG Enterprise ($gguf${extra:+, $extra})"
    (cd "$ENGINE_DIR" && exec ./i3k-rag-engine) > "$OUT_DIR/engine-$label.log" 2>&1 < /dev/null &
    PID=$!
    up=0
    for _ in $(seq $((START_TIMEOUT / 5))); do
        kill -0 "$PID" 2>/dev/null || break
        if curl -fs "$URL/health" >/dev/null 2>&1; then up=1; break; fi
        sleep 5
    done
    if [ $up = 1 ]; then
        python3 "$EVAL" ask --url "$URL" --label "$label-$TAG" --items "$ITEMS" --out "$out" \
            || echo "[batch] $label: ask stopped early (see above); its answers so far are kept"
    else
        echo "[batch] $label: RAG Enterprise did not come up — last lines of its log:"
        tail -5 "$OUT_DIR/engine-$label.log"
    fi
    kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""
    for _ in $(seq 24); do curl -fs "$URL/health" >/dev/null 2>&1 || break; sleep 5; done
    echo "[batch] $(date +%H:%M) $label: done ($( [ -s "$out" ] && grep -c . "$out" || echo 0)/$total answers)"
done
