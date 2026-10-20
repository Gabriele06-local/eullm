#!/bin/bash
# A candidate teacher on the legal-it questions, open book, in the format of
# the exams on Leonardo: legal_eval.py with hybrid retrieval (BM25 + Qwen3
# embedder + Qwen3 reranker), no missing-article note, answers written for
# the judge, which runs on Leonardo with the other models'. One question set
# and one mode a job:
#
#     eullm && EVAL_ITEMS=<set.jsonl> EVAL_SET=practical-dev [EVAL_THINK=1] \
#         sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/teacher-eval-%j.out" \
#         forge/scripts/jupiter/sbatch_teacher_eval.slurm
#
# EVAL_MODEL defaults to Mistral Small 4 (mistralai/Mistral-Small-4-119B-2603,
# Apache 2.0; 119B MoE, its checkpoint FP8, ~120 GB), spread over the node's
# four GH200 and loaded from a copy in the node's memory (stage_model.sh).
# Its FP8 weights do not run here as they are: transformers' FP8 path needs
# kernels from the Hub, out of a compute node's reach, and its dequantize on
# loading left the fused experts in FP8 (jobs 2418651-2418654). So the model
# is converted once to BF16 on a login node (dequantize_fp8.py, ~240 GB) and
# EVAL_MODEL points at that directory; too big for the node's memory, it is
# read from the project storage, in large tensors (the experts are fused). EVAL_THINK=1 lets it reason first
# (reasoning_effort "high", 4,000 new tokens by default); the reasoning is
# kept apart in the answers file and only the answer is graded.
#
# Writes $FORGE_RUN_DIR/teacher-eval/answers-<set>-<label>[-think].jsonl and
# a row in teacher-eval.csv; the log holds totals and times only, never a
# question (--quiet). The question files may be private: they stay on the
# project storage, never in the repository nor anywhere online, and so do
# the answers files, which repeat the questions; the output directory
# (EVAL_OUT) is made readable by the user alone.

#SBATCH --job-name=forge-teacher-eval
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
MODEL="${EVAL_MODEL:-mistralai/Mistral-Small-4-119B-2603}"
LABEL="${EVAL_LABEL:-mistral-small-4}"
ITEMS="${EVAL_ITEMS:?EVAL_ITEMS: the question set (JSONL)}"
SET="${EVAL_SET:?EVAL_SET: a short name for the set, for the file names}"
THINK="${EVAL_THINK:-0}"
MAX_NEW="${EVAL_MAX_NEW:-$([ "$THINK" = 1 ] && echo 4000 || echo 400)}"
BATCH="${EVAL_BATCH:-16}"
NORMS="${EVAL_NORMS:-$(ls "$FORGE_DATA_DIR"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')}"
EMBEDDER="${EVAL_EMBEDDER:-Qwen/Qwen3-Embedding-0.6B}"
RERANKER="${EVAL_RERANKER:-Qwen/Qwen3-Reranker-0.6B}"
OUT="${EVAL_OUT:-$FORGE_RUN_DIR/teacher-eval}"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$FORGE_RUN_DIR/power" "$FORGE_RUN_DIR/retrieval-cache"
# The answers files carry the questions, and some sets are private (the
# LexIA traps): the directory and everything written in it are the user's
# alone, without the ACLs the project storage hands down.
umask 077
chmod 700 "$OUT" && { setfacl -b "$OUT" 2>/dev/null || true; }
TAG="$LABEL$([ "$THINK" = 1 ] && echo -think)"
ANSWERS="$OUT/answers-$SET-$TAG.jsonl"

echo "=== teacher eval: job $JOB on $(hostname), $(date -Is) ==="
echo "  $MODEL as $TAG on $SET ($(wc -l <"$ITEMS") questions), think=$THINK, max new $MAX_NEW, batch $BATCH"
echo "  open book, no note: $(wc -w <<<"$NORMS") norm files, $EMBEDDER + $RERANKER"
python -c "import transformers; print('  transformers', transformers.__version__)"
for m in "$MODEL" "$EMBEDDER" "$RERANKER"; do
    python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$m" 2>/dev/null ||
        { echo "[!!] $m is not in the offline cache: download it on a login node"; exit 1; }
done

# shellcheck source=stage_model.sh
source "$EULLM_REPO/forge/scripts/jupiter/stage_model.sh"
t0=$(date +%s)
WEIGHTS=$(stage_model "$MODEL")
if [ "${EVAL_DEQUANTIZE:-1}" = 1 ] && grep -q '"quant_method": "fp8"' "$(stage_source "$WEIGHTS")/config.json" 2>/dev/null; then
    # Transformers runs FP8 weights through kernels fetched from the Hugging
    # Face Hub (the `kernels` package), which a compute node cannot reach:
    # jobs 2414120 and 2414122 stopped there on 19-10-2026. Dequantized on
    # loading, the weights are BF16 (~240 GB over the four GPUs). A view of
    # the model with its own config.json: the weights are linked, not copied.
    src=$(stage_source "$WEIGHTS") || exit 1
    view="$STAGE_DIR/dequantized"
    mkdir -p "$view"
    for f in "$src"/*; do ln -sf "$(readlink -f "$f")" "$view/$(basename "$f")"; done
    rm -f "$view/config.json"
    python - "$src/config.json" "$view/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["quantization_config"]["dequantize"] = True
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
    WEIGHTS="$view"
    echo "  FP8 weights dequantized to BF16 on loading (EVAL_DEQUANTIZE=0 keeps FP8, which needs the kernels package)"
fi
echo "  weights from $WEIGHTS, ready in $(( $(date +%s) - t0 )) s"

POWER="$FORGE_RUN_DIR/power/teacher-eval-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!

THINK_ARGS=()
[ "$THINK" = 1 ] && THINK_ARGS=(--think)
t1=$(date +%s)
# shellcheck disable=SC2086
python "$EULLM_REPO/forge/scripts/legal_eval.py" "$WEIGHTS" --label "$SET-$TAG" \
    --items "$ITEMS" --norms $NORMS --embedder "$EMBEDDER" --reranker "$RERANKER" \
    --retrieval-cache "$FORGE_RUN_DIR/retrieval-cache" --no-note "${THINK_ARGS[@]}" \
    --max-new-tokens "$MAX_NEW" --batch-size "$BATCH" \
    --answers "$ANSWERS" --csv "$OUT/teacher-eval.csv" --quiet
RC=$?
echo "  legal_eval: exit $RC, $(( $(date +%s) - t1 )) s (model load, retrieval and generation)"

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
[ -s "$ANSWERS" ] && echo "  answers: $(wc -l <"$ANSWERS") in $ANSWERS"
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"
