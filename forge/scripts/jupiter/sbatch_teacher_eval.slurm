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
# EVAL_DEQUANTIZE=1 loads the FP8 weights as BF16 (~240 GB) instead, if the
# FP8 path does not run here. EVAL_THINK=1 lets it reason first
# (reasoning_effort "high", 4,000 new tokens by default); the reasoning is
# kept apart in the answers file and only the answer is graded.
#
# Writes $FORGE_RUN_DIR/teacher-eval/answers-<set>-<label>[-think].jsonl and
# a row in teacher-eval.csv; the log holds totals and times only, never a
# question (--quiet). The question files may be private: they stay on the
# project storage, never in the repository.

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
OUT="$FORGE_RUN_DIR/teacher-eval"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$FORGE_RUN_DIR/power" "$FORGE_RUN_DIR/retrieval-cache"
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
if [ "${EVAL_DEQUANTIZE:-0}" = 1 ]; then
    [ "$WEIGHTS" != "$MODEL" ] || { echo "[!!] EVAL_DEQUANTIZE needs the copy in memory"; exit 1; }
    python - "$WEIGHTS/config.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["quantization_config"]["dequantize"] = True
json.dump(c, open(p, "w"), indent=2)
print("  FP8 weights loaded as BF16 (dequantize)")
PY
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
