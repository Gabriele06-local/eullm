#!/bin/bash
# The large held-out trap test (make_trap_test.py): hundreds of false
# premises and their true twins, by kind, drawn from the legislation on
# articles neither the exams nor the premise pairs trained on use. Built
# once into $FORGE_DATA_DIR/eval/trap-test.jsonl; an existing file is kept,
# so every model is asked the same questions. One GPU of a whole node for the
# retrieval models: minutes.
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/trap-test-%j.out" \
#         forge/scripts/jupiter/sbatch_trap_test.slurm
#
# submit_trap_test.sh runs it, then asks the models and grades them. The log
# carries counts only, never a question.

#SBATCH --job-name=forge-trap-test
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
OUT="${TT_OUT:-$FORGE_DATA_DIR/eval/trap-test.jsonl}"
P="${FORGE_PRIVATE_EVAL:-$FORGE_HOME/private-eval}"
read -r -a NORMS <<<"${TT_NORMS:-$(ls "$FORGE_DATA_DIR"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')}"
read -r -a EXCLUDE <<<"${TT_EXCLUDE:-$(ls "$FORGE_DATA_DIR"/eval/norm-exam*.jsonl | tr '\n' ' ') $P/practical-dev.jsonl $P/lexia-trappole.jsonl}"
read -r -a TRAINED <<<"${TT_TRAINED:-$FORGE_DATA_DIR/premise/sft-steps.jsonl}"

echo "=== Forge trap test: job ${SLURM_JOB_ID:-local} on $(hostname), $(date -Is) ==="
if [ -s "$OUT" ]; then
    echo "[ok] $OUT holds $(wc -l <"$OUT") questions, kept"
    exit 0
fi
fail=0
for f in "${NORMS[@]}" "${EXCLUDE[@]}" "${TRAINED[@]}"; do
    [ -s "$f" ] || { echo "[!!] missing $f"; fail=1; }
done
[ "$fail" = 0 ] || exit 1
mkdir -p "$(dirname "$OUT")" "$FORGE_RUN_DIR/retrieval-cache"
CUDA_VISIBLE_DEVICES=0 python "$EULLM_REPO/forge/scripts/make_trap_test.py" \
    --norms "${NORMS[@]}" --exclude-exam "${EXCLUDE[@]}" --exclude-trained "${TRAINED[@]}" \
    --per-code "${TT_PER_CODE:-30}" --seed "${TT_SEED:-1}" \
    --embedder Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \
    --retrieval-cache "$FORGE_RUN_DIR/retrieval-cache" --out "$OUT"
RC=$?
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"
