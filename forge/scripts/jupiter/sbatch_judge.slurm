#!/bin/bash
# The held-out exam graded by a large model (deliverable 6). The keyword
# score of legal_eval.py rewards length: on 16-10-2026 Qwen3-4B-Instruct-2507
# scored 0.260 on the 217 questions while ending its turn on 55 of them,
# against 0.181 for our 8B, which ended 212. judge_answers.py has
# Qwen3-30B-A3B-Instruct-2507 grade every answer against the text of the
# article (correct / partial / wrong, rubric v3, as on Leonardo), whatever
# its length.
#
# Grades every norm-exam-answers.jsonl under $FORGE_RUN_DIR/sft*/ (one a
# model, the reference included), four graders at once, one a GPU. Quiet:
# totals only, never a question (the exam stays held out).
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/judge-%j.out" \
#         forge/scripts/jupiter/sbatch_judge.slurm
#
# Needs the grader in the offline cache (prefetch_models.py --models
# Qwen/Qwen3-30B-A3B-Instruct-2507, 61 GB, on a login node). Resumable: a
# model already graded is skipped, so a rerun grades only the new ones.
# Whole node: 4 GPU-hours an hour.

#SBATCH --job-name=forge-judge
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
RUNS="$FORGE_RUN_DIR"
JUDGE="${FORGE_JUDGE_DIR:-$RUNS/judge}"
GRADER="${FORGE_JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$JUDGE/answers" "$JUDGE/graded-v3" "$RUNS/power"

echo "=== Forge judge: job $JOB on $(hostname), $(date -Is) ==="
python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$GRADER" ||
    { echo "[!!] $GRADER is not in the offline cache: prefetch_models.py on a login node"; exit 1; }

# answers-<set>-<model>.jsonl: judge_answers.py takes the label from the name.
for f in "$RUNS"/sft*/*/norm-exam-answers.jsonl; do
    [ -s "$f" ] || continue
    model=$(basename "$(dirname "$f")") set=$(basename "$(dirname "$(dirname "$f")")")
    ln -sf "$f" "$JUDGE/answers/answers-$set-$model.jsonl"
done
FILES=("$JUDGE"/answers/answers-*.jsonl)
[ -e "${FILES[0]}" ] || { echo "[!!] no norm-exam-answers.jsonl under $RUNS/sft*/"; exit 1; }
echo "  ${#FILES[@]} answer sets, grader $GRADER, rubric v3, into $JUDGE/graded-v3"

POWER="$RUNS/power/judge-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!

# The sets dealt round-robin to the four GPUs, each grader with its own CSV.
PIDS=()
for g in 0 1 2 3; do
    mine=()
    for i in "${!FILES[@]}"; do [ $((i % 4)) -eq "$g" ] && mine+=("${FILES[$i]}"); done
    [ "${#mine[@]}" -gt 0 ] || continue
    CUDA_VISIBLE_DEVICES=$g taskset -c "$((72 * g))-$((72 * g + 71))" \
        python "$EULLM_REPO/forge/scripts/judge_answers.py" "${mine[@]}" \
        --model "$GRADER" --rubric v3 --quiet --graded-dir "$JUDGE/graded-v3" \
        --csv "graded-gpu$g.csv" >"$JUDGE/judge-$JOB-gpu$g.log" 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=$?; done

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
grep -h '^\[judge\]' "$JUDGE"/judge-"$JOB"-gpu*.log | sed 's/^/  /'
echo
echo "--- graded (label, items, correct, partial, wrong, unparsed, score)"
awk -F, 'FNR > 1 { print $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 }' \
    "$JUDGE"/graded-v3/graded-gpu*.csv 2>/dev/null | sort -t, -k7 -rn |
    { if command -v column >/dev/null; then column -s, -t; else cat; fi; } | sed 's/^/  /'
echo
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"
