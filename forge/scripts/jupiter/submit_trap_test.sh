#!/usr/bin/env bash
# The trap test, unattended, from a login node: the test built
# (sbatch_trap_test.slurm), asked of each model open book as the product asks
# (sbatch_teacher_eval.slurm), then graded by the judge on JUPITER
# (sbatch_judge.slurm with JUDGE_ANSWERS_DIR). Each step waits for the one
# before. The judge-free grades by kind come from score_trap_test.py on a
# login node, once the answers are written:
#
#     eullm && bash forge/scripts/jupiter/submit_trap_test.sh
#
# TRAP_PARA=1 asks the trap test rewritten as people ask instead
# (paraphrase_trap_test.py through sbatch_gen_instruct.slurm,
# FORGE_GEN_KIND=paraphrase): the trap test itself must be built already.
#
# TRAP_MODELS names the models ("label=merged dir ..."); by default the
# candidate (the distilled Qwen3.5-9B with the false premises), the same
# without them, the 4B with them, the 9B not distilled and Ministral.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
# shellcheck source=sbatch_retry.sh
source "$HERE/sbatch_retry.sh"
REPO="$(cd "$HERE/../../.." && pwd)"
R="$FORGE_RUN_DIR" LOGS="$FORGE_RUN_DIR/logs"
TEST="${TT_OUT:-$FORGE_DATA_DIR/eval/trap-test.jsonl}"
SET=trap-test
if [ "${TRAP_PARA:-0}" = 1 ]; then
    [ -s "$TEST" ] || { echo "[!!] no trap test at $TEST: submit_trap_test.sh without TRAP_PARA first"; exit 1; }
    TEST="$FORGE_DATA_DIR/eval/trap-test-para.jsonl" SET=trap-test-para
fi
LIBS="${FORGE_EXTRA_LIBS:-$FORGE_HOME/libs-qwen35}"
SYSTEM_FILE="$REPO/forge/model_cards/legal-it-8b/system-prompt.txt"
MODELS="${TRAP_MODELS:-dqp-d-q9b=$R/sft-dqp/d-q9b/merged dq35f-f-q9b=$R/sft-dq35f/f-q9b/merged dqp-d-q4b=$R/sft-dqp/d-q4b/merged ob1-q9b=$R/sft-ob1/p-q9b/merged ob1-m8i=$R/sft-ob1/p-m8i/merged}"
bad=0
for m in $MODELS; do
    [ -f "${m#*=}/config.json" ] && echo "[ok] ${m%%=*}: ${m#*=}" || { echo "[!!] no model at ${m#*=}"; bad=1; }
done
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }
wait_for_slurm || exit 1
mkdir -p "$LOGS"
cd "$REPO" || exit 1

if [ "$SET" = trap-test-para ]; then
    tt=$(export FORGE_GEN_KIND=paraphrase; sbatch_retry --parsable --account "$SBATCH_ACCOUNT" \
        --output="$LOGS/gen-instruct-%j.out" forge/scripts/jupiter/sbatch_gen_instruct.slurm) ||
        { echo "[!!] sbatch of the rewriting failed"; exit 1; }
else
    tt=$(sbatch_retry --parsable --account "$SBATCH_ACCOUNT" \
        --output="$LOGS/trap-test-%j.out" forge/scripts/jupiter/sbatch_trap_test.slurm) ||
        { echo "[!!] sbatch of the trap test failed"; exit 1; }
fi
tt="${tt%%;*}"
echo "[ok] the $SET built by job $tt into $TEST"
ids=""
for m in $MODELS; do
    label="${m%%=*}" dir="${m#*=}" sys=""
    [[ "$label" == *m8i* ]] && sys="$SYSTEM_FILE"
    j=$(export EVAL_MODEL="$dir" EVAL_LABEL="$label" EVAL_SYSTEM_FILE="$sys" \
               EVAL_ITEMS="$TEST" EVAL_SET="$SET" FORGE_EXTRA_LIBS="$LIBS"
        sbatch_retry --parsable --account "$SBATCH_ACCOUNT" --time=01:00:00 \
        --dependency="afterok:$tt" --output="$LOGS/teacher-eval-%j.out" \
        forge/scripts/jupiter/sbatch_teacher_eval.slurm) || { echo "[!!] sbatch of $label failed"; exit 1; }
    ids="$ids:${j%%;*}"
    echo "[ok] $label: job ${j%%;*}"
done
j=$(export JUDGE_ANSWERS_DIR="$R/teacher-eval"; sbatch_retry --parsable --account "$SBATCH_ACCOUNT" \
    --dependency="afterany$ids" --output="$LOGS/judge-%j.out" forge/scripts/jupiter/sbatch_judge.slurm) ||
    { echo "[!!] sbatch of the judge failed"; exit 1; }
echo "[ok] graded by job ${j%%;*}"
echo "     then, on a login node: python forge/scripts/score_trap_test.py $TEST $R/teacher-eval/answers-$SET-*.jsonl"
