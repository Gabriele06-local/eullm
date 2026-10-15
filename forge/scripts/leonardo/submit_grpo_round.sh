#!/usr/bin/env bash
# One GRPO round, start to finish, in one submission: train, merge, package,
# evaluate. Nothing to time by hand afterwards.
#
#   bash forge/scripts/leonardo/submit_grpo_round.sh jg-q35-9b-s750 \
#       $WORK/eullm_runs/opd/opd-q38t-q35-9b-r2n/merged 5 750
#
#   NAME          run name: $WORK/eullm_runs/grpo/NAME, jobs eullm-*-NAME
#   START         merged model the policy starts from
#   LINKS         2 h GRPO links (each resumes from the last checkpoint)
#   MAX_STEPS     grpo_train.py --max-steps
#
# What it queues, each step waiting on the one it needs:
#
#   1. LINKS GRPO links (sbatch_grpo.slurm, judge as reward), afterany;
#   2. one packaging job on the serial partition: the merge of the adapter
#      into START, then the GGUF Q4_K_M of the merged model, its F16 removed
#      after ($WORK is the constraint, not CPU time). One job and not two:
#      lrd_all_serial takes 10 queued jobs a user, and on 2026-10-15 the
#      fourth round of the day was refused there half-submitted;
#   4. the case-law exam on the bf16 weights (one GPU), after the packaging:
#      a GRPO round on statutes must not cost the case law;
#   5. the statute exam round on the development set with the GGUF and the
#      released models' hybrid retrieval, graded under rubric v3 into the
#      graded directory every other model is in, with a report against
#      BASELINES (sbatch_exam_round.slurm), after the packaging.
#
# A round takes 3 places on the serial partition (packaging, exam round and
# the report the exam round queues later); with fewer free it is refused
# before anything is submitted. LINKS=0 queues only steps 2-5, after AFTER
# (afterany): the packaging and exams of a chain already queued.
#
# Written on 2026-10-15: the first judge-reward rounds were submitted as
# training plus merge only, the GGUF and the exams were forgotten, and the
# allocation stood idle until somebody asked.
#
# Settable:
#   GRPO_PROMPTS   prompts (default $WORK/eullm_runs/grpo/prompts-judge.jsonl)
#   GRPO_JUDGE_GGUF  judge (default Qwen3-30B-A3B-Instruct-2507 Q8_0 in $WORK/gguf)
#   GRPO_LR        learning rate (default 1e-5)
#   GRPO_EXTRA     more grpo_train.py options
#   ROUND_BASELINES  labels the report compares against (default
#                  "opd-v4abs-4b-q4-hyb", the published 4B)
#   AFTER          a job id the first link waits for (afterok), e.g. the
#                  merge of the round START comes from; START is then not
#                  checked, since it does not exist yet
#   NO_CDS=1       skip the case-law exam
#   SERIAL_LIMIT   jobs a user may queue on lrd_all_serial (default 10)
#
# Idempotent: a round whose GRPO jobs are already queued is not submitted
# again, and one whose adapter is written skips the training.

set -uo pipefail

[ $# -eq 4 ] || { sed -n '2,15p' "$0" >&2; exit 2; }
NAME=$1 START=$2 LINKS=$3 STEPS=$4
L="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$L/../../.." && pwd)"
G="$WORK/eullm_runs/grpo"
E="$WORK/eval"
OUT="$G/$NAME"
PROMPTS="${GRPO_PROMPTS:-$G/prompts-judge.jsonl}"
JUDGE="${GRPO_JUDGE_GGUF:-$WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf}"
LR="${GRPO_LR:-1e-5}"
BASELINES="${ROUND_BASELINES:-opd-v4abs-4b-q4-hyb}"
mkdir -p "$G/logs"

fail=0
check() { [ -e "$1" ] || { echo "[round] missing $2: $1" >&2; fail=1; }; }
[ -n "${AFTER:-}" ] || check "$START/config.json" "start model"
check "$PROMPTS" "prompts"
check "$JUDGE" "judge GGUF"
case "$LINKS$STEPS" in *[!0-9]*|"") echo "[round] LINKS and MAX_STEPS are numbers" >&2; fail=1 ;; esac
[ "$LINKS" = 0 ] && [ -z "${AFTER:-}" ] && { echo "[round] LINKS=0 needs AFTER" >&2; fail=1; }
[ "$fail" = 0 ] || exit 1
if [ "$LINKS" != 0 ] && [ -n "$(squeue --me -h -n "eullm-grpo-$NAME" -o %i 2>/dev/null)" ]; then
    echo "[round] $NAME: GRPO already queued, nothing submitted"
    exit 0
fi
serial=$(squeue --me -h -p lrd_all_serial -o %i 2>/dev/null | wc -l)
if [ $((serial + 3)) -gt "${SERIAL_LIMIT:-10}" ]; then
    echo "[round] $NAME: $serial jobs queued on lrd_all_serial, a round needs 3 of" \
         "${SERIAL_LIMIT:-10}: nothing submitted, try when some have ended" >&2
    exit 1
fi

sub() {  # sbatch that prints the job id; fails (and says so) when sbatch does
    local id
    id=$(sbatch --parsable "$@") && [ -n "$id" ] ||
        { echo "[round] sbatch failed: $*" >&2; return 1; }
    echo "${id%%;*}"
}
ENVSH="source $REPO/forge/scripts/leonardo/env.sh"

dep="${AFTER:+afterok:$AFTER}"
[ "$LINKS" = 0 ] && dep="afterany:$AFTER"
last=""
if [ "$LINKS" = 0 ]; then
    echo "[round] $NAME: packaging and exams only, after job $AFTER"
elif [ -f "$OUT/adapter/adapter_config.json" ]; then
    echo "[round] $NAME: adapter already written, training skipped"
else
    for i in $(seq 1 "$LINKS"); do
        last=$(GRPO_MODEL="$START" GRPO_PROMPTS="$PROMPTS" GRPO_OUT="$OUT" GRPO_JUDGE_GGUF="$JUDGE" \
               GRPO_ARGS="--lr $LR --max-steps $STEPS ${GRPO_EXTRA:-}" \
               sub -J "eullm-grpo-$NAME" -o "$G/logs/%x-%j.out" ${dep:+--dependency=$dep} \
                   "$L/sbatch_grpo.slurm") || exit 1
        echo "[round] grpo link $i/$LINKS: $last"
        dep="afterany:$last"
    done
fi

pkg=$(sub -J "eullm-package-$NAME" -p lrd_all_serial -c 4 --mem=30G -t 04:00:00 \
    -o "$G/logs/%x-%j.out" ${dep:+--dependency=$dep} \
    --wrap "$ENVSH && PYTHONPATH=$REPO/forge python -c 'import sys; from eullm_forge.identity import merge_identity_adapter as m; m(*sys.argv[1:])' $START $OUT/adapter $OUT/merged && GGUF_NAME=$NAME LCPP_SKIP_UPDATE=1 bash $REPO/forge/scripts/quantize_to_gguf.sh $OUT/merged $OUT/gguf && rm -f $OUT/gguf/$NAME-f16.gguf") || exit 1
echo "[round] merge + GGUF: $pkg"

if [ "${NO_CDS:-0}" != 1 ]; then
    cds=$(CE_LABEL="$NAME-bf16" CE_MODEL="$OUT/merged" \
          sub -J eullm-cds-exam --dependency="afterok:$pkg" "$L/sbatch_cds_exam.slurm") || exit 1
    echo "[round] case-law exam: $cds"
fi

exam=$(ROUND_MODELS="$NAME-q4=$OUT/merged" ROUND_GGUF="$NAME-q4=$OUT/gguf/$NAME-q4_k_m.gguf" \
       ROUND_PAIRS="$PROMPTS" ROUND_ITEMS="$E/norm-exam-devbig.jsonl" ROUND_OUT="$E/devbig-answers" \
       ROUND_OPEN_ONLY=1 ROUND_EMBEDDER=Qwen/Qwen3-Embedding-0.6B \
       ROUND_RERANKER=Qwen/Qwen3-Reranker-0.6B ROUND_OPEN_TAG=hyb ROUND_EXAM_LINKS=2 \
       ROUND_JUDGE_ARGS="--rubric v3 --graded-dir $E/devbig-graded-v3" \
       ROUND_REPORT="$BASELINES" ROUND_REPORT_DIR="$E/devbig-graded-v3" \
       ROUND_REPORT_FILE="report-$NAME.txt" \
       sub -J "eullm-exam-round-$NAME" --dependency="afterok:$pkg" "$L/sbatch_exam_round.slurm") || exit 1
echo "[round] statute exam round: $exam (report: $E/devbig-graded-v3/report-$NAME.txt)"
echo "[round] $NAME: merged model will be $OUT/merged (job $pkg)"
