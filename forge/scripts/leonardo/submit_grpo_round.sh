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
#   5. the statute exam on the development set with the GGUF and the released
#      models' hybrid retrieval (two links), graded under rubric v3 into the
#      graded directory every other model is in; and the practical exam
#      ($WORK/eval/practical-dev.jsonl, when it exists) the same way, into
#      $WORK/eval/practical-answers; and the statute exam once more without
#      the note on a missing article (label <NAME>-q4-hybnn), as RAG
#      Enterprise asks. Compare with compare_graded.py.
#
# A round takes 1 place on the serial partition (the packaging); with none
# free it is refused before anything is submitted. LINKS=0 queues only steps 2-5, after AFTER
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
#   AFTER          job id(s) the first link waits for (afterok; "a:b" for
#                  two), e.g. the packaging of the round START comes from or
#                  the job that writes GRPO_PROMPTS; START and the prompts are
#                  then not checked, since they do not exist yet
#   NO_CDS=1       skip the case-law exam
#   NO_PRACTICAL=1 skip the practical exam
#   NO_LEXIA=1     skip the trap questions ($WORK/eval/lexia/lexia-trappole.jsonl)
#   PACKAGED       the job id of a packaging already queued for NAME (LINKS=0):
#                  only the exams are queued, after it, and no serial place is
#                  taken -- to move a round's exams off the serial partition
#   SERIAL_LIMIT   jobs a user may queue on lrd_all_serial (default 10)
#   GRPO_MERGED_ROOT  where the merged bf16 weights go (default the project's
#                  fast scratch, .../grpo-merged/NAME, behind a symlink at
#                  NAME/merged; unused when the scratch is not there)
#
# Idempotent: a round whose GRPO jobs or packaging are already queued, or
# whose GGUF is already written, is not submitted again; one whose adapter is
# written skips the training. (On 2026-10-16 a rerun over a finished round
# queued its packaging and exams a second time.)
#
# The merged weights are a day's file: the case-law exam reads them, then
# nothing does, and the adapter and START make them again in minutes. At
# 8-18 GB a round they filled $WORK on 2026-10-17, so they are written to the
# fast scratch (purged when unused); the adapter and the GGUF stay on $WORK.
# The scratch has a 1 TB quota of its own, and on 2026-10-24 it was full with
# 33 rounds' merged weights: so each new round removes those of the rounds
# already packaged, except the newest GRPO_KEEP_MERGED (default 6: with an
# 8B and a 4B lane, the two new rounds, their starts and the two before, the
# fallback if a new round turns out worse), the START of this round and of
# every round not packaged yet (each round writes its START to NAME/start),
# and any round with a job still queued. Only this round's START was kept at
# first, and on 2026-10-25 the 4B round submitted second removed the merged
# weights of r2l, the START of the 8B round submitted a minute before. The
# symlink NAME/merged stays, pointing where the weights were: a round removed
# is merged again there, with its ancestors if they are gone too, by
# forge/scripts/remerge_round.py.

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
mkdir -p "$G/logs"

fail=0
check() { [ -e "$1" ] || { echo "[round] missing $2: $1" >&2; fail=1; }; }
[ -n "${AFTER:-}${PACKAGED:-}" ] || check "$START/config.json" "start model"
# with AFTER, the prompts may be what that job writes (sbatch_practical_prompts.slurm)
[ -n "${AFTER:-}${PACKAGED:-}" ] || check "$PROMPTS" "prompts"
check "$JUDGE" "judge GGUF"
case "$LINKS$STEPS" in *[!0-9]*|"") echo "[round] LINKS and MAX_STEPS are numbers" >&2; fail=1 ;; esac
[ "$LINKS" = 0 ] && [ -z "${AFTER:-}${PACKAGED:-}" ] &&
    { echo "[round] LINKS=0 needs AFTER or PACKAGED" >&2; fail=1; }
[ "$fail" = 0 ] || exit 1
if [ "$LINKS" != 0 ] && [ -n "$(squeue --me -h -n "eullm-grpo-$NAME" -o %i 2>/dev/null)" ]; then
    echo "[round] $NAME: GRPO already queued, nothing submitted"
    exit 0
fi
if [ -z "${PACKAGED:-}" ] && [ -s "$OUT/gguf/$NAME-q4_k_m.gguf" ]; then
    echo "[round] $NAME: already packaged ($OUT/gguf), nothing submitted"
    exit 0
fi
if [ -z "${PACKAGED:-}" ] && [ -n "$(squeue --me -h -n "eullm-package-$NAME" -o %i 2>/dev/null)" ]; then
    echo "[round] $NAME: packaging already queued, nothing submitted"
    exit 0
fi
serial=$(squeue --me -h -p lrd_all_serial -o %i 2>/dev/null | wc -l)
if [ -z "${PACKAGED:-}" ] && [ $((serial + 1)) -gt "${SERIAL_LIMIT:-10}" ]; then
    echo "[round] $NAME: $serial jobs queued on lrd_all_serial, a round needs 1 of" \
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

MERGED_ROOT="${GRPO_MERGED_ROOT:-/leonardo_scratch/fast/$(basename "$WORK")/grpo-merged}"
prune_merged() {  # the merged weights of packaged rounds nothing needs any more
    local keep="${GRPO_KEEP_MERGED:-6}" starts queued d r f n=0
    # the START of this round and of every round still on its way to a GGUF
    starts="$(readlink -f "$START" 2>/dev/null)"
    for f in "$G"/*/start; do
        [ -f "$f" ] || continue
        r="$(basename "$(dirname "$f")")"
        [ -s "$G/$r/gguf/$r-q4_k_m.gguf" ] && continue
        starts="$starts"$'\n'"$(readlink -f "$(cat "$f")" 2>/dev/null)"
    done
    queued="$(squeue --me -h -o %j 2>/dev/null)"
    for d in $(ls -td "$MERGED_ROOT"/*/ 2>/dev/null); do
        d="$(readlink -f "$d")" r="$(basename "$d")" n=$((n + 1))
        [ "$n" -gt "$keep" ] && [ "$r" != "$NAME" ] || continue
        printf '%s\n' "$starts" | grep -qxF -- "$d" && continue
        [ -s "$G/$r/gguf/$r-q4_k_m.gguf" ] || continue
        printf '%s\n' "$queued" | grep -q -- "-$r\$" && continue
        rm -rf "$d"
        echo "[round] merged weights of $r removed from the scratch (adapter and GGUF kept)"
    done
}
# Only when the scratch is filling up: on 2026-10-26, at 68%, the cleanup
# removed jpn14's weights from under a held-out exam whose jobs carry no
# round's name. Below GRPO_PRUNE_AT percent (default 80) nothing is removed.
used=$(df -P "$MERGED_ROOT" 2>/dev/null | awk 'NR == 2 {sub("%", "", $5); print $5}')
[ -z "${PACKAGED:-}" ] && [ -d "$MERGED_ROOT" ] && [ "${used:-100}" -ge "${GRPO_PRUNE_AT:-80}" ] &&
    prune_merged
[ -z "${PACKAGED:-}" ] && mkdir -p "$OUT" && printf '%s\n' "$START" > "$OUT/start"
if [ -z "${PACKAGED:-}" ] && [ ! -e "$OUT/merged" ] && [ -d "$(dirname "$MERGED_ROOT")" ] &&
   mkdir -p "$MERGED_ROOT/$NAME" "$OUT"; then
    ln -s "$MERGED_ROOT/$NAME" "$OUT/merged" &&
        echo "[round] merged weights on the scratch: $MERGED_ROOT/$NAME"
fi

dep="${AFTER:+afterok:$AFTER}"
[ "$LINKS" = 0 ] && [ -n "${AFTER:-}" ] && dep="afterany:$AFTER"
last=""
if [ -n "${PACKAGED:-}" ]; then
    echo "[round] $NAME: exams only, after packaging job $PACKAGED"
elif [ "$LINKS" = 0 ]; then
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

if [ -n "${PACKAGED:-}" ]; then
    pkg=$PACKAGED
else
    pkg=$(sub -J "eullm-package-$NAME" -p lrd_all_serial -c 4 --mem=30G -t 04:00:00 \
        -o "$G/logs/%x-%j.out" ${dep:+--dependency=$dep} \
        --wrap "$ENVSH && PYTHONPATH=$REPO/forge python -c 'import sys; from eullm_forge.identity import merge_identity_adapter as m; m(*sys.argv[1:])' $START $OUT/adapter $OUT/merged && GGUF_NAME=$NAME LCPP_SKIP_UPDATE=1 bash $REPO/forge/scripts/quantize_to_gguf.sh $OUT/merged $OUT/gguf && rm -f $OUT/gguf/$NAME-f16.gguf") || exit 1
    echo "[round] merge + GGUF: $pkg"
fi

if [ "${NO_CDS:-0}" != 1 ]; then
    cds=$(CE_LABEL="$NAME-bf16" CE_MODEL="$OUT/merged" \
          sub -J eullm-cds-exam --dependency="afterok:$pkg" "$L/sbatch_cds_exam.slurm") || exit 1
    echo "[round] case-law exam: $cds"
fi

# The exams go to the GPU partition directly, each graded by its own judge
# job: the round helper (sbatch_exam_round.slurm) would take two more places
# on the serial partition, and a day's rounds did not fit (2026-10-15).
NORMS="$WORK/norms/legislazione_*.chunks.jsonl"
exam() {  # items out-dir graded-dir links tag [legal_eval options] -> id of the judge job
    local items=$1 out=$2 graded=$3 links=$4 tag=$5 extra=${6:-} d="afterok:$pkg" x i
    mkdir -p "$out"
    for i in $(seq 1 "$links"); do
        x=$(EXAM_MODELS="$NAME-q4=$OUT/merged" EXAM_GGUF="$NAME-q4=$OUT/gguf/$NAME-q4_k_m.gguf" \
            EXAM_ITEMS="$items" EXAM_OUT="$out" EXAM_NORMS="$NORMS" EXAM_OPEN_ONLY=1 \
            EXAM_OPEN_TAG="$tag" EXAM_OPEN_ARGS="$extra" EXAM_EMBEDDER=Qwen/Qwen3-Embedding-0.6B \
            EXAM_RERANKER=Qwen/Qwen3-Reranker-0.6B \
            sub -J "eullm-exam-$NAME" --dependency="$d" "$L/sbatch_norm_exam.slurm") || return 1
        d="afterany:$x"
    done
    sub --dependency="$d" --export="ALL,JUDGE_DIR=$out,JUDGE_QUIET=1,JUDGE_GLOB=answers-$NAME-q4-$tag.jsonl,JUDGE_ARGS=--rubric v3 --graded-dir $graded" \
        "$L/sbatch_judge.slurm"
}
j1=$(exam "$E/norm-exam-devbig.jsonl" "$E/devbig-answers" "$E/devbig-graded-v3" 2 hyb) || exit 1
echo "[round] statute exam, graded by job $j1 into $E/devbig-graded-v3"
# the same exam without the note on a missing article, as RAG Enterprise asks
# it: the note had become the signal (2026-10-16, see legal_eval.py --no-note)
j3=$(exam "$E/norm-exam-devbig.jsonl" "$E/devbig-answers" "$E/devbig-graded-v3" 2 hybnn \
     --no-note) || exit 1
echo "[round] statute exam without the note, graded by job $j3 (label $NAME-q4-hybnn)"
if [ -s "$E/practical-dev.jsonl" ] && [ "${NO_PRACTICAL:-0}" != 1 ]; then
    j2=$(exam "$E/practical-dev.jsonl" "$E/practical-answers" "$E/practical-answers" 1 hyb) || exit 1
    echo "[round] practical exam, graded by job $j2 into $E/practical-answers"
fi
# The traps of an outside test (section 7 of docs/legal-it-experiments.md),
# without the note, as the product asks: the set that decides a promotion,
# so every round measures it. Private: the file stays on $WORK, and the log
# holds totals only.
if [ -s "$E/lexia/lexia-trappole.jsonl" ] && [ "${NO_LEXIA:-0}" != 1 ]; then
    j4=$(exam "$E/lexia/lexia-trappole.jsonl" "$E/lexia/answers" "$E/lexia/graded" 1 hybnn \
         --no-note) || exit 1
    echo "[round] trap questions without the note, graded by job $j4 into $E/lexia/graded"
fi
echo "[round] $NAME: merged model will be $OUT/merged (job $pkg)"
