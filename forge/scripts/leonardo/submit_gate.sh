#!/usr/bin/env bash
# The context gate start to finish, in one submission: off-topic questions,
# labelled examples, training, then a released round examined through it.
#
#   bash forge/scripts/leonardo/submit_gate.sh r2h-steps-8b-mixc
#
#   ROUND   a GRPO round in $WORK/eullm_runs/grpo whose GGUF answers the exams
#
# What it queues, each step waiting on the one it needs (sbatch_gate.slurm):
#
#   1. offtopic -> 2. pairs -> 3. train, one GPU each;
#   4. after the training, ROUND's GGUF asked through the gate, graded like
#      every other model: the LexIA traps without the note, gated questions
#      answered by the fixed reply (tag hybnn-gate) and, apart, with a note
#      to the model instead (hybnn-gatenote); devbig without the note and the
#      practical exam through the gate (hybnn-gate, hyb-gate), the cost side:
#      a question gated wrongly loses its answer there.
#
# Idempotent: nothing is queued while an eullm-gate job is in the queue, and
# each step keeps what is already written.

set -uo pipefail

[ $# -eq 1 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
ROUND=$1
L="$(cd "$(dirname "$0")" && pwd)"
G="$WORK/eullm_runs/grpo"
GD="${GATE_DIR:-$WORK/eullm_runs/gate}"
E="$WORK/eval"
OUT="$G/$ROUND"

fail=0
check() { [ -e "$1" ] || { echo "[gate] missing $2: $1" >&2; fail=1; }; }
check "$OUT/merged/config.json" "the round's merged model (its tokenizer)"
check "$OUT/gguf/$ROUND-q4_k_m.gguf" "the round's GGUF"
OB="${GATE_OPENBOOK:-}"
if [ -z "$OB" ]; then
    for v in v06 v05 v04; do
        [ -s "$WORK/eullm_runs/stage3/openbook-$v.jsonl" ] && { OB="$WORK/eullm_runs/stage3/openbook-$v.jsonl"; break; }
    done
fi
check "${OB:-/no-openbook-pairs}" "open-book SFT pairs"
check "${GATE_PREMISE:-$WORK/eullm_runs/premise5/grpo.jsonl}" "false-premise GRPO prompts"
if squeue --me -h -o "%j" 2>/dev/null | grep -q "^eullm-gate"; then
    echo "[gate] a gate job is already queued, nothing submitted"; exit 0
fi
[ "$fail" -eq 0 ] || exit 1

sub() {  # sbatch that prints the job id; fails (and says so) when sbatch does
    local id
    id=$(sbatch --parsable "$@") && [ -n "$id" ] ||
        { echo "[gate] sbatch failed: $*" >&2; return 1; }
    echo "${id%%;*}"
}
mkdir -p "$GD"
a=$(GATE_STEP=offtopic sub -J eullm-gate-offtopic "$L/sbatch_gate.slurm") || exit 1
b=$(GATE_STEP=pairs GATE_OPENBOOK="$OB" sub -J eullm-gate-pairs --dependency="afterok:$a" \
    "$L/sbatch_gate.slurm") || exit 1
c=$(GATE_STEP=train sub -J eullm-gate-train --dependency="afterok:$b" "$L/sbatch_gate.slurm") || exit 1
echo "[gate] off-topic questions $a, examples $b, training $c -> $GD/model"

NORMS="$WORK/norms/legislazione_*.chunks.jsonl"
exam() {  # items out-dir graded-dir links tag legal_eval-options -> id of the judge job
    local items=$1 out=$2 graded=$3 links=$4 tag=$5 extra=$6 d="afterok:$c" x i
    mkdir -p "$out"
    for i in $(seq 1 "$links"); do
        x=$(EXAM_MODELS="$ROUND-q4=$OUT/merged" EXAM_GGUF="$ROUND-q4=$OUT/gguf/$ROUND-q4_k_m.gguf" \
            EXAM_ITEMS="$items" EXAM_OUT="$out" EXAM_NORMS="$NORMS" EXAM_OPEN_ONLY=1 \
            EXAM_OPEN_TAG="$tag" EXAM_OPEN_ARGS="$extra" EXAM_EMBEDDER=Qwen/Qwen3-Embedding-0.6B \
            EXAM_RERANKER=Qwen/Qwen3-Reranker-0.6B \
            sub -J "eullm-exam-$ROUND-gate" --dependency="$d" "$L/sbatch_norm_exam.slurm") || return 1
        d="afterany:$x"
    done
    sub --dependency="$d" --export="ALL,JUDGE_DIR=$out,JUDGE_QUIET=1,JUDGE_GLOB=answers-$ROUND-q4-$tag.jsonl,JUDGE_ARGS=--rubric v3 --graded-dir $graded" \
        "$L/sbatch_judge.slurm"
}
GATE="--gate $GD/model"
if [ -s "$E/lexia/lexia-trappole.jsonl" ]; then
    j=$(exam "$E/lexia/lexia-trappole.jsonl" "$E/lexia/answers" "$E/lexia/graded" 1 hybnn-gate \
        "--no-note $GATE --gate-mode answer") || exit 1
    echo "[gate] traps through the gate, fixed reply: judge $j"
    j=$(exam "$E/lexia/lexia-trappole.jsonl" "$E/lexia/answers" "$E/lexia/graded" 1 hybnn-gatenote \
        "--no-note $GATE --gate-mode note") || exit 1
    echo "[gate] traps through the gate, note to the model: judge $j"
fi
j=$(exam "$E/norm-exam-devbig.jsonl" "$E/devbig-answers" "$E/devbig-graded-v3" 2 hybnn-gate \
    "--no-note $GATE --gate-mode answer") || exit 1
echo "[gate] devbig without the note through the gate: judge $j"
if [ -s "$E/practical-dev.jsonl" ]; then
    j=$(exam "$E/practical-dev.jsonl" "$E/practical-answers" "$E/practical-answers" 1 hyb-gate \
        "$GATE --gate-mode answer") || exit 1
    echo "[gate] practical exam through the gate: judge $j"
fi
