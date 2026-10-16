#!/bin/bash
# The practical-questions exam (make_practical_exam.py, roadmap 1.1): one
# teacher llama-server per GPU, three GPUs, one two-hour job.
#
#   PE_TEACHER=$WORK/gguf/<teacher>.gguf sbatch forge/scripts/leonardo/sbatch_practical_exam.slurm
#
# Writes $PE_DIR/practical-dev.jsonl and practical-heldout.jsonl once: a job
# that finds both exits at once (the held-out exam is drawn once). Quiet:
# counts only.
#
# The teacher should not be the judge: a judge grading answers to questions
# its own family wrote would grade its own reading of the law. The default is
# the judge's model only because it is the GGUF known to be on $WORK.
#
# Settable:
#   PE_TEACHER   teacher GGUF (default Qwen3-30B-A3B-Instruct-2507 Q8_0)
#   PE_DIR       output directory (default $WORK/eval)
#   PE_NORMS     legislation files (default $WORK/norms/legislazione_*.chunks.jsonl)
#   PE_EXCLUDE   training files whose articles are never asked (default every
#                GRPO prompts file, stage-3 open-book pairs and OPD prompts file)
#   PE_ARGS      more make_practical_exam.py options, e.g. "--per-code 80"
#   PE_REPO      checkout to run (default: the one this file was submitted from)
#
#SBATCH --job-name=eullm-practical-exam
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=340G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${PE_REPO:-}" ]; then
    REPO="$PE_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

TEACHER="${PE_TEACHER:-$WORK/gguf/qwen3-30b-a3b-instruct-2507/qwen3-30b-a3b-instruct-2507-q8_0.gguf}"
DIR="${PE_DIR:-$WORK/eval}"
# shellcheck disable=SC2206
NORMS=(${PE_NORMS:-$WORK/norms/legislazione_*.chunks.jsonl})
# shellcheck disable=SC2206
EXCL=(${PE_EXCLUDE:-$WORK/eullm_runs/grpo/prompts*.jsonl $WORK/eullm_runs/stage3/openbook*.jsonl $WORK/eullm_runs/opd/prompts*.jsonl})
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"

if [ -s "$DIR/practical-dev.jsonl" ] && [ -s "$DIR/practical-heldout.jsonl" ]; then
    echo "[pe] the exam is already drawn in $DIR: nothing left to do"
    exit 0
fi
fail=0
[ -s "$TEACHER" ] || { echo "[pe] no teacher GGUF at $TEACHER" >&2; fail=1; }
[ -x "$SERVER" ] || { echo "[pe] no llama-server at $SERVER" >&2; fail=1; }
for f in "${NORMS[@]}"; do [ -s "$f" ] || { echo "[pe] missing $f" >&2; fail=1; }; done
[ "$GPUS" -ge 1 ] || { echo "[pe] no GPU visible" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1
EX=()
for f in "${EXCL[@]}"; do [ -s "$f" ] && EX+=("$f"); done
echo "[pe] ${#EX[@]} training file(s) whose articles are excluded"

if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
fi
DEVS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
echo "[pe] repo $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')), GPUs $DEVS"

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

# shellcheck disable=SC2086
python "$REPO/forge/scripts/make_practical_exam.py" "${NORMS[@]}" --exclude-pairs "${EX[@]}" \
    --serve "$TEACHER" --gpus "$DEVS" --llama-server "$SERVER" \
    --out-dev "$DIR/practical-dev.jsonl" --out-heldout "$DIR/practical-heldout.jsonl" \
    ${PE_ARGS:-}
