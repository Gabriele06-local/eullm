#!/bin/bash
# Instruction tuning on JUPITER Booster (deliverable 4, against Leonardo's
# 46 minutes for a 4B on 3,400 pairs, 2 epochs, one A100), then the
# held-out evaluation of what it made (deliverable 6). Four runs side by
# side on one node, one GPU each (GPU i with cores 72i-72i+71):
#
#   d4b  the 4B distilled on legislation   (norms-4b, merged)
#   d8b  the 8B distilled on legislation   (norms-8b, merged)
#   b4b  Qwen3-4B-Base, not distilled      what distillation adds,
#   b8b  Qwen3-8B-Base, not distilled      on the same pairs and recipe
#
# FORGE_SFT_ARMS names others as "name=model name=model" (a merged
# directory, a hub id in the offline cache, or "base+adapter" for a
# distillation checkpoint, merged first), four at most. Each run is
# stage3_sft.py (Leonardo's stage 3: LoRA r32, 2 epochs, the identity
# examples mixed in) on FORGE_SFT_PAIRS, then the adapter merged into the
# weights, chat_smoke.py (does it end its turn) and legal_eval.py (the
# legal-it held-out questions, keyword score). The reference,
# Qwen3-4B-Instruct-2507, is evaluated the same way on GPU 0 once its run is
# done, if it is in the offline cache. Everything lands in
# $FORGE_RUN_DIR/sft/: one directory a run, chat-smoke.csv, legal-eval.csv
# and times.csv (seconds of training per run: the deliverable 4 figure).
#
#     eullm && sbatch --account e-aif-2026pg01-1434 \
#         --output="$PROJECT/$USER/forge/runs/logs/sft-%j.out" \
#         forge/scripts/jupiter/sbatch_sft.slurm
#
# FORGE_SFT_TAG puts the runs in $FORGE_RUN_DIR/sft-<tag> (another set of
# pairs, other arms). With a held-out exam at FORGE_SFT_EXAM (default
# $FORGE_DATA_DIR/eval/norm-exam.jsonl, make_norm_exam.py) every model also
# sits it, into norm-exam.csv; with FORGE_SFT_NORMS (legislation chunk files)
# it sits it open book too, as "<label>-open".
#
# Resumable: a rerun continues a run from its checkpoint, skips a finished
# adapter, a merged model and an evaluation already done, so rerunning it on
# finished runs only adds what is missing (the exam, say). Whole node: 4
# GPU-hours an hour.

#SBATCH --job-name=forge-sft
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
RUNS="$FORGE_RUN_DIR"
ARMS="${FORGE_SFT_ARMS:-d4b=$RUNS/norms-4b/checkpoints/qwen3_4b_norms_distilled/merged d8b=$RUNS/norms-8b/checkpoints/qwen3_8b_norms_distilled/merged b4b=Qwen/Qwen3-4B-Base b8b=Qwen/Qwen3-8B-Base}"
PAIRS="${FORGE_SFT_PAIRS:-$FORGE_DATA_DIR/norms-instruct/pairs.jsonl}"
REF="${FORGE_SFT_REFERENCE:-Qwen/Qwen3-4B-Instruct-2507}"
TRAIN_S="${FORGE_SFT_TRAIN_S:-5700}"
SFT="$RUNS/sft${FORGE_SFT_TAG:+-$FORGE_SFT_TAG}"
EXAM="${FORGE_SFT_EXAM:-$FORGE_DATA_DIR/eval/norm-exam.jsonl}"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$SFT" "$RUNS/power"

echo "=== Forge instruction tuning: job $JOB on $(hostname), $(date -Is) ==="
echo "  pairs $PAIRS ($(wc -l <"$PAIRS" 2>/dev/null || echo 0) lines), training at most ${TRAIN_S}s a run, into $SFT"
if [ -s "$EXAM" ]; then echo "  held-out exam $EXAM ($(wc -l <"$EXAM") questions; totals only in the logs)"
else echo "  no held-out exam at $EXAM: make_norm_exam.py on a login node; the ten seed questions only"; fi
[ -s "$PAIRS" ] || { echo "[!!] no pairs at $PAIRS: sbatch_gen_instruct.slurm first"; exit 1; }
python -c "import transformers, peft; print('  imports: transformers', transformers.__version__, 'peft', peft.__version__)" ||
    { echo "[!!] the venv does not import on this node"; exit 1; }

# A hub id counts when the offline cache has it, a directory when it holds a
# model; "base+adapter" when the base counts and the adapter is there.
have_model() {
    case "$1" in *+*) have_model "${1%%+*}" && [ -f "${1#*+}/adapter_config.json" ]; return ;; esac
    if [ -d "$1" ]; then [ -f "$1/config.json" ]; return; fi
    python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$1" >/dev/null 2>&1
}

in_csv() { awk -F, -v l="$1" '$2 == l { found = 1 } END { exit !found }' "$2" 2>/dev/null; }

evaluate() {   # evaluate <label> <merged dir or hub id> <dir for the answers>
    if in_csv "$1" "$SFT/legal-eval.csv"; then
        echo "[eval] $1 already in legal-eval.csv"
    else
        python "$EULLM_REPO/forge/scripts/chat_smoke.py" "$2" --label "$1" --csv "$SFT/chat-smoke.csv"
        python "$EULLM_REPO/forge/scripts/legal_eval.py" "$2" --label "$1" \
            --csv "$SFT/legal-eval.csv" --answers "$3/legal-eval-answers.jsonl" --quiet
    fi
    # The held-out exam drawn from the law (make_norm_exam.py), quiet: the
    # logs hold totals, never a question.
    if [ -s "$EXAM" ] && ! in_csv "$1" "$SFT/norm-exam.csv"; then
        python "$EULLM_REPO/forge/scripts/legal_eval.py" "$2" --label "$1" --items "$EXAM" \
            --csv "$SFT/norm-exam.csv" --answers "$3/norm-exam-answers.jsonl" --quiet
    fi
    # The same exam open book (FORGE_SFT_NORMS: legislation chunk files): each
    # question with the articles BM25 finds for it in front, as "<label>-open".
    # Closed book, no model knew the articles (16-10-2026); this asks whether
    # it can read them.
    if [ -s "$EXAM" ] && [ -n "${FORGE_SFT_NORMS:-}" ] && ! in_csv "$1-open" "$SFT/norm-exam.csv"; then
        # shellcheck disable=SC2086
        python "$EULLM_REPO/forge/scripts/legal_eval.py" "$2" --label "$1-open" --items "$EXAM" \
            --norms $FORGE_SFT_NORMS --csv "$SFT/norm-exam.csv" \
            --answers "$3/norm-exam-open-answers.jsonl" --quiet
    fi
}

run_arm() {    # run_arm <name> <model>; runs on the GPU it is given
    local name="$1" model="$2" out="$SFT/$1" t0 rc
    mkdir -p "$out"
    # "base+adapter": a distillation checkpoint, merged into its base first.
    if [[ "$model" == *+* ]]; then
        if [ ! -f "$out/start/config.json" ]; then
            python - "${model%%+*}" "${model#*+}" "$out/start" <<'PY' || { echo "[!!] $name: merging the checkpoint failed"; return 1; }
import sys
from eullm_forge.identity import merge_identity_adapter
merge_identity_adapter(sys.argv[1], sys.argv[2], sys.argv[3])
PY
        fi
        model="$out/start"
    fi
    if [ ! -f "$out/adapter/adapter_config.json" ]; then
        t0=$(date +%s)
        timeout --signal=TERM "$TRAIN_S" python "$EULLM_REPO/forge/scripts/stage3_sft.py" \
            --model "$model" --pairs "$PAIRS" --out "$out"
        rc=$?
        echo "$name,$model,$(wc -l <"$PAIRS"),$(( $(date +%s) - t0 )),$rc,$JOB" >>"$SFT/times.csv"
        [ -f "$out/adapter/adapter_config.json" ] || { echo "[!!] $name: no adapter (exit $rc)"; return 1; }
    fi
    if [ ! -f "$out/merged/config.json" ]; then
        python - "$model" "$out/adapter" "$out/merged" <<'PY' || { echo "[!!] $name: merge failed"; return 1; }
import sys
from eullm_forge.identity import merge_identity_adapter
merge_identity_adapter(sys.argv[1], sys.argv[2], sys.argv[3])
PY
    fi
    evaluate "$name" "$out/merged" "$out"
}
# The merge imports eullm_forge from the checkout.
export PYTHONPATH="$EULLM_REPO/forge${PYTHONPATH:+:$PYTHONPATH}"

# A subshell pinned to a GPU and its Grace cores; what it starts inherits both.
on_gpu() {     # on_gpu <gpu> <command...>
    local g="$1"; shift
    taskset -cp "$((72 * g))-$((72 * g + 71))" "$BASHPID" >/dev/null
    CUDA_VISIBLE_DEVICES=$g "$@"
}

POWER="$RUNS/power/sft-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!

PIDS=() NAMES=() LOGS=()
g=0
for arm in $ARMS; do
    name="${arm%%=*}" model="${arm#*=}"
    [ "$g" -lt 4 ] || { echo "[!!] $name: four runs a node, the rest wait"; continue; }
    if ! have_model "$model"; then
        echo "[!!] $name: no model at $model, skipped"; g=$((g + 1)); continue
    fi
    log="$SFT/$name/sft-$JOB.log"
    mkdir -p "$SFT/$name"
    echo "--- $name: GPU $g, cores $((72 * g))-$((72 * g + 71)), $model -> $SFT/$name"
    ( on_gpu "$g" run_arm "$name" "$model" ) >"$log" 2>&1 &
    PIDS+=($!) NAMES+=("$name") LOGS+=("$log")
    g=$((g + 1))
done

for k in "${!PIDS[@]}"; do
    wait "${PIDS[$k]}"; rc=$?
    echo
    echo "=== ${NAMES[$k]}: $([ "$rc" -eq 0 ] && echo done || echo "FAILED (exit $rc)") ==="
    grep -E "^(\[stage3\]|\[eval\]|\[!!\])|'loss'|train_runtime" "${LOGS[$k]}" | tail -6 | sed 's/^/  /'
    [ "$rc" -ne 0 ] && tail -15 "${LOGS[$k]}" | sed 's/^/    /'
    # The reference waits for the GPU the first run leaves free.
    if [ "$k" -eq 0 ]; then
        if have_model "$REF"; then
            mkdir -p "$SFT/reference"
            ( on_gpu 0 evaluate reference "$REF" "$SFT/reference" ) >"$SFT/reference/eval-$JOB.log" 2>&1
            echo "=== reference $REF: evaluated (exit $?) ==="
        else
            echo "=== reference $REF: not in the offline cache, not evaluated ==="
        fi
    fi
done

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null
echo
echo "times.csv: name, model, pairs, seconds of training, exit, job"
for f in times legal-eval norm-exam chat-smoke; do
    [ -f "$SFT/$f.csv" ] || continue
    echo "--- $f.csv"
    if command -v column >/dev/null; then column -s, -t <"$SFT/$f.csv"; else cat "$SFT/$f.csv"; fi | sed 's/^/  /'
done
echo
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) ==="
