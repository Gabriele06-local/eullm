#!/bin/bash
# One complete vertical, timed (deliverables 1 and 8): instruction-tuned and
# merged models (sbatch_sft.slurm) made into what ships and examined as it
# ships. For each arm, on its own pair of GPUs and half of the node's cores:
#
#   1. the merged model converted to GGUF F16 (llama.cpp's
#      convert_hf_to_gguf.py, at the commit EULLM Engine is built on), then
#      quantized to Q4_K_M (llama-quantize, tools/jupiter/install_runtimes.sh);
#   2. the held-out exam, open book, with and without the note on missing
#      articles, answered by the Q4_K_M through EULLM Engine on the first GPU
#      and through llama-server on the second, at once: against the bf16
#      answers of the same model the first says what quantization and the
#      engine cost together, the second splits the two.
#
#     eullm && bash forge/scripts/jupiter/submit_vertical.sh
#
# The answers land in $FORGE_RUN_DIR/sft-vert/<arm>-q4-eullm/ and
# <arm>-q4-llama/, where sbatch_judge.slurm finds them; the GGUFs in
# $FORGE_RUN_DIR/gguf/<arm>/; each stage's seconds in sft-vert/vertical.csv,
# with the arm's training seconds from the SFT run's times.csv, so the
# vertical's time adds up in one place. Resumable: a GGUF or an exam already
# there is not made again. Whole node: 4 GPU-hours an hour.

#SBATCH --job-name=forge-vertical
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

set -uo pipefail
EULLM_REPO="${EULLM_REPO:-$SLURM_SUBMIT_DIR}"
# shellcheck source=env.sh
source "$EULLM_REPO/forge/scripts/jupiter/env.sh"
RUNS="$FORGE_RUN_DIR"
SFT="${VERT_SFT:-$RUNS/sft-ob1}"
read -r -a ARMS <<<"${VERT_ARMS:-p-m8i p-q4b}"
OUT="$RUNS/sft-vert"
EXAM="${FORGE_SFT_EXAM:-$FORGE_DATA_DIR/eval/norm-exam.jsonl}"
NORMS="${FORGE_SFT_NORMS:-$(ls "$FORGE_DATA_DIR"/norms-src/legislazione_*.chunks.jsonl | grep -v eurlex | tr '\n' ' ')}"
SYSTEM_FILE="${FORGE_SFT_SYSTEM_FILE:-$EULLM_REPO/forge/model_cards/legal-it-8b/system-prompt.txt}"
EULLM_BIN="${EULLM_BIN:-$PROJECT/$USER/eullm-target/sites/release/eullm}"
RT="${VERT_RUNTIMES:-$PROJECT/$USER/campaign/runtimes}"
LCPP="$EULLM_REPO/engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" "$RUNS/gguf" "$RUNS/power"
export PYTHONPATH="$EULLM_REPO/forge${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Forge vertical: job $JOB on $(hostname), $(date -Is) ==="
echo "  arms ${ARMS[*]} from $SFT; engine $("$EULLM_BIN" --version 2>&1 | head -1)"
bad=0
[ -x "$RT/llama-quantize" ] || { echo "[!!] no $RT/llama-quantize: tools/jupiter/install_runtimes.sh"; bad=1; }
[ -x "$RT/llama-server" ] || { echo "[!!] no $RT/llama-server: tools/jupiter/install_runtimes.sh"; bad=1; }
[ -x "$EULLM_BIN" ] || { echo "[!!] no engine at $EULLM_BIN"; bad=1; }
[ -s "$EXAM" ] || { echo "[!!] no exam at $EXAM"; bad=1; }
python -c "import sys; sys.path.insert(1, sys.argv[1]); import gguf" "$LCPP/gguf-py" ||
    { echo "[!!] llama.cpp's gguf-py does not import in the venv"; bad=1; }
for arm in "${ARMS[@]}"; do
    [ -f "$SFT/$arm/merged/config.json" ] || { echo "[!!] no merged model at $SFT/$arm/merged"; bad=1; }
done
[ "${#ARMS[@]}" -le 2 ] || { echo "[!!] two arms a node (two GPUs each)"; bad=1; }
[ "$bad" = 0 ] || exit 1

stage() {   # stage <arm> <name> <seconds>: one row of vertical.csv
    echo "$1,$2,$3,$JOB" >>"$OUT/vertical.csv"
}

exam() {    # exam <arm> <runtime> <gpu> <gguf>: open book, with and without the note
    local arm=$1 rt=$2 g=$3 gguf=$4 dir="$OUT/$1-q4-${2%%-*}" ask=() t0 kind extra
    case "$arm" in *m8i*|*mistral*|*ministral*) ask=(--system-file "$SYSTEM_FILE") ;; esac
    mkdir -p "$dir"
    for kind in open open-nonote; do
        [ -s "$dir/norm-exam-$kind-answers.jsonl" ] && continue
        extra=()
        [ "$kind" = open-nonote ] && extra=(--no-note)
        t0=$(date +%s)
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$g python "$EULLM_REPO/forge/scripts/legal_eval.py" \
            "$SFT/$arm/merged" --label "$arm-q4-${rt%%-*}-$kind" --items "$EXAM" \
            --norms $NORMS "${extra[@]}" "${ask[@]}" --gguf "$gguf" --runtime "$rt" \
            --eullm "$EULLM_BIN" --llama-server "$RT/llama-server" \
            --server-parallel 16 --server-ctx 8192 \
            --csv "$OUT/norm-exam.csv" --answers "$dir/norm-exam-$kind-answers.jsonl" --quiet
        stage "$arm" "exam-$kind-$rt" $(( $(date +%s) - t0 ))
    done
}

run_arm() { # run_arm <arm> <first gpu>
    local arm=$1 g=$2 dir="$RUNS/gguf/$1" t0
    mkdir -p "$dir"
    local f16="$dir/$arm-f16.gguf" q4="$dir/$arm-q4_k_m.gguf"
    if [ ! -s "$q4" ]; then
        if [ ! -s "$f16" ]; then
            t0=$(date +%s)
            python "$LCPP/convert_hf_to_gguf.py" "$SFT/$arm/merged" --outtype f16 \
                --outfile "$f16.tmp" && mv "$f16.tmp" "$f16" || { echo "[!!] $arm: conversion failed"; return 1; }
            stage "$arm" convert-f16 $(( $(date +%s) - t0 ))
        fi
        t0=$(date +%s)
        "$RT/llama-quantize" "$f16" "$q4.tmp" Q4_K_M && mv "$q4.tmp" "$q4" ||
            { echo "[!!] $arm: quantization failed"; return 1; }
        stage "$arm" quantize-q4_k_m $(( $(date +%s) - t0 ))
    fi
    echo "[vert] $arm: F16 $(du -h "$f16" | cut -f1), Q4_K_M $(du -h "$q4" | cut -f1)"
    # The training seconds of this arm, from the SFT run that made it.
    awk -F, -v a="$arm" '$1 == a { s += $4 } END { if (s) print a ",train-sft," s ",sft" }' \
        "$SFT/times.csv" >>"$OUT/vertical.csv" 2>/dev/null
    exam "$arm" eullm "$g" "$q4" &
    exam "$arm" llama-server "$((g + 1))" "$q4" &
    wait
}

POWER="$RUNS/power/vertical-$JOB.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -l 5 >"$POWER" 2>/dev/null &
SAMPLER=$!

PIDS=()
for i in "${!ARMS[@]}"; do
    g=$((2 * i))
    ( taskset -cp "$((144 * i))-$((144 * i + 143))" "$BASHPID" >/dev/null
      run_arm "${ARMS[$i]}" "$g" ) >"$OUT/vertical-$JOB-${ARMS[$i]}.log" 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null

for arm in "${ARMS[@]}"; do
    echo
    echo "=== $arm ==="
    grep -hE '^\[(vert|eval)\].*(F16|items|ended|abstained)|\[!!\]|Error' "$OUT/vertical-$JOB-$arm.log" | sed 's/^/  /'
done
echo
echo "--- vertical.csv (arm, stage, seconds, job)"
column -s, -t <"$OUT/vertical.csv" | sed 's/^/  /'
echo
python "$EULLM_REPO/forge/scripts/jupiter/energy.py" "$POWER" --interval 5
echo "=== done $(date -Is) (exit $RC) ==="
exit "$RC"
