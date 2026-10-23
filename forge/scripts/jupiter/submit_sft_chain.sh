#!/usr/bin/env bash
# Instruction tuning as a chain that finishes by itself: LINKS links of
# sbatch_sft.slurm, each starting when the one before ends (afterany; a link
# resumes the runs, skips what is done and only adds what is missing), then
# the judge (sbatch_judge.slurm), which grades every exam answered so far.
# From a login node, with the FORGE_SFT_* settings of sbatch_sft.slurm:
#
#     eullm && FORGE_SFT_TAG=ob1 FORGE_SFT_PAIRS=... FORGE_SFT_ARMS="..." \
#         bash forge/scripts/jupiter/submit_sft_chain.sh [LINKS]
#
# Checked first, and nothing submitted when one fails: the pairs, the exam,
# each arm's model in the offline cache (compute nodes have no network), the
# system prompt, and no chain of the same tag already queued.
# FORGE_LINK_H sets a link's hours (default 2): training stops an hour
# before a longer link's end, for the exams.
# FORGE_CHAIN_REPLACE=1 cancels the links not started and their judge first.
# FORGE_CHAIN_MORE=1 adds LINKS links behind the last one queued and moves
# the judge that waits for the chain behind them: runs slower than their
# links (ob1, 21-10-2026: the 4B at 754 of 900 steps after the first).
# FORGE_CHAIN_AFTER=<job> starts the chain when that job ends (the last link
# of a distillation chain): an arm's directory not there yet is accepted, as
# the job will make it; a link that finds it still missing skips that arm.

set -uo pipefail
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# A partition down for maintenance or a silent controller is waited out.
# shellcheck source=sbatch_retry.sh
source "$(dirname "${BASH_SOURCE[0]}")/sbatch_retry.sh"
LINKS="${1:-2}"
TAG="${FORGE_SFT_TAG:?FORGE_SFT_TAG: the name of this set of runs}"
PAIRS="${FORGE_SFT_PAIRS:?FORGE_SFT_PAIRS: the pairs to train on}"
ARMS="${FORGE_SFT_ARMS:?FORGE_SFT_ARMS: name=model ...}"
EXAM="${FORGE_SFT_EXAM:-$FORGE_DATA_DIR/eval/norm-exam.jsonl}"
ACCOUNT="${SBATCH_ACCOUNT:-e-aif-2026pg01-1434}"
AFTER="${FORGE_CHAIN_AFTER:-}"
LOGS="$FORGE_RUN_DIR/logs"
NAME="forge-sft-$TAG"
bad=0

[ -s "$PAIRS" ] && echo "[ok] pairs $PAIRS ($(wc -l <"$PAIRS"))" || { echo "[!!] no pairs at $PAIRS"; bad=1; }
[ -s "$EXAM" ] && echo "[ok] exam $EXAM ($(wc -l <"$EXAM") questions)" || { echo "[!!] no exam at $EXAM"; bad=1; }
[ -z "${FORGE_SFT_SYSTEM_FILE:-}" ] || [ -s "$FORGE_SFT_SYSTEM_FILE" ] ||
    { echo "[!!] no system prompt at $FORGE_SFT_SYSTEM_FILE"; bad=1; }
[ -n "${FORGE_SFT_NORMS:-}" ] && echo "[ok] open book over $(wc -w <<<"$FORGE_SFT_NORMS") norm files" ||
    echo "[..] no FORGE_SFT_NORMS: closed book only"
in_cache() { python -c "import sys; from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)" "$1" >/dev/null 2>&1; }
for arm in $ARMS; do
    model="${arm#*=}"
    # "base+adapter": a distillation checkpoint, merged into its base by the job
    # (sbatch_sft.slurm); the base in the cache or a directory, the adapter there.
    if [[ "$model" == *+* ]]; then
        base="${model%%+*}" adapter="${model#*+}"
        { [ -f "$base/config.json" ] || in_cache "$base"; } && [ -f "$adapter/adapter_config.json" ] &&
            echo "[ok] ${arm%%=*}: $base + $adapter" ||
            { echo "[!!] ${arm%%=*}: no base $base or no adapter_config.json in $adapter"; bad=1; }
    elif [ -d "$model" ]; then
        [ -f "$model/config.json" ] && echo "[ok] ${arm%%=*}: $model" || { echo "[!!] ${arm%%=*}: no model in $model"; bad=1; }
    elif [ -n "$AFTER" ] && [[ "$model" == /* ]]; then
        echo "[..] ${arm%%=*}: $model not there yet, made by the time job $AFTER ends"
    elif in_cache "$model"; then
        echo "[ok] ${arm%%=*}: $model in the offline cache"
    else
        echo "[!!] ${arm%%=*}: $model is not in the offline cache (download it on this login node)"; bad=1
    fi
done
wait_for_slurm || exit 1
[ -z "$AFTER" ] || [ -n "$(squeue -h -j "$AFTER" -o %i 2>/dev/null)" ] ||
    { echo "[!!] job $AFTER is not in the queue (FORGE_CHAIN_AFTER)"; bad=1; }
# FORGE_CHAIN_REPLACE=1: the links of this chain not yet started, and the
# judge waiting for them, are cancelled and the chain submitted again (other
# link hours, say); a running link stays, and the new links follow it.
if [ "${FORGE_CHAIN_REPLACE:-0}" = 1 ] && [ "$bad" = 0 ]; then
    for l in $(squeue -h -u "$USER" -n "$NAME" -t PENDING -o %i); do
        for j in $(squeue -h -u "$USER" -n forge-judge -t PENDING -o %i); do
            scontrol show job "$j" 2>/dev/null | grep -q "Dependency=.*:${l}\b" &&
                scancel "$j" && echo "[ok] judge $j, waiting for $l, cancelled"
        done
        scancel "$l" && echo "[ok] link $l, not started, cancelled"
    done
    sleep 5
    FORGE_CHAIN_MORE=1
fi
QUEUED=$(squeue -h -u "$USER" -n "$NAME" -o %i 2>/dev/null | sort -n | tail -1)
if [ -n "$QUEUED" ] && [ "${FORGE_CHAIN_MORE:-0}" != 1 ]; then
    echo "[!!] a $NAME chain is already queued: squeue -u $USER -n $NAME (FORGE_CHAIN_MORE=1 adds links)"; bad=1
fi
[ "$bad" = 0 ] || { echo "[!!] nothing submitted"; exit 1; }

mkdir -p "$LOGS"
prev="${QUEUED:-$AFTER}" jobs=()
for i in $(seq 1 "$LINKS"); do
    j=$(cd "$REPO" && sbatch_retry --parsable --account "$ACCOUNT" --job-name "$NAME" \
        --time="${FORGE_LINK_H:-2}:00:00" \
        ${prev:+--dependency=afterany:$prev} --output="$LOGS/sft-%j.out" \
        forge/scripts/jupiter/sbatch_sft.slurm) || { echo "[!!] sbatch failed at link $i"; exit 1; }
    prev="${j%%;*}" jobs+=("$prev")
done
# A judge already waiting for this chain is moved behind the new last link,
# not submitted twice.
judge=""
if [ -n "${QUEUED:-}" ]; then
    for j in $(squeue -h -u "$USER" -n forge-judge -t PENDING -o %i 2>/dev/null); do
        scontrol show job "$j" 2>/dev/null | grep -q "Dependency=.*:${QUEUED}\b" || continue
        scontrol update JobId="$j" Dependency="afterany:$prev" && judge="$j" &&
            echo "[ok] judge $j now waits for $prev"
    done
fi
[ -n "$judge" ] || judge=$(cd "$REPO" && sbatch_retry --parsable --account "$ACCOUNT" --dependency="afterany:$prev" \
    --output="$LOGS/judge-%j.out" forge/scripts/jupiter/sbatch_judge.slurm) ||
    { echo "[!!] sbatch of the judge failed"; exit 1; }
echo "[ok] $NAME: links ${jobs[*]}, then the judge ${judge%%;*}"
echo "     results: $FORGE_RUN_DIR/sft-$TAG/norm-exam.csv, $FORGE_RUN_DIR/logs/judge-${judge%%;*}.out"
