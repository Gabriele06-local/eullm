#!/bin/bash
# GRPO with verifiable rewards on a merged open-book model — three GPUs.
#
# Three GPUs, one process each (data parallel): every step each GPU writes
# --per-device-batch answers to the same prompts its group shares, they are
# scored (eullm_forge.rl.rewards) and the LoRA moves towards the better ones.
# Generation is most of the time, which is why this is the job that uses
# whole nodes: SFT fits one GPU, sampling eight answers per question does not
# finish on one.
#
#   GRPO_MODEL=$WORK/eullm_runs/stage3/sft-v03-instruct-ob/merged \
#   GRPO_PROMPTS=$WORK/eullm_runs/grpo/prompts.jsonl GRPO_OUT=$WORK/eullm_runs/grpo/v03 \
#       sbatch forge/scripts/leonardo/sbatch_grpo.slurm
#
# Needs TRL, which is NOT installed in the shared venv: it lives in
# $GRPO_LIBS (setup_grpo_libs.sh puts it there), on PYTHONPATH for this job
# only. The shared venv is what every other job imports; on 2026-09-30 one
# upgrade inside it stopped them all.
#
# Then package it like a stage-3 run, with the model it started from:
#
#   S3_BASE=$GRPO_MODEL S3_OUT=$GRPO_OUT S3_NAME=... sbatch sbatch_stage3_package.slurm
#
# Settable:
#   GRPO_MODEL    merged model to start from (required)
#   GRPO_PROMPTS  make_grpo_prompts.py output (required)
#   GRPO_OUT      checkpoints + adapter (required)
#   GRPO_ARGS     extra grpo_train.py options, e.g. "--max-steps 600 --lr 3e-6"
#   GRPO_LIBS     where TRL is (default $WORK/pylibs/grpo)
#   GRPO_REPO     checkout to run (default: the one this file was submitted from)
#   GRPO_JUDGE_GGUF  a judge model as GGUF: started with llama-server on the
#                 LAST GPU of the job, the policy trains on the others, and
#                 grpo_train.py gets --judge-url (prompts made with
#                 make_grpo_prompts.py --judged). Still three GPUs, so still
#                 placed like every other link; needs the CUDA llama-server
#                 (sbatch_build_llama_server.slurm)
#
# Two-hour links, chained, never one long job: on Leonardo a 2 h request is
# backfilled into gaps within the hour, a 24 h one waits for a whole node to
# stay free for a day (2026-10-01: five 24 h GRPO jobs sat in Priority all
# afternoon). Each link resumes from the last checkpoint (every --save-steps,
# 25 by default) and one that finds the adapter already saved exits at once:
#
#   GRPO_MODEL=... GRPO_PROMPTS=... GRPO_OUT=... \
#       bash forge/scripts/leonardo/submit_chain.sh sbatch_grpo.slurm 4 -J eullm-grpo-<name>
#
# Four links cover 250 steps of a 4 B model; give an 8 B five.
#
# THREE GPUs, 24 cores, 340 GB — the geometry sbatch_phase2_8b.slurm and
# sbatch_gen_instruct.slurm measured to be placed quickly. Four GPUs is a
# whole node, which has to be entirely idle before it is given away; three
# make every node with one card in use by someone else a candidate. On
# 2026-10-01 boost_usr_prod had 3253 of 3264 nodes allocated and the
# whole-node GRPO chains sat in the queue all evening. The script counts the
# GPUs it gets: three processes x --per-device-batch 8 = 24 answers a step,
# three prompts of eight.
#
#SBATCH --job-name=eullm-grpo
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
if [ -n "${GRPO_REPO:-}" ]; then
    REPO="$GRPO_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    REPO="${EULLM_REPO:-$WORK/eullm}"
fi
# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/env.sh"

LIBS="${GRPO_LIBS:-$WORK/pylibs/grpo}"
GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "[grpo] repo    $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "[grpo] model   ${GRPO_MODEL:-}"
echo "[grpo] prompts ${GRPO_PROMPTS:-} ($(wc -l < "${GRPO_PROMPTS:-/dev/null}" 2>/dev/null || echo 0) lines)"
echo "[grpo] out     ${GRPO_OUT:-}"
echo "[grpo] args    ${GRPO_ARGS:-}"
echo "[grpo] libs    $LIBS, $GPUS GPU(s)"

fail=0
[ -f "${GRPO_MODEL:-}/config.json" ] || { echo "[grpo] no merged model at '${GRPO_MODEL:-}'" >&2; fail=1; }
[ -s "${GRPO_PROMPTS:-}" ] || { echo "[grpo] no prompts at '${GRPO_PROMPTS:-}'" >&2; fail=1; }
[ -n "${GRPO_OUT:-}" ] || { echo "[grpo] GRPO_OUT is required" >&2; fail=1; }
PYTHONPATH="$LIBS:$REPO/forge:${PYTHONPATH:-}" python -c "import trl" 2>/dev/null ||
    { echo "[grpo] TRL does not import from $LIBS — run setup_grpo_libs.sh" >&2; fail=1; }
[ "$GPUS" -ge 1 ] || { echo "[grpo] no GPU visible" >&2; fail=1; }
SERVER="${LCPP_SERVER:-$LCPP_DIR/build-cuda/bin/llama-server}"
if [ -n "${GRPO_JUDGE_GGUF:-}" ]; then
    [ -s "$GRPO_JUDGE_GGUF" ] || { echo "[grpo] no judge GGUF at $GRPO_JUDGE_GGUF" >&2; fail=1; }
    [ -x "$SERVER" ] || { echo "[grpo] no llama-server at $SERVER" >&2; fail=1; }
    [ "$GPUS" -ge 2 ] || { echo "[grpo] a judge needs a GPU of its own: $GPUS visible" >&2; fail=1; }
fi
[ "$fail" -eq 0 ] || exit 1

# shellcheck disable=SC1091
source "$REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

export PYTHONPATH="$LIBS:$REPO/forge:${PYTHONPATH:-}"
JUDGE=()
PROCS="$GPUS"
if [ -n "${GRPO_JUDGE_GGUF:-}" ]; then
    # The GPUs as Slurm numbered them; the last one judges.
    IFS=, read -r -a DEVS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
    JUDGE_DEV="${DEVS[-1]}"
    POLICY_DEVS="$(IFS=,; echo "${DEVS[*]:0:${#DEVS[@]}-1}")"
    PROCS=$((${#DEVS[@]} - 1))
    if [ -s "$(dirname "$SERVER")/../modules.txt" ] && command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC2046
        module load $(cat "$(dirname "$SERVER")/../modules.txt") 2>/dev/null || true
    fi
    PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
    mkdir -p "$GRPO_OUT"
    SLOTS="${GRPO_JUDGE_SLOTS:-16}"
    CUDA_VISIBLE_DEVICES="$JUDGE_DEV" "$SERVER" -m "$GRPO_JUDGE_GGUF" --host 127.0.0.1 \
        --port "$PORT" -ngl 999 -np "$SLOTS" -c $((SLOTS * 8192)) --no-webui \
        > "$GRPO_OUT/judge-server-${SLURM_JOB_ID:-local}.log" 2>&1 &
    JUDGE_PID=$!
    # Added to the heartbeat's trap, not in place of it: replacing it left
    # the heartbeat loop running after the job, holding its output open.
    trap 'kill "$JUDGE_PID" 2>/dev/null; _heartbeat_end' EXIT INT TERM
    echo "[grpo] judge   $GRPO_JUDGE_GGUF on GPU $JUDGE_DEV, port $PORT; policy on $POLICY_DEVS"
    ready=0
    for _ in $(seq 1 "${GRPO_JUDGE_WAIT:-900}"); do
        kill -0 "$JUDGE_PID" 2>/dev/null || break
        if python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5).status != 200)" 2>/dev/null; then
            ready=1; break
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "[grpo] the judge did not come up:" >&2
        tail -15 "$GRPO_OUT/judge-server-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$POLICY_DEVS"
    JUDGE=(--judge-url "http://127.0.0.1:$PORT")
fi
# shellcheck disable=SC2086
accelerate launch --num_processes "$PROCS" --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no "$REPO/forge/scripts/grpo_train.py" \
    --model "$GRPO_MODEL" --prompts "$GRPO_PROMPTS" --out "$GRPO_OUT" "${JUDGE[@]}" ${GRPO_ARGS:-}
