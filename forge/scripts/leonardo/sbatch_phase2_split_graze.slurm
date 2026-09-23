#!/bin/bash
# The split arm, in SHORT LINKS that keep themselves going.
#
# Measured on 2026-09-22, and the whole design follows from three numbers:
#
#   * a one-node, 4-GPU, QOS `normal` job asking for 30 minutes or 1 hour is
#     placed in FIFTY SECONDS and actually runs; the same job asking for 4
#     hours was not placed in 28 hours. At 2 hours the split arm's head was
#     given a backfill START_TIME within a minute of the change — and the
#     projection was withdrawn on the next cycle and the job did not start.
#     A START_TIME is a plan, not a reservation, and reading one as a start
#     is how this was briefly called a success when it was not.
#     So the threshold is somewhere between 1 and 2 hours: one hour is
#     PROVEN, two hours is the hypothesis this script is set up to test.
#   * startup costs ~14 minutes per link — 3 of pre-flight, 11 loading the
#     61 GB teacher — and it is paid every time.
#   * throughput is ~379 steps/hour.
#
# So the loss is ~22 minutes per link (startup, plus on average half a
# save interval of unsaved tail) and it is FIXED: it does not grow with the
# link. That is the whole argument for two hours rather than one — 22 minutes
# out of 120 is 82 % useful, out of 60 it is 63 %.
#
# This does not replace sbatch_phase2_split.slurm and does not modify it. The
# 24 h chain stays queued and keeps catching the large windows at 99 %
# efficiency, as it did on 20 and 21 September. This grazes the small ones.
# The two add up; neither is a substitute for the other.
#
# HOW IT KEEPS ITSELF GOING, and why the order matters. A link ends in
# TIMEOUT by design, so anything after the training command never runs — the
# re-submission has to happen BEFORE training, not after. But a successor
# submitted now with no dependency could start while we are still training,
# and two jobs writing one output_dir would resume from each other's
# checkpoints. So the successor is submitted immediately AND held on
# `afterany` against this job. First thing, held, every time.
#
# THE KILL SWITCH. A self-perpetuating chain that nobody can stop is a
# liability, and `scancel` on the running link only lets the successor
# through. Touch this file and the chain stops after the current link:
#
#   touch $EULLM_RUN_DIR/STOP_GRAZING
#
# It is checked before the successor is submitted, so it takes effect within
# one link rather than at the next walltime.
#
# BEFORE THE FIRST SUBMISSION — this is not optional. The existing 24 h chain
# must not have links that could run beside this one. Cancel the split arm's
# queued links, or leave only ones held behind a job this chain will not
# outlive. The guard below catches the mistake at runtime, but a job that
# starts and immediately exits still spent a window.
#
#   squeue --me -n eullm-p2b-split -o "%.10i %.2t %.18R"
#   scancel <the links you are replacing>
#
# Launch, once. It re-arms itself from there:
#
#   export EULLM_RUN_DIR=$WORK/eullm_runs/legal_it_v12
#   cd $EULLM_RUN_DIR
#   sbatch $WORK/eullm-v12/forge/scripts/leonardo/sbatch_phase2_split_graze.slurm
#
# Settable:
#   EULLM_GRAZE_GUARD   comma-separated job names that must not be RUNNING
#                       beside this one (default: the split arm's two names)
#   EULLM_GRAZE_END     yyyymmdd after which it stops re-arming
#                       (default: 20261102, the allocation's end)
#
#SBATCH --job-name=eullm-p2b-graze
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=450G
# TWO HOURS as the default, because the economics are much better — 22 fixed
# minutes of loss out of 120 is 82 % useful against 63 % at one hour — and
# because `save_steps: 300` in the split config yields two checkpoints at
# this length and none at one hour.
#
# But two hours is NOT the proven number. One hour is: a probe of this exact
# shape ran at 30 minutes and at 1 hour, both placed in 50 seconds, while 2,
# 3, 4 and 6 hours sat pending. If two hours turns out not to be placed
# either, the fallback is
#
#   sbatch --time=01:00:00 …        AND save_steps lowered to 100
#
# and the second half is not optional: at one hour the link yields ~291 steps
# against a first save at 300, so it would train, time out and hand the
# successor the identical starting point. The check further down warns about
# exactly this, and the successor inherits whatever --time this job ran with.
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

# Which checkout this job runs is decided by where THIS FILE came from, not by
# the submitting shell's environment — `sbatch` exports the submitter's
# environment and `env.sh` sets EULLM_REPO to whatever tree a run is pinned
# to, which has already silently run the wrong tree's code once.
SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null |
        sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "${EULLM_GRAZE_REPO:-}" ]; then
    EULLM_REPO="$EULLM_GRAZE_REPO"
elif [ -n "$SELF" ] && [ -f "$SELF" ]; then
    EULLM_REPO="$(cd "$(dirname "$SELF")/../../.." && pwd)"
else
    echo "[graze] cannot locate the repository — set EULLM_GRAZE_REPO" >&2
    exit 1
fi
export EULLM_REPO

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/env.sh"
cd "$EULLM_RUN_DIR" || exit 1

echo "[graze] repo   $EULLM_REPO"
echo "[graze] commit $(git -C "$EULLM_REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[graze] run    $EULLM_RUN_DIR"
echo "[graze] job    ${SLURM_JOB_ID:-?} on $(hostname) at $(date -Is)"

# GUARD: never train beside another link of this arm.
#
# Two jobs writing one output_dir resume from each other's checkpoints and
# the damage is silent — the loss curve looks plausible either way. This is
# the one failure mode of running a grazing chain next to a 24 h chain, and
# it is cheap to exclude.
GUARD="${EULLM_GRAZE_GUARD:-eullm-p2b-split,eullm-p2b-graze}"
others="$(squeue --me -h -t RUNNING -n "$GUARD" -o "%i" 2>/dev/null |
          grep -v "^${SLURM_JOB_ID:-0}$" || true)"
if [ -n "$others" ]; then
    echo "[graze] another link of this arm is RUNNING: $(echo "$others" | tr '\n' ' ')" >&2
    echo "[graze] refusing to train beside it — releasing this node now." >&2
    echo "[graze] The successor is still held behind this job, so the chain" >&2
    echo "[graze] continues once the other link ends." >&2
    # Still re-arm: the collision is transient, and exiting without a
    # successor would end the chain over a condition that clears by itself.
    ARM_ONLY=1
else
    ARM_ONLY=0
fi

# RE-ARM FIRST, because a link ends in TIMEOUT and nothing after the training
# command would run. Held on afterany against this job, so the successor
# cannot start beside us.
END="${EULLM_GRAZE_END:-20261102}"
if [ -f "$EULLM_RUN_DIR/STOP_GRAZING" ]; then
    echo "[graze] STOP_GRAZING present — this is the last link."
elif [ "$(date +%Y%m%d)" -ge "$END" ]; then
    echo "[graze] past $END — not re-arming."
elif [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[graze] no job id (interactive run?) — not re-arming."
else
    # Carry THIS job's walltime to the successor, rather than letting it fall
    # back to the #SBATCH default. The link length is the one thing we tune
    # from outside — `sbatch --time=01:00:00 …` when the measured threshold
    # moves — and a chain that silently reverted to two hours on its first
    # re-arm would undo the change without saying so.
    tl=("--time=${SLURM_JOB_TIME_LIMIT:-120}")
    if next=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
                     "${tl[@]}" "$SELF" 2>&1); then
        echo "[graze] next link $next at ${SLURM_JOB_TIME_LIMIT:-120}m, held until this one ends"
    else
        # Loud, and non-zero at the end, because a chain that quietly stops
        # re-arming looks exactly like a chain that is waiting.
        echo "[graze] FATAL: could not submit the successor: $next" >&2
        echo "[graze] The chain stops here. Re-submit this script by hand." >&2
        REARM_FAILED=1
    fi
fi

if [ "$ARM_ONLY" = "1" ]; then
    exit 0
fi

# Phase 1 must have produced the teacher adapter the config points at.
ADAPTER="./checkpoints/qwen3_30b_a3b_legal_it_continued_pt"
[ -f "$ADAPTER/adapter_config.json" ] || {
    echo "[graze] Phase-1 adapter not found at $EULLM_RUN_DIR/$ADAPTER" >&2
    echo "[graze]   ln -s \$WORK/eullm_runs/legal_it/$ADAPTER \\" >&2
    echo "[graze]         \$EULLM_RUN_DIR/checkpoints/" >&2
    exit 1
}

CFG="$EULLM_REPO/forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_split.yaml"

# SANITY CHECK ON save_steps, which is what makes a short link produce
# anything at all. At ~379 steps/hour and ~14 minutes of startup, a link of
# T hours trains for (T - 0.233) hours; if save_steps exceeds what that
# yields, the link runs, times out and hands the successor the same starting
# point. It is a complete waste that looks exactly like work.
#
# Warned rather than enforced: the arithmetic uses a measured throughput that
# can drift, and refusing to start over an estimate would be worse than
# running a link that saves once instead of twice.
SAVE_STEPS="$(grep -E '^save_steps:' "$CFG" | awk '{print $2}')"
LIMIT_MIN="$(( ${SLURM_JOB_TIME_LIMIT:-120} ))"
if [ -n "$SAVE_STEPS" ]; then
    train_min=$(( LIMIT_MIN - 14 ))
    steps=$(( train_min * 379 / 60 ))
    echo "[graze] walltime ${LIMIT_MIN}m, ~${train_min}m training, ~${steps} steps"
    if [ "$steps" -lt "$SAVE_STEPS" ]; then
        echo "[graze] WARNING: save_steps=$SAVE_STEPS but this link yields" >&2
        echo "[graze] only ~$steps steps — it would save NOTHING and the next" >&2
        echo "[graze] link would restart from the same checkpoint. Lower" >&2
        echo "[graze] save_steps or raise the walltime." >&2
    fi
fi

# shellcheck disable=SC1091
source "$EULLM_REPO/forge/scripts/leonardo/heartbeat.sh"
heartbeat_start

python "$EULLM_REPO/forge/scripts/check_training_env.py" \
    --offline --expect-gpus 4 --data-dir "$EULLM_DATA_DIR" \
    --tokenizer-model Qwen/Qwen3-4B-Base \
    --assert-frozen-base "$CFG" || exit 1

echo "[graze] teacher is BF16 — no quantized-logit validation needed"

bash "$EULLM_REPO/forge/scripts/distill.sh" "$CFG" "$EULLM_DATA_DIR"
rc=$?

# A TIMEOUT never reaches here; this covers the ways a link can end early.
echo "[graze] distill.sh exited $rc at $(date -Is)"
exit "$(( rc != 0 ? rc : ${REARM_FAILED:-0} ))"
