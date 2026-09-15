#!/usr/bin/env bash
# Tests for submit_chain.sh's dependency wiring.
#
# The thing under test is which --dependency each link gets, and getting it
# wrong does not fail loudly — it queues forever. Two ways, both seen:
#
#   * a --dependency in the extra args overrides the chain's own, so every
#     link waits on the same external job and all of them start at once,
#     resuming from the same checkpoint;
#   * `--after` means afterok, and a 24 h link ends in TIMEOUT by design, so
#     extending a chain with it leaves the extension in
#     DependencyNeverSatisfied until a human notices.
#
# Run: bash forge/tests/test_submit_chain.sh
#
# `sbatch` is replaced by a stub on PATH that records its arguments, so this
# needs no scheduler and runs anywhere.

set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts/leonardo" && pwd)/submit_chain.sh"
PASS=0
FAIL=0

setup() {
    WORK="$(mktemp -d)"
    mkdir -p "$WORK/bin"
    cat > "$WORK/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
# Records the full argument list, one submission per line, and hands back a
# job id that increments so the chain has something to depend on.
echo "$*" >> "$SBATCH_CALLS"
n=$(wc -l < "$SBATCH_CALLS")
echo "$((1000 + n))"
STUB
    chmod +x "$WORK/bin/sbatch"
    export SBATCH_CALLS="$WORK/calls.txt"
    : > "$SBATCH_CALLS"
    export PATH="$WORK/bin:$PATH"
    export SBATCH_ACCOUNT=test_account
    cd "$WORK"
    touch job.slurm
}

teardown() { cd /; rm -rf "$WORK"; }

check() {  # check <description> <expected> <actual>
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
        printf '  ok   %s\n' "$1"
    else
        FAIL=$((FAIL + 1))
        printf '  FAIL %s\n       expected: %s\n       actual:   %s\n' "$1" "$2" "$3"
    fi
}

deps() { grep -oE '\-\-dependency=[a-z]+:[0-9]+' "$SBATCH_CALLS" | tr '\n' ' '; }

# --- extending a chain: afterany, because TIMEOUT is the normal end --------
setup
bash "$SCRIPT" --after-any 57353624 job.slurm 3 >/dev/null 2>&1
check "--after-any holds the first link on afterany, then chains internally" \
      "--dependency=afterany:57353624 --dependency=afterany:1001 --dependency=afterany:1002 " \
      "$(deps)"
teardown

# --- a different phase: afterok, because a failure wrote no checkpoint -----
setup
bash "$SCRIPT" --after 56912622 job.slurm 2 >/dev/null 2>&1
check "--after still means afterok for a cross-phase hold" \
      "--dependency=afterok:56912622 --dependency=afterany:1001 " \
      "$(deps)"
teardown

# --- the two must not be confusable silently ------------------------------
setup
out="$(bash "$SCRIPT" --after 56912622 job.slurm 1 2>&1)"
case "$out" in
    *"--after-any"*) r=mentions ;;
    *) r=silent ;;
esac
check "--after says out loud that a TIMEOUT will not release it" mentions "$r"
teardown

# --- no hold at all: only the internal chain ------------------------------
setup
bash "$SCRIPT" job.slurm 3 >/dev/null 2>&1
check "without a hold the first link has no dependency" \
      "--dependency=afterany:1001 --dependency=afterany:1002 " \
      "$(deps)"
teardown

setup
bash "$SCRIPT" job.slurm >/dev/null 2>&1
check "a count of 1 is a plain submission" "" "$(deps)"
check "and it is exactly one submission" "1" "$(wc -l < "$SBATCH_CALLS" | tr -d ' ')"
teardown

# --- the foot-gun that started all this -----------------------------------
setup
bash "$SCRIPT" job.slurm 3 --dependency=afterok:999 >/dev/null 2>&1
check "a --dependency in the extra args is refused, not appended" \
      "0" "$(wc -l < "$SBATCH_CALLS" | tr -d ' ')"
teardown

setup
bash "$SCRIPT" --after-any 57353624 job.slurm 2 -d afterok:999 >/dev/null 2>&1
check "the short -d form is refused too" \
      "0" "$(wc -l < "$SBATCH_CALLS" | tr -d ' ')"
teardown

# --- bad input fails before anything is submitted -------------------------
setup
bash "$SCRIPT" --after-any 57353624 job.slurm two >/dev/null 2>&1
check "a non-numeric count submits nothing" \
      "0" "$(wc -l < "$SBATCH_CALLS" | tr -d ' ')"
teardown

setup
bash "$SCRIPT" --after-any 57353624 absent.slurm 2 >/dev/null 2>&1
check "a missing sbatch script submits nothing" \
      "0" "$(wc -l < "$SBATCH_CALLS" | tr -d ' ')"
teardown

# --- extra args do reach sbatch, since that is the point of them ----------
setup
bash "$SCRIPT" job.slurm 1 --qos=boost_qos_dbg >/dev/null 2>&1
case "$(cat "$SBATCH_CALLS")" in
    *--qos=boost_qos_dbg*) r=forwarded ;;
    *) r=dropped ;;
esac
check "harmless extra args are forwarded" forwarded "$r"
teardown

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
