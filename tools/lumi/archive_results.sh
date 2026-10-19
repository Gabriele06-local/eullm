#!/usr/bin/env bash
# Keep the campaign's results where LUMI keeps them: one compressed copy of
# every result file and of summary.csv in the project's own directory, since
# /scratch is not permanent storage. The server logs stay out (they are most
# of the size); they go to Zenodo with the rest at the end of the allocation.
#
#   bash tools/lumi/archive_results.sh
#
# Then, from your own computer, the summary for the repository
# (bench/results/lumi/README.md):
#   scp <user>@lumi.csc.fi:<CAMPAIGN_DIR>/results/summary.csv summary-$(date +%F).csv

set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"

DEST="${EULLM_ARCHIVE_DIR:-/projappl/${SBATCH_ACCOUNT:-project_465003366}/$USER/campaign-archive}"
mkdir -p "$DEST" || exit 1
$CAMPAIGN collect --queue "$CAMPAIGN_DIR" || exit 1
OUT="$DEST/results-$(date +%F).tar.gz"
# Write beside the destination, rename only on success: tar opens its output
# before it reads the tree, so writing $OUT directly truncated the previous
# archive for the day first, and a failure (disk full in the small projappl
# quota, an unreadable operand) left a partial file under the final name
# with the good one gone. Reruns on the same date are the documented
# workflow, so this was one failed tar away from losing the safety copy.
TMP="$(mktemp "$DEST/.results-$(date +%F).XXXXXX.tar.gz")" || exit 1
trap 'rm -f "$TMP"' EXIT
# Running jobs keep writing their node files while tar reads them: tar says
# "file changed as we read it" and exits with 1, the archive whole all the
# same. Only 2 and above are errors.
tar czf "$TMP" -C "$CAMPAIGN_DIR" --exclude=results/logs --warning=no-file-changed results
rc=$?
if [ "$rc" -gt 1 ]; then
    echo "[err] tar failed ($rc): $OUT is unchanged" >&2
    exit "$rc"
fi
mv -f "$TMP" "$OUT" || { echo "[err] cannot replace $OUT" >&2; exit 1; }
trap - EXIT
echo "$(tar tzf "$OUT" | wc -l) files → $OUT ($(du -h "$OUT" | cut -f1))"
ls -la --block-size=M "$DEST" | tail -3
