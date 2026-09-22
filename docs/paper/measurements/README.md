# Scheduler measurements

Dated evidence for the Final Report's hardest question: an allocation sized
for two months of continuous use was not consumed — why.

"The cluster was busy" is an assertion. These are the numbers behind it,
recorded on the day, because neither source keeps history for long and one
keeps none at all.

## `queue_stats_*.json`

Where the calendar went, reconstructed from `sacct`: intervals of running,
idle-because-the-cluster-was-full, and idle-because-nothing-was-submitted —
the last being ours and reported as such. Written daily by
`sbatch_queue_stats.slurm`.

Frozen because `sacct` has a retention window of a few weeks. After that the
intervals stop being recoverable, and an interval is what a report can cite:
"the chain stalled from 2026-09-10 21:07 to 2026-09-11 04:13" is evidence,
"527 hours of queue wait" is not.

## `priority_*.json`

Why we were behind, from `sprio`, `sshare` and `sinfo`. These keep **no
history at all** — they report the current instant, so a number not written
down when it is read is simply gone.

Each snapshot holds our jobs' priority broken into its factors, the account's
fairshare, the distribution of competing priorities as quantiles (not a list
of other people's jobs), and the partition's node states.

The first one, 2026-09-22, is the shape of the argument:

* fairshare **0.751**, effective usage 0.000086 against 0.000208 of shares —
  the account was using **a third of what it was entitled to**
* priority **141,963** = QOS 120,000 + fairshare 18,783 + age 3,175
* the partition's pending priorities that moment: median well above ours, top
  **60,259,060** — 424x
* **one idle node** in `boost_usr_prod`, and it was not responding

An account under-using its share, sitting behind jobs with 1.5x to 424x its
priority, on a partition with nothing free, is a condition of the machine
rather than a failure of planning. That distinction is worth a lot in a
report, and it survives only if somebody recorded it on the day.


## How these get here

Both snapshots are written daily on the cluster by
`sbatch_queue_stats.slurm`, which re-arms itself until the allocation ends.
Nobody has to remember a command — which matters most for the priority
snapshot, since a day not captured is a day that cannot be reconstructed.

They land in the job's submit directory on `$WORK`, which has no backup, so
they are copied into git periodically. Weekly is enough: `sacct` keeps weeks
of history and the files accumulate safely in the meantime.

From a machine that can push (not the cluster — GitHub credentials do not
belong on a shared HPC home directory):

```sh
rsync -avP \
  <user>@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_1147/eullm_runs/qstats/*.json \
  docs/paper/measurements/
git add docs/paper/measurements && git commit -m "docs(paper): sync measurements"
```

The daily job runs from the checkout named by `EULLM_QSTATS_REPO`, which
defaults to `$WORK/eullm-v11` — the pilot tree, kept separate so the frozen
chain has its own. Pull there too, or the priority snapshot is skipped with a
line in the log saying so.
