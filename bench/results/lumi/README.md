# LUMI-G campaign results

`summary-YYYY-MM-DD.csv`: every point the campaign runner has measured on
LUMI-G (EHPC-DEV-2026D09-278) up to that date, one row per result, as
`bench/campaign/campaign.py collect` writes it on LUMI from the result files
in `$CAMPAIGN_DIR/results/<campaign>/<point>.<job>.json`. A copy kept here
because LUMI's `/scratch` is not permanent storage; the full result files
and server logs go to Zenodo at the end of the allocation.

The columns are `campaign.COLUMNS`: the point's parameters (model, GCDs,
replicas, batch, KV, split, `env`, round), what was measured (aggregate,
decode and prompt tokens/s, first token, load time, VRAM, device use,
neighbours) and its provenance (job, node, engine version, repository
revision). What the numbers mean, and which ones a decision rests on, is in
`docs/lumi/allocation-plan.md`.

To make one, on a LUMI login node:

```bash
eullm && $CAMPAIGN collect --queue "$CAMPAIGN_DIR"
```

then, from your own computer:

```bash
scp <user>@lumi.csc.fi:/scratch/project_465003366/<user>/campaign/results/summary.csv \
    summary-$(date +%F).csv
```
