# Campaign runner — a whole GPU node, measuring all the time

`campaign.py` turns campaign specs into a queue of measurement points and
drains that queue on every device of a node, so a whole-node allocation is
never billed for idle GPUs. Written for LUMI-G (8 GCDs per node, EuroHPC
allocation EHPC-DEV-2026D09-278); site-neutral apart from the LUMI core
binding, so the same specs run on CUDA nodes for the cross-site comparison.

Plan and budget: [`docs/lumi/allocation-plan.md`](../../docs/lumi/allocation-plan.md).
Slurm side: `tools/lumi/campaign_setup.sh`, `submit_campaign.sh`,
`sbatch_campaign.slurm`, `status.sh`.

## On LUMI

```bash
export SBATCH_ACCOUNT=project_465003366 SALLOC_ACCOUNT=$SBATCH_ACCOUNT SLURM_ACCOUNT=$SBATCH_ACCOUNT
export EULLM_BIN=$PWD/eullm-rocm          # the gfx90a release binary

tmux new -s setup
bash tools/lumi/campaign_setup.sh         # login node: models, F32 models, sets, plan
bash tools/lumi/submit_campaign.sh 2 3    # two nodes, three 48 h jobs each
bash tools/lumi/status.sh                 # spend vs calendar, jobs, queue
```

A new engine build is a new round — the same specs measured again:

```bash
ROUND=v0.7.30 SKIP_PULLS=1 bash tools/lumi/campaign_setup.sh
```

## Specs

JSON: defaults, then groups, each a fixed `set` and a cartesian product of
`axes`. See `tools/lumi/campaigns/` and the docstring of `spec.py` for every
field. The ones that matter:

| field | meaning |
|---|---|
| `kind` | `throughput` (the Leonardo method, repeated), `workload` (sustained load, graded) or `finetune` (the engine's trainer) |
| `gcds` | devices the point uses: 1, 2, 4, 8 |
| `replica_gcds` | devices per server; `gcds / replica_gcds` servers (replicas) |
| `exclusive` | alone on the node: the control for neighbour interference |
| `batch`, `slot_ctx` / `ctx`, `kv` | the server's `--batch-size`, KV pool, `--cache-type-k/v` |
| `concurrency` | requests in flight; default fills every slot of every server |
| `prompt_tokens` | a synthetic prompt of about this many tokens, prefix cache defeated |
| `sets`, `min/max_duration_s` | workload: which sets, and how long; it stretches to the time free |
| `extra_args` | anything else for `eullm serve` (`--fit-strict` in the shipped specs) |
| `priority`, `est_s` | scheduling only; not part of a point's identity |

A `finetune` point runs `eullm finetune` on one GCD instead of a server. Its
own fields are the command's flags — `data`, `ft_ctx` (a multiple of 256),
`epochs`, `lr`, `optimizer` (`adamw`, `sgd`), `train_tensors` (glob patterns,
empty for every tensor), `limit_tokens`, `val_split` — and `keep_output`. Its
`model` is a file in `<queue>/f32/`, converted from the Hugging Face repo the
spec's top-level `f32` map names for it by `tools/lumi/make_f32_models.sh`
(the trainer needs F32 weights, which no store holds); its `data` is a file in
`<queue>/sets/` that `prefetch` writes (`finetune-gsm8k-train.jsonl`: GSM8K's
training split, never the test split the workload points grade on).

A point's id is a hash of what it measures: planning a spec twice adds
nothing, widening an axis adds only the new combinations, and `--round`
measures everything again under a new label.

## What a node does

`run` packs points onto aligned device groups — one GCD, a pair on one
MI250X module, half the node, all of it — and starts the next point the
moment devices free up. A wide point that cannot start yet reserves its
devices; narrower ones only use them meanwhile if they will be done by then
(EASY backfill). Workload points shrink or stretch between their minimum and
maximum duration to fill exactly the time that is free, including the end of
the job. Several jobs can drain one queue: a claim is an atomic rename.

At the walltime Slurm's SIGTERM stops the servers at once and puts the
interrupted points back; the next job takes them up.

## Outcomes

| state | when |
|---|---|
| done, `outcome: measured` | measured |
| done, `outcome: does-not-fit` | `--fit-strict` refused the load, or `eullm finetune` estimated more memory than the GCD has free (or ran out of it): the memory boundary, recorded |
| blocked | the model is not in the store (not in `f32/`, or not F32, for a finetune point), or a set was not prefetched — `unblock` after fixing |
| failed | anything else, after one retry; the traceback is in `results/<campaign>/failed/` |

## Results

`results/<campaign>/<point>.<job>.json`, also printed as one
`BENCH_RESULT {...}` line, schema `eullm.bench/1`: the point's parameters,
provenance (engine version and binary hash, repository revision, ROCm
version, devices, cores, how many other points shared the node), load time
(cold or warm), and then

- throughput: per repeat, aggregate tok/s over the batch wall clock, TTFT
  p50/p95, client-side prefill and decode rates, the server's own rates
  when it reports them; mean, stdev and CV across repeats;
- workload: accuracy per set (first pass), answer consistency of later
  passes against the first, a per-interval series of tok/s, latency and HBM,
  throughput drift from the first tenth to the last, and the graded answers
  in `<point>.<job>.answers.jsonl`;
- finetune: `eullm finetune`'s own report, schema `eullm.finetune/1` — loss,
  perplexity and accuracy on the held-out tokens before training and after
  each epoch, training tokens per second, trainable parameters, the memory
  estimate and the HBM in use at the end. The trained GGUF is deleted unless
  `keep_output`, then it stays in `results/logs/<job>/<point>.gguf`;
- per device: peak HBM and mean/max utilisation over the measured window.

Each job also leaves `<job>.summary.json` — the share of the job each GCD had
a point, which is the node-usage evidence for the final report — and
`<job>.node.jsonl`, the raw device samples. `campaign.py collect` flattens
everything into `results/summary.csv`.

## Tests

```bash
pytest -q bench/campaign        # against fake_eullm.py, no GPU
```
