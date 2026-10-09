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

A job takes points from the queue for as long as it runs, up to 48 hours,
with the binary it was submitted with. To queue a round for a new build
while jobs of the old one are still running, give the round an engine
label and submit the new build's jobs with the same label:

```bash
$CAMPAIGN plan --queue "$CAMPAIGN_DIR" --round r-next3 --engine-label next3 \
    tools/lumi/campaigns/c07-runtimes.json
EULLM_BIN=/scratch/.../eullm-target/next3/release/eullm EULLM_ENGINE_LABEL=next3 \
    bash tools/lumi/submit_campaign.sh 1 2
```

A point planned with a label runs only in a job with that label
(`EULLM_ENGINE_LABEL`), and one planned without a label only in a job
without one; runners older than the label (`RUNNER_VERSION` 3) leave
labelled points alone.

## Specs

JSON: defaults, then groups, each a fixed `set` and a cartesian product of
`axes`. See `tools/lumi/campaigns/` and the docstring of `spec.py` for every
field. The ones that matter:

| field | meaning |
|---|---|
| `kind` | `throughput` (the Leonardo method, repeated), `workload` (sustained load, graded), `decision` (`/v1/systemone`) or `finetune` (the engine's trainer) |
| `runtime` | `eullm` (default), `llama-server` or `ollama`: the same point served by another runtime, for comparison |
| `gcds` | devices the point uses: 1, 2, 4, 8 |
| `replica_gcds` | devices per server; `gcds / replica_gcds` servers (replicas) |
| `exclusive` | alone on the node: the control for neighbour interference |
| `cold` | drop the model's files from the node's page cache before the server starts, so the load reads the file system whatever ran before (`load.cache` is then `evicted`); needs the runner of 07-10-2026 |
| `batch`, `slot_ctx` / `ctx`, `kv` | the server's `--batch-size`, KV pool, `--cache-type-k/v` |
| `concurrency` | requests in flight; default fills every slot of every server |
| `prompt_tokens` | a synthetic prompt of about this many tokens, prefix cache defeated |
| `sets`, `min/max_duration_s` | workload: which sets, and how long; it stretches to the time free |
| `extra_args` | anything else for `eullm serve` (`--fit-strict` in the shipped specs) |
| `priority`, `est_s` | scheduling only; not part of a point's identity |

A `decision` point serves a decision model alone (`eullm serve
--decision-model <model> --decision-ctx <decision_ctx>`, a Jev-Style release
or a code-readout model) and sends `requests` decisions from `concurrency`
clients: synthetic ticket histories of `state_tokens` tokens with
`questions` questions each, in `decision_mode` (`shared_prefix`, `batched`,
`separate`). Each state is opened by its own line, so no request reuses the
state the server kept, and `distinct_states` of them are cycled, so each is
asked several times. It reports decisions per second, client latency p50/p95/
p99, the server's decode time and its wait (server time minus decode: the
engine runs one decision at a time per server), and consistency — whether a
state asked again, under concurrency, got the same answers to four decimals.

A point with `runtime: llama-server` or `ollama` runs the same
measurement against that server instead of `eullm serve`, with the same KV
pool, slots and cache types; `extra_args` are EuLLM's own and are not passed,
`runtime_args` are. llama-server gets the GGUF from the EuLLM store, Ollama a
model of the same name in its own store; both binaries come from
`LLAMA_SERVER_BIN` and `OLLAMA_BIN` (`tools/lumi/install_runtimes.sh`).
Requests are translated to llama-server's OpenAI endpoints and back.

Points with a new kind or a runtime carry `runner: 2`: a runner older than
that leaves them in the queue. A job keeps the code it started with, so
after a pull that adds a kind, the running jobs are cancelled (their
`afterany` successors start on the new code) before the new specs are planned.

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

No point starts on a device that already holds VRAM with nothing running on
it (more than 2 GiB): the device is set aside, the servers of this job that
no running point owns are killed (Ollama's model process can outlive
`ollama serve`), and the device comes back once it is empty. Every result
records what its devices held as it started (`vram_at_start_mib`, the
`vram0` column of `report`). Points measured before 08-10-2026 have no such
record; a peak VRAM far above the model's size with the device nearly idle
marks the ones that shared their GCD with a leftover.

## Outcomes

| state | when |
|---|---|
| done, `outcome: measured` | measured |
| done, `outcome: does-not-fit` | `--fit-strict` refused the load, or `eullm finetune` estimated more memory than the GCD has free (or ran out of it): the memory boundary, recorded |
| blocked | the model is not in the store (not in `f32/`, or not F32, for a finetune point), a set was not prefetched, or the engine predates `finetune` — `unblock` after fixing |
| failed | anything else, after one retry; the traceback is in `results/<campaign>/failed/` |

## Results

`results/<campaign>/<point>.<job>.json`, also printed as one
`BENCH_RESULT {...}` line, schema `eullm.bench/1`: the point's parameters,
provenance (engine version and binary hash, repository revision, ROCm
version, devices, cores, how many other points shared the node), load time
(cold or warm, and the file system the model was read from, `/scratch` or
`/flash`, links followed: `tools/lumi/stage_flash.sh`), and then

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
everything into `results/summary.csv`, and `campaign.py report [--campaign
c06-mtp]` prints a table per group: the parameters the group varies, the
metrics that fit its kind (tok/s, accuracy, decisions/s and latencies...)
averaged over the results of a configuration, and how many points failed or
did not fit.

## Tests

```bash
pytest -q bench/campaign        # against fake_eullm.py, no GPU
```
